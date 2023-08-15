from . import SCRIPT_DIR
from posture_6d.posture import Posture
from posture_6d.derive import PnPSolver
from posture_6d.metric import MetricCalculator
from posture_6d.data.mesh_manager import MeshManager
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import torch
import os

from models.results import LandmarkDetectionResult, ImagePosture, ObjPosture
from models.utils import tensor_to_numpy, normalize_points, denormalize_points
from utils.yaml import load_yaml
from scipy.optimize import linear_sum_assignment

def create_model_manager(cfg_file) -> MeshManager:
    cfg_paras = load_yaml(cfg_file)
    model_manager = MeshManager(os.path.join(SCRIPT_DIR, cfg_paras["models_dir"]),
                                 cfg_paras["pcd_models"])
    return model_manager

def create_pnpsolver(cfg_file) -> PnPSolver:
    cfg_paras = load_yaml(cfg_file)
    default_K = os.path.join(SCRIPT_DIR, cfg_paras["models_dir"], cfg_paras["default_K"])
    return PnPSolver(default_K)

class PostProcesser():
    '''
    后处理

    所有的变量名，不带_batch的表示是单个关键点对应的数据，例如坐标、热图
    带有_batch的是一整组，形状为 [N, ..., kpnum, ...]
    '''
    def __init__(self, pnpsolver:PnPSolver, mesh_manager:MeshManager, out_bbox_threshold=0.2):
        '''
        parameters
        -----
        heatmap_field_batch:    np.ndarray 整组区域热图[N, HM_SIZE, HM_SIZE, KEYPOINTS_NUM]
        heatmap_coord_batch:    np.ndarray 整组偏移热图[N, HM_SIZE, HM_SIZE, KEYPOINTS_NUM, 4]
        class_ids_batch:        np.ndarray 整组偏移热图[N]
        bboxes_batch:           np.ndarray 整组包围框[N, 4]
        
        return
        -----
        Description of the return
        '''
        self.pnpsolver:PnPSolver        = pnpsolver
        self.mesh_manager:MeshManager   = mesh_manager
        self.ldmk_out_upper:float       = out_bbox_threshold

        self.desktop_plane:np.ndarray = None

    def parse_exclusively(self, probs:torch.Tensor, coords:torch.Tensor):
        '''
        只选取得分最高的
        '''
        conf = 0.5
        probs = tensor_to_numpy(probs)[-1] #[tgt_num, landmark_num + 1]
        coords = tensor_to_numpy(coords)[-1] #[tgt_num, landmark_num + 1]
        landmark_num = probs.shape[-1] - 1
        # pred_class = np.argmax(probs, axis=-1) # [tgt_num]
        # 获取对应正例的tgt，用匈牙利算法匹配
        # positive = np.where(pred_class < landmark_num)[0]
        positive_probs = probs[:, :-1]
        row_ind, col_ind = linear_sum_assignment(positive_probs, maximize=True)
        # 过滤一次，最大小于conf的会被丢弃
        probs_filter = np.max(probs, axis = 0)[col_ind] > conf
        col_ind = col_ind[probs_filter]
        row_ind = row_ind[probs_filter]
        # 获取对应的坐标
        mask = np.zeros(landmark_num, dtype=np.bool8)
        ldmks = np.zeros((landmark_num, 2), dtype=np.float32)
        mask[col_ind] = True
        ldmks[col_ind] = coords[row_ind]
        return ldmks, mask

    def parse_by_voting(self, probs:torch.Tensor, coords:torch.Tensor):
        '''
        对于同一个landmark，选取多个tgt进行投票
        '''
        conf = 0.25
        probs = tensor_to_numpy(probs)[-1]    #[tgt_num, landmark_num + 1]
        coords = tensor_to_numpy(coords)[-1]         #[tgt_num, landmark_num + 1]
        landmark_num = probs.shape[-1] - 1

        mask = np.zeros(landmark_num, dtype=np.bool8)
        ldmks = np.zeros((landmark_num, 2), dtype=np.float32)
        for li in range(landmark_num):
            ok_i = np.where(probs[:, li] > conf)[0]
            if len(ok_i > 0):
                mask[li] = True
                ldmks[li, :] = np.sum(coords[ok_i].T * probs[ok_i, li], axis=-1) / np.sum(probs[ok_i, li])
        return ldmks, mask

    def obj_out_bbox(self, ldmks, bbox):
        in_bbox = ( (ldmks[..., 0] >= bbox[0]) &
                    (ldmks[..., 0] <= bbox[2]) &
                    (ldmks[..., 1] >= bbox[1]) &
                    (ldmks[..., 1] <= bbox[3]))
        return np.sum(in_bbox) < int((1 - self.ldmk_out_upper) * ldmks.shape[0])

    # 沿着物体中心与相机坐标系原点的连线移动物体
    @staticmethod
    def move_obj_by_optical_link(posture:Posture, move_dist) -> Posture:
        optical_link = posture.tvec
        optical_link = optical_link / np.linalg.norm(optical_link)# normalize
        new_t = posture.tvec + optical_link * move_dist
        return Posture(rvec=posture.rvec, tvec=new_t)

    def desktop_assumption(self, trans_vecs, points_3d):
        '''
        桌面假设，假设物体的最低点处于桌面上，重新计算物体的trans_vecs
        '''
        if self.desktop_plane is None:
            return trans_vecs

        orig_posture = Posture(rvec=trans_vecs[0], tvec=trans_vecs[1])

        # 获取桌面法向量和原点到桌面的距离
        normal_vector = self.desktop_plane[:3]
        distance = self.desktop_plane[3]

        # 计算物体在相机坐标系下的坐标点云
        points_camera = orig_posture * points_3d

        # 计算物体最低点的索引
        lowest_point_index = np.argmax(points_camera[:, 2])

        # 获取物体最低点在相机坐标系下的坐标
        lowest_point_camera = points_camera[lowest_point_index]

        # 计算物体最低点到桌面的距离
        distance_to_desktop = np.dot(lowest_point_camera, normal_vector) - distance

        # 计算平移向量
        move_vec = np.mean(points_3d, axis=0)# 平移方向
        normed_move_vec = move_vec / np.linalg.norm(move_vec)
        translation_vector = -distance_to_desktop * normed_move_vec / normed_move_vec[2]

        # 更新平移向量
        updated_tvec = trans_vecs[1] + translation_vector

        return (trans_vecs[0], updated_tvec)

    def bbox_area_assumption(self, posture:Posture, class_id, bbox):
        # 假设预测的物体的投影面积和真实的投影面积相同，重新计算物体的trans_vecs
        points = self.mesh_manager.get_model_pcd(class_id)
        reproj = self.pnpsolver.calc_reproj(points, posture = posture)#[N, 2]
        reproj_bbox = (np.min(reproj[:, 0]), np.min(reproj[:, 1]), np.max(reproj[:, 0]), np.max(reproj[:, 1]))
        reproj_bbox_area = (reproj_bbox[2] - reproj_bbox[0]) * (reproj_bbox[3] - reproj_bbox[1])
        bbox_area = tensor_to_numpy((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        scale = np.sqrt(bbox_area / reproj_bbox_area)
        move_dist = -(1 - 1/scale) * posture.tvec[2]
        new_posture = self.move_obj_by_optical_link(posture, move_dist)
        return new_posture

    def process_one(self, pred_probs, pred_ldmks, class_id, bbox, mode = 'v', intr_M = None):
        if mode == "e":
            ldmks, mask = self.parse_exclusively(pred_probs, pred_ldmks) # 独占式
        elif mode == "v":
            ldmks, mask = self.parse_by_voting(pred_probs, pred_ldmks)   # 投票式
        else:
            raise ValueError                    
        if np.sum(mask) < 8:
            posture = None
        elif self.obj_out_bbox(ldmks, tensor_to_numpy(bbox)):
            posture = None
        else:
            # 计算姿态
            points_3d = self.mesh_manager.get_ldmk_3d(class_id)
            posture = self.pnpsolver.solvepnp(ldmks, points_3d, mask, K=intr_M, return_posture=True)
            posture = self.bbox_area_assumption(posture, class_id, bbox)
        return ldmks, posture

    def process(self, image_list:list[np.ndarray], ldmk_detection:list[list[LandmarkDetectionResult]], mode = "v"):
        image_posture_list = []
        for bi, batch in enumerate(ldmk_detection):
            image_posture = ImagePosture(image_list[bi])
            for rlt in batch:
                ldmks, posture = self.process_one(rlt.landmarks_probs, rlt.landmarks, rlt.class_id, rlt.bbox, mode)
                image_posture.obj_list.append(ObjPosture(ldmks, 
                                                rlt.bbox_n, 
                                                rlt.class_id, 
                                                image_posture.image_size,
                                                posture)) #TODO
            image_posture_list.append(image_posture)
        return image_posture_list
            

