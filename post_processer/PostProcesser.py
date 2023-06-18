from post_processer.pnpsolver import PnPSolver

from MyLib.posture import Posture
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math



from models.results import LandmarkDetectionResult, ImagePosture, ObjPosture
from models.utils import tensor_to_numpy
from scipy.optimize import linear_sum_assignment
class PostProcesser():
    '''
    后处理

    所有的变量名，不带_batch的表示是单个关键点对应的数据，例如坐标、热图
    带有_batch的是一整组，形状为 [N, ..., kpnum, ...]
    '''
    def __init__(self, pnpsolver:PnPSolver, out_bbox_threshold=1/3):
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
        self.pnpsolver = pnpsolver
        self.model_manager = self.pnpsolver.model_manager
        self.ldmk_out_threshold = out_bbox_threshold

    def parse_exclusively(self, rlt:LandmarkDetectionResult):
        '''
        只选取得分最高的
        '''
        conf = 0.5
        probs = tensor_to_numpy(rlt.landmarks_probs)[-1] #[tgt_num, landmark_num + 1]
        coords = tensor_to_numpy(rlt.landmarks)[-1] #[tgt_num, landmark_num + 1]
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

    def parse_by_voting(self, rlt:LandmarkDetectionResult):
        '''
        对于同一个landmark，选取多个tgt进行投票
        '''
        conf = 0.25
        probs = tensor_to_numpy(rlt.landmarks_probs)[-1]    #[tgt_num, landmark_num + 1]
        coords = tensor_to_numpy(rlt.landmarks)[-1]         #[tgt_num, landmark_num + 1]
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
        return np.sum(in_bbox) < int((1 - self.ldmk_out_threshold) * ldmks.shape[0])

    def process(self, image_list:list[np.ndarray], ldmk_detection:list[list[LandmarkDetectionResult]], mode = "e"):
        image_posture_list = []
        for bi, batch in enumerate(ldmk_detection):
            image_posture = ImagePosture(image_list[bi])
            for rlt in batch:
                if mode == "e":
                    ldmks, mask = self.parse_exclusively(rlt) # 独占式
                elif mode == "v":
                    ldmks, mask = self.parse_by_voting(rlt)   # 投票式
                else:
                    raise ValueError                    
                if np.sum(mask) < 8:
                    trans_vecs = None
                elif self.obj_out_bbox(ldmks, tensor_to_numpy(rlt.bbox)):
                    trans_vecs = None
                else:
                    # 计算姿态
                    points_3d = self.model_manager.get_ldmk_3d(rlt.class_id)
                    rvec, tvec = self.pnpsolver.solvepnp(ldmks, points_3d, mask)
                    trans_vecs = (rvec, tvec)
                image_posture.obj_list.append(ObjPosture(ldmks, 
                                                rlt.bbox_n, 
                                                rlt.class_id, 
                                                image_posture.image_size,
                                                trans_vecs))
            image_posture_list.append(image_posture)
        return image_posture_list
            

