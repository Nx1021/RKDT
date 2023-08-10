from .pnpsolver import PnPSolver
from models.results import ImagePosture
from posture_6d.posture import Posture

from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou
from torch import Tensor
import torch
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import open3d as o3d
from typing import Iterable, Union
from models.utils import denormalize_bbox
import cv2

def is_close_to_integer(x, tolerance=1e-6):
    return np.isclose(x, np.round(x), atol=tolerance)

def match_roi(gt:ImagePosture, pred:ImagePosture):
    gt_image, gt_landmarks, gt_class_ids, gt_bboxes, gt_trans_vecs = gt.split()
    pred_image, pred_landmarks, pred_class_ids, pred_bboxes, pred_trans_vecs = pred.split()
    ### 匹配，类别必须一致，bbox用giou评估
    M = len(gt_class_ids)
    N = len(pred_class_ids)
    if N == 0 or M == 0:
        return []
    # bbox
    gt_bboxes_tensor = Tensor(np.array(gt_bboxes))
    pred_bboxes_tensor = torch.stack(pred_bboxes).to(gt_bboxes_tensor.device)
    cost_matrix_bbox = generalized_box_iou(pred_bboxes_tensor, gt_bboxes_tensor).numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix_bbox, maximize=True)

    cost_matrix_id = np.zeros((N, M), dtype=np.float32)  # 创建一个整数类型的全零矩阵
    for i in range(N):
        for j in range(M):
            if pred_class_ids[i] == gt_class_ids[j]:
                cost_matrix_id[i, j] = 2  # 不同元素的cost为1
    ###
    matched:list[tuple] = []
    for ri, ci in zip(row_ind, col_ind):
        if pred_class_ids[ri] == gt_class_ids[ci]:
            matched.append((gt_class_ids[ci], pred_trans_vecs[ri], gt_trans_vecs[ci]))
    return matched

class ErrorResult():
    def __init__(self, 
                 type:int, 
                 passed:bool,
                 error:float) -> None:
        self.type = type
        self.passed = passed
        self.error = error
        
    @property
    def type_name(self):
        if self.type == ErrorCalculator.REPROJ:
            return "2d reproj"
        if self.type == ErrorCalculator.ADD:
            return "ADD(s)"
        if self.type == ErrorCalculator._5CM5D:
            return "5cm5°"

    def print(self):
        print("{:<10}  error: {:<6.2}  passed: {}".format(self.type_name, self.error, self.passed))


class ErrorCalculator():
    ALL = 0
    REPROJ = 1
    ADD = 2
    _5CM5D = 3
    def __init__(self, pnpsolver:PnPSolver, class_num = 15) -> None:
        '''
        class_num: 包含背景在内的类数
        '''
        super().__init__()
        self.pnpsolver = pnpsolver
        self.model_manager = self.pnpsolver.mesh_manager
        # 读取内参(所有内参都一样)
        self.pass_record = np.zeros((4, class_num)) # 4行分别是总数、重投影、ADD、5cm5°
        self.error_record = np.zeros((4, class_num)) # 4行分别是总数、重投影error和、ADDerror和、5cm5°error和
        self.class_num = class_num

    def clear(self):
        self.pass_record[:] = 0
        self.error_record[:] = 0

    def get_metrics(self, type:int = 0):
        if type == ErrorCalculator.REPROJ:
            return self.reprojection_error
        elif type == ErrorCalculator.ADD:
            return self.add_error
        elif type == ErrorCalculator._5CM5D:
            return self._5cm5d_error
        else:
            raise ValueError(f"unexpected metrics type: {type}")

    def record(self, error_result:list[ErrorResult], class_id):
        self.pass_record[0, class_id] += 1
        self.error_record[0, class_id] += 1
        for er in error_result:
            self.pass_record[er.type, class_id] += er.passed
            self.error_record[er.type, class_id]  += er.error

    def calc_one_error(self, class_id:int, 
                       pred_vectors:np.ndarray, gt_vectors:np.ndarray, 
                       selected_metrics:Union[int, list[int]] = REPROJ) -> tuple[ErrorResult]:
        pred_vector_R, pred_vector_T = pred_vectors
        if pred_vector_R is None:
            return tuple([])
        gt_vector_R, gt_vector_T = gt_vectors

        assert isinstance(selected_metrics , (int, list))
        if selected_metrics == ErrorCalculator.ALL:
            selected_metrics = [ErrorCalculator.REPROJ, ErrorCalculator.ADD, ErrorCalculator._5CM5D]
        elif isinstance(selected_metrics, int):
            selected_metrics = [selected_metrics]

        error_result:list[ErrorResult] = []
        # print(gt_vector_T[2] / pred_vector_T[2])        
        for metrics_type in selected_metrics:
            metrics = self.get_metrics(metrics_type)
            # pred_vector_T = pred_vector_T * (gt_vector_T[2] / pred_vector_T[2])
            rlt, error = metrics(class_id, pred_vector_R, pred_vector_T, gt_vector_R, gt_vector_T)
            error_result.append(ErrorResult(metrics_type, rlt, error))

        self.record(error_result, class_id)
        return tuple(error_result)

    def print_result(self, print_rate=False):
        col_names= [str(x) for x in range(self.class_num)]
        row_names = ["total number", "reproj", "ADD(s)", "5cm5°"]

        print("accuracy:")
        if print_rate:
            with np.errstate(divide='ignore', invalid='ignore'):
                data = self.pass_record / self.pass_record[0,:]
        else:
            data = self.pass_record.astype(np.int32)
        df = pd.DataFrame(data, index=row_names, columns=col_names)
        print(df)

        print("error:")
        with np.errstate(divide='ignore', invalid='ignore'):
            data = self.error_record / self.error_record[0,:]
        df = pd.DataFrame(data, index=row_names, columns=col_names)
        print(df)
        print()

    def reprojection_error(self, class_id, pred_vector_R, pred_vector_T, gt_vector_R, gt_vector_T):
        '''
        * pred_point   [n,16]
        * gt_point     [n,16]
        * pred_class_ids  [n]
        * gt_class_ids    [n]
        * point_idx
        * 真值点默认使用所有点计算
        '''
        import cv2
        gt_point2D = self.pnpsolver.calc_reproj(gt_vector_R, gt_vector_T, 
                                    self.model_manager.get_model_pcd(class_id).astype(np.float32))
        gt_point2D = np.squeeze(gt_point2D)
        # 重投影
        pred_point2D = self.pnpsolver.calc_reproj(pred_vector_R, pred_vector_T,
                                    self.model_manager.get_model_pcd(class_id).astype(np.float32))
        pred_point2D = np.squeeze(pred_point2D) 
        # 重投影误差
        error = np.mean(np.sqrt(np.sum(np.square(gt_point2D-pred_point2D), axis=-1)))  # type: ignore
        if error < 5:
            return 1, error
        else:
            return 0, error

    def add_error(self, class_id, pred_vector_R, pred_vector_T, gt_vector_R, gt_vector_T):
        from sklearn.neighbors import NearestNeighbors
        pred_m2c    = Posture(rvec = pred_vector_R, tvec = pred_vector_T).trans_mat #预测的变换矩阵
        gt_m2c      = Posture(rvec = gt_vector_R,   tvec = gt_vector_T).trans_mat #真值的变换矩阵
        gt_pointcloud_OCS = self.model_manager.get_model_pcd(class_id).astype(np.float32) #物体坐标系下的点云
        homo_gt_pointcloud_OCS = np.concatenate((gt_pointcloud_OCS.T, np.ones((1, gt_pointcloud_OCS.T.shape[1]))))
        pred_pointcloud_OCS = np.dot(np.linalg.inv(pred_m2c), np.dot(gt_m2c, homo_gt_pointcloud_OCS)).T[:, :3]
        diameter = self.model_manager.get_model_diameter(class_id) #直径

        symmetries = self.model_manager.get_model_symmetries(class_id)
        if not symmetries:
            # ADD
            delta = pred_pointcloud_OCS - gt_pointcloud_OCS
            error = np.mean(np.linalg.norm(delta, axis= -1))
            if error < diameter * 0.1:
                return 1, error
            else:
                return 0, error
        else:
            # ADD-S
            samples = gt_pointcloud_OCS
            detect = pred_pointcloud_OCS
            neigh = NearestNeighbors(n_neighbors=1)
            neigh.fit(samples)
            a,d = neigh.kneighbors(detect,return_distance=True) 
            a.ravel() #看一下输出数据格式
            return 0, 0
            
    def _5cm5d_error(self, class_id, pred_vector_R, pred_vector_T, gt_vector_R, gt_vector_T):
        """
        计算6D姿态估计的5cm5°误差。

        Parameters:
            pred_vector_R (numpy.ndarray): 预测的旋转向量，形状为 (3,)。
            pred_vector_T (numpy.ndarray): 预测的平移向量，形状为 (3,)。
            gt_vector_R (numpy.ndarray): 真实的旋转向量，形状为 (3,)。
            gt_vector_T (numpy.ndarray): 真实的平移向量，形状为 (3,)

        Returns:
            error (bool): 是否满足5cm5°准则，满足为True，不满足为False。
        """
        # 将旋转向量转换为旋转矩阵
        pred_matrix_R = cv2.Rodrigues(pred_vector_R)[0]
        gt_matrix_R = cv2.Rodrigues(gt_vector_R)[0]

        # 计算平移向量之间的欧氏距离
        translation_distance = np.linalg.norm(pred_vector_T - gt_vector_T)

        # 计算旋转矩阵之间的角度差
        rotation_angle = np.arccos(np.clip((np.trace(np.dot(pred_matrix_R.T, gt_matrix_R)) - 1) / 2.0, -1, 1))
        rotation_angle_deg = np.degrees(rotation_angle)

        # 判断是否满足5cm5°准则
        if translation_distance <= 50 and rotation_angle_deg <= 5.0:
            rlt = True
        else:
            rlt = False

        return rlt, rotation_angle_deg
