from post_processer.model_manager import create_model_manager
import os
import json
import numpy as np
import cv2
from scipy.optimize import dual_annealing
TEST_RPnP = False

from ultralytics.yolo.utils import yaml_load

class PnPSolveMode(enumerate):
    SOLVEMODE_RPNP = 0
    SOLVEMODE_EPNP = cv2.SOLVEPNP_EPNP    

class PnPSolver():
    '''
    解PnP
    '''
    def __init__(self, cfg:str, image_resize = ((640, 480), (640, 480)), matrix_camera = None) -> None:
        '''
        brief
        -----
        PnP求解器，内置了各个物体的关键点3D坐标、包围盒3D坐标
        
        parameters
        -----
        image_resize: 经过网络预测的图像的大小会变化，这会导致内参的变换，需要传入该参数以确保内参的正确
        matrix_camera: 相机内参（原图）
        models_info_path: 模型信息路径
        keypoint_info_path: 关键点位置信息路径
        '''
        cfg_paras = yaml_load(cfg)
        self.model_manager = create_model_manager(cfg)
        if matrix_camera is None:
            # 读取内参(所有内参都一样)
            self.matrix_camera = np.loadtxt(cfg_paras["default_K"])
            # 由于图像已经缩放，内参也会变化
            self.matrix_camera = self.transform_K(self.matrix_camera, image_resize)
        self.distortion_coeffs = np.zeros((5,1))

    @staticmethod
    def transform_K(K, image_resize):
        # 图像缩放，内参也会变化
        resize_ratio = image_resize[1][0] / np.max(image_resize[0])
        M1 = np.array([[1,0,image_resize[1][0]/2],[0,1,image_resize[1][1]/2],[0, 0, 1]])
        M2 = np.array([[resize_ratio,0,0],[0,resize_ratio,0],[0, 0, 1]])
        # M2 = np.array([[1,0,0],[0,1,0],[0, 0, resize_ratio]])
        M3 = np.array([[1,0,-image_resize[0][0]/2],[0,1,-image_resize[0][1]/2],[0, 0, 1]])
        M = np.linalg.multi_dot((M1, M2, M3))
        return M.dot(K)

    def restore_points(self):
        pass

    def __filter_cam_parameter(self, K, D):
        #内参矩阵
        if isinstance(K, np.ndarray) and K.shape == (3,3):
            pass
        else:
            K = self.matrix_camera
        # 畸变矩阵
        if isinstance(D, np.ndarray) and D.shape == (5,1):
            pass
        else:
            D = self.distortion_coeffs
        return K, D

    def __filter_visib(self,
                        points      :np.ndarray, 
                        points_3d   :np.ndarray, 
                        points_visib:np.ndarray):
        if points_visib.size == 0:
            pass
        elif points_visib.shape[0] == points_3d.shape[0]:
            points_visib = points_visib.astype(np.bool_)
            points_3d = points_3d[points_visib]
            points = points[points_visib]
        else:
            raise ValueError("PnPSolver.__filter_visib: she shape of 'points_visib' is not matched with 'points' or 'points_3d'")

        return points, points_3d

    def solvepnp_simple(self,
                        points      :np.ndarray, 
                        class_id:   int):
        K,D = self.matrix_camera, self.distortion_coeffs
        points_3d = self.model_manager.get_ldmk_3d(class_id)
        success, vector_R, vector_T  = cv2.solvePnP(points_3d, points, K, D, flags=cv2.SOLVEPNP_EPNP)
        return vector_R, vector_T

    def solvepnp(self,
                        points      :np.ndarray, 
                        points_3d   :np.ndarray,
                        points_visib:np.ndarray = np.array([]),  # type: ignore
                        K           :np.ndarray = np.array([]),  # type: ignore
                        D           :np.ndarray = np.array([]),  # type: ignore
                        mode        :int    = PnPSolveMode.SOLVEMODE_EPNP):
        '''
        point_type: 'kp', 'bbox'
        '''
        assert mode == PnPSolveMode.SOLVEMODE_EPNP or mode == PnPSolveMode.SOLVEMODE_RPNP
        if TEST_RPnP:
            from RPnP import solveRPnP
        else:
            from MyLib.RPnP import solveRPnP
        K,D = self.__filter_cam_parameter(K, D)
        points, points_3d = self.__filter_visib(points, points_3d, points_visib)
        # 计算
        if mode == PnPSolveMode.SOLVEMODE_EPNP:
            success, vector_R, vector_T  = cv2.solvePnP(points_3d, points, K, D, flags=cv2.SOLVEPNP_EPNP)
        elif mode == PnPSolveMode.SOLVEMODE_RPNP:
            success, vector_R, vector_T  = solveRPnP(points_3d, points, K)
        else:
            vector_R, vector_T = np.zeros(3), np.zeros(3)
        return vector_R, vector_T

    def solvepnp_batch(self, 
                        points_batch      :np.ndarray,
                        points_3d_batch   :np.ndarray = np.array([]), # type: ignore
                        class_id_batch    :np.ndarray = np.array([]), # type: ignore
                        points_visib_batch:np.ndarray = np.array([]),  # type: ignore
                        K           :np.ndarray = np.array([]),  # type: ignore
                        D           :np.ndarray = np.array([]),  # type: ignore
                        point_type = 'kp'):
        '''
        brief
        -----
        solvepnp by a list of points
        (points will be resize to the origin size in this function)

        parameter
        -----
        * points_batch: np.ndarray [N, kpnum, 2]
        * points_3d_batch: np.ndarray [N, kpnum, 3], Optional, 'points_3d_batch' and 'class_id_batch' should not all be empty
        * class_id_batch: np.ndarray [N], Optional, 'points_3d_batch' and 'class_id_batch' should not all be empty.
                                To indicate which model's keypoints will be used as points_3d
        * points_visib_batch: np.ndarray [N, kpnum], Optional
        * point_type: 'kp', 'bbox'
        * use_table_assumption: bool, whether you want to assump all the object is on the table 
                                        to compensate the error caused by the lack of depth information
        '''
        target_num = len(points_batch)
        K,D = self.__filter_cam_parameter(K, D)
        results = []
        
        ### 输入合法性检测
        if points_3d_batch.size == 0 and class_id_batch.size==0:
            raise ValueError("PnPSolver.solvepnp_batch: 'points_3d_batch' and 'class_id_batch' should not all be None ")
        elif points_3d_batch.size == 0:
            if len(class_id_batch) != target_num:
                raise ValueError("PnPSolver.solvepnp_batch: 'class_id_batch' must have the same length as 'points_batch'")
            if point_type == 'kp':
                points_3d_batch = np.array([self.get_kp_3d(class_id_batch[x]) for x in range(target_num)])
            elif point_type == 'bbox':
                points_3d_batch = np.array([self.get_bbox_3d(class_id_batch[x]) for x in range(target_num)])
            else:
                raise ValueError("PnPSolver.solvepnp_batch: 'point_type' excepoint 'kp' or 'bbox', got {}".format(point_type))
        else: 
            if len(points_3d_batch) != target_num:
                raise ValueError("PnPSolver.solvepnp_batch: 'points_3d_batch' must have the same length as 'points_batch'")
        if points_visib_batch.size == 0:
            points_visib_batch = np.ones(points_batch.shape[:2], np.bool_)

        for points, points_3d, points_visib in zip(points_batch, points_3d_batch, points_visib_batch):
            points = utils.inv_resize_pointarray(points, 0.8, [(64, 64), (0, 0), (0, 0)], None) #还原要尺寸
            vector_R, vector_T = self.solvepnp(points[:, [1,0]], points_3d, points_visib, K, D)
            results.append([vector_R, vector_T])
        return results
        
    def calc_reproj(self, vector_R, vector_T,
                        points_3d   :np.ndarray, 
                        K           :np.ndarray = np.array([]), 
                        D           :np.ndarray = np.array([])) ->np.ndarray:
        '''
        point_type: "kp", "bbox", "customized"

        return 
        -----
        point2D
        '''
        K,D = self.__filter_cam_parameter(K, D)
        point2D, _ = cv2.projectPoints(points_3d, vector_R, vector_T, K, D)
        point2D = np.squeeze(point2D)
        return point2D

    def calc_bbox_by_kp_batch(self, keypoints, class_ids, points_visib):
        '''
        keypoints: [N, kpnum, 2]
        '''
        bbox_2d_list = []
        for kp, id, visib in zip(keypoints, class_ids, points_visib):
            kp = utils.inv_resize_pointarray(kp, 0.8, [(64, 64), (0, 0), (0, 0)], None)
            kp_3d = self.get_kp_3d(id)
            vecter_R, vector_T = self.solvepnp(kp[:, [1,0]], kp_3d, points_visib = visib)
            bbox_3d = self.get_bbox_3d(id)
            bbox_2d = self.calc_reproj(vecter_R, vector_T, bbox_3d)
            bbox_2d = bbox_2d[:, [1,0]]
            bbox_2d = utils.resize_pointarray(bbox_2d, 0.8, [(64, 64), (0, 0), (0, 0)], None)
            bbox_2d_list.append(bbox_2d)
        bboxes_2d = np.stack(bbox_2d_list, 0)
        return bboxes_2d

    def calc_bbox_by_vector_batch(self, vecters, class_ids):
        '''
        keypoints: [N, kpnum, 2]
        '''
        bbox_2d_list = []
        for (vecter_R, vector_T), id in zip(vecters, class_ids):
            vecter_R = np.array(vecter_R)
            vector_T = np.array(vector_T)
            bbox_3d = self.get_bbox_3d(id)
            bbox_2d = self.calc_reproj(vecter_R, vector_T, bbox_3d)
            bbox_2d = bbox_2d[:, [1,0]]
            bbox_2d = utils.resize_pointarray(bbox_2d, 0.8, [(64, 64), (0, 0), (0, 0)], None)
            bbox_2d_list.append(bbox_2d)
        bboxes_2d = np.stack(bbox_2d_list, 0)
        return bboxes_2d
