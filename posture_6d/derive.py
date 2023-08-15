'''
derive data from base data
'''

import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
import os
from typing import Union

from .data.mesh_manager import MeshMeta
from .posture import Posture
from .intr import CameraIntr

def inv_proj(intr_M:np.ndarray, pixels:np.ndarray, depth:float):
    '''
    brief
    -----
    reproj 2d pixels to 3d sapce on a specified depth
    
    parameters
    -----
    * intr_M: [3, 3]
    * pixels: [n, (x,y)] 
    * depth: float
    '''
    CAM_FX, CAM_FY, CAM_CX, CAM_CY = CameraIntr.parse_intr_matrix(intr_M)
    px = (pixels[:, 0] - CAM_CX) * depth / CAM_FX
    py = (pixels[:, 1] - CAM_CY) * depth / CAM_FY
    pz = np.sqrt(np.square(depth) - np.square(px) - np.square(py))
    return np.array([px, py, pz]).T # [N,3]

def calc_masks(mesh_metas:list[MeshMeta], postures:list[Posture], intrinsics:CameraIntr, 
            ignore_depth = False, tri_mode = True, reserve_empty = True):
    '''
    parameters
    -----

    return
    -----
    masks: list[cv2.Mat]
    visib_fract: list[float]
    '''
    def draw_one_mask(meta:MeshMeta, posture:Posture):
        CAM_WID, CAM_HGT    = intrinsics.CAM_WID, intrinsics.CAM_HGT # 重投影到的深度图尺寸
        EPS = intrinsics.EPS
        MAX_DEPTH = intrinsics.MAX_DEPTH
        pc = meta.points_array #[N, 3]
        triangles = meta.tris_array #[T, 3]
        pc = posture * pc #变换
        z = pc[:, 2]
        # 点云反向映射到像素坐标位置
        orig_proj = intrinsics * pc
        u, v = (orig_proj).T
        # 滤除镜头后方的点、滤除超出图像尺寸的无效像素
        valid = np.bitwise_and(np.bitwise_and((u >= 0), (u < CAM_WID)),
                            np.bitwise_and((v >= 0), (v < CAM_HGT)))
        mask = np.zeros((CAM_HGT, CAM_WID), np.uint8) #掩膜，物体表面的完全投影

        if np.sum(valid) != 0:
            u, v, z = u[valid], v[valid], z[valid]
            pts = np.array([u, v]).T.astype(np.int32) #[P, 2]
            new_pc_index = -np.ones(valid.size).astype(np.int32)
            new_pc_index[valid] = np.arange(np.sum(valid)).astype(np.int32)
            ### 绘制掩膜
            if not tri_mode:
                mask[v, u] = 255
                kernel_size = max(int((u.max() - u.min()) * (v.max() - v.min()) / u.shape[0]), 3)
                kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, 
                                        cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size)),
                                        iterations=2)
            else:
                valid_tri = np.all(valid[triangles], axis = -1)
                triangles = new_pc_index[triangles]
                triangles = triangles[valid_tri]
                # 由于点的数量有限，直接投影会形成空洞，所有将三角面投影和点投影结合
                # 先用三角形生成掩膜
                tri_pts = pts[triangles] #[T, 3, 2]
                for i, t in enumerate(tri_pts):
                    mask = cv2.fillPoly(mask, [t], 255) # 必须以循环来画，cv2.fillPoly一起画会有部分三角丢失，原因尚不清楚
            mask[mask > 0] = 1
            ### 计算深度
            if ignore_depth:
                min_depth = np.full((CAM_HGT, CAM_WID), MAX_DEPTH) 
                min_depth[mask.astype(np.bool_)] = np.mean(z)
            else:
                # 用点生成深度
                depth = np.full((CAM_HGT, CAM_WID), MAX_DEPTH) 
                for i, p in enumerate(pts):
                    z_value = z[i]
                    depth[p[1], p[0]] = min(z_value, depth[p[1], p[0]])
                # 对掩膜上的点进行深度值的最小值滤波
                iter_num = min(int(np.ceil(np.sum(mask) / pts.shape[0]/255)), 4)
                min_depth = cv2.morphologyEx(mask * depth, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations= iter_num)
                # 膨胀用来填充边缘点
                dilate = cv2.morphologyEx(min_depth, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations= iter_num + 1)
                min_depth[min_depth == 0] = dilate[min_depth == 0]
                min_depth[mask == 0] = MAX_DEPTH
        else:
            min_depth = np.full((CAM_HGT, CAM_WID), MAX_DEPTH)
        return min_depth, orig_proj

    depth_images = []
    orig_proj_list = []
    for meta, posture in zip(mesh_metas, postures):
        min_depth, orig_proj = draw_one_mask(meta, posture)
        depth_images.append(min_depth)
        orig_proj_list.append(orig_proj)

    ### 计算不同mask之间的遮挡
    masks = []
    if len(mesh_metas) == 1:
        mask = np.zeros(depth_images[0].shape, np.uint8)
        mask[min_depth < min_depth.max()] = 255
        masks.append(mask)
    else:
        depth_tensor = np.array(depth_images) #[N, H, W]
        scene_mask = np.argmin(depth_tensor, axis=0)
        back_ground = np.all(depth_tensor == intrinsics.MAX_DEPTH, axis=0)
        scene_mask[back_ground] = -1
        if reserve_empty:
            label_range = depth_tensor.shape[0] - 1
        else:
            label_range = scene_mask.max()
        for label in range(label_range+1):
            mask = np.zeros(scene_mask.shape, np.uint8)
            mask[scene_mask == label] = 255
            masks.append(mask)

    ### 计算可见比例
    visib_fract:list[float] = []
    for mask, orig_proj, meta in zip(masks, orig_proj_list, mesh_metas):
        orig_proj = orig_proj.astype(np.int32)
        orig_proj = intrinsics.filter_in_view(orig_proj)
        vf = np.sum(mask[orig_proj[:,1], orig_proj[:,0]].astype(np.bool8)) / meta.points_array.shape[0]
        visib_fract.append(vf)
   
    return masks, visib_fract

def calc_landmarks_proj(mesh_meta:MeshMeta, postures:Posture, intrinsics:CameraIntr):
    landmarks = mesh_meta.ldmk_3d
    return intrinsics * (postures * landmarks)

def calc_bbox_3d_proj(mesh_meta:MeshMeta, postures:Posture, intrinsics:CameraIntr):
    bbox_3d = mesh_meta.bbox_3d
    return intrinsics * (postures * bbox_3d)

class PnPSolver():
    '''
    解PnP
    '''
    def __init__(self, matrix_camera:Union[np.ndarray, str, None] = None, distortion_coeffs:Union[np.ndarray, None] = None) -> None:
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
        self.init_default_intr(matrix_camera)
        self.init_distortion_coeffs(distortion_coeffs)

    def init_default_intr(self, matrix_camera:Union[np.ndarray, str, None] = None):
        if isinstance(matrix_camera, str):
            # read from file
            if os.path.splitext(matrix_camera)[-1] == ".txt":
                K = np.loadtxt(matrix_camera)
                self.intr = CameraIntr(K)
            elif os.path.splitext(matrix_camera)[-1] == ".json":
                self.intr = CameraIntr.from_json(matrix_camera)
            elif os.path.splitext(matrix_camera)[-1] == ".npy":
                self.intr = CameraIntr(np.load(matrix_camera))
            else:
                raise ValueError("Unknown file type: {}, '.txt', '.json', '.npy' is expected.".format(os.path.splitext(matrix_camera)[-1]))
        elif isinstance(matrix_camera, np.ndarray):
            self.intr = CameraIntr(matrix_camera)
        elif matrix_camera is None:
            self.intr = None
        else:
            raise TypeError("matrix_camera should be str or np.ndarray, but got {}".format(type(matrix_camera)))        

    def init_distortion_coeffs(self, distortion_coeffs:Union[np.ndarray, None] = None):
        if distortion_coeffs is None:
            self.distortion_coeffs = np.zeros((5,1))
        elif isinstance(distortion_coeffs, np.ndarray):
            self.distortion_coeffs = distortion_coeffs
        else:
            raise ValueError("distortion_coeffs should be None or np.ndarray, but got {}".format(type(distortion_coeffs)))

    @staticmethod
    def get_zoomed_K(K:np.ndarray, image_resize:tuple[tuple[int, int], tuple[int, int]]) -> np.ndarray:
        '''
        K: np.ndarray, 3x3
        image_resize: ((w, h), (w, h)), eg.((640, 480), (640, 480))
        '''
        # the intrinsics matrix should be scaled accordingly
        resize_ratio = image_resize[1][0] / np.max(image_resize[0])
        M1 = np.array([[1,0,image_resize[1][0]/2],[0,1,image_resize[1][1]/2],[0, 0, 1]])
        M2 = np.array([[resize_ratio,0,0],[0,resize_ratio,0],[0, 0, 1]])
        # M2 = np.array([[1,0,0],[0,1,0],[0, 0, resize_ratio]])
        M3 = np.array([[1,0,-image_resize[0][0]/2],[0,1,-image_resize[0][1]/2],[0, 0, 1]])
        M:np.ndarray = np.linalg.multi_dot((M1, M2, M3))
        return M.dot(K)

    def __filter_cam_parameter(self, 
                               K:Union[np.ndarray, CameraIntr, None], 
                               D:Union[np.ndarray, None]):
        # intrinsics
        assert isinstance(K, (np.ndarray, CameraIntr, type(None))), "K should be np.ndarray or CameraIntr, but got {}".format(type(K))
        assert isinstance(D, (np.ndarray, type(None))), "D should be np.ndarray, but got {}".format(type(D))
        if isinstance(K, np.ndarray):
            assert K.shape == (3,3)
        elif isinstance(K, CameraIntr):
            K = K.intr_M
        else:
            assert self.intr is not None, "K is None and self.intr is None, please at least set one of them."
            K = self.intr.intr_M
        
        # distortion_coeffs
        if isinstance(D, np.ndarray):
            assert D.shape == (5,1)
        else:
            D = self.distortion_coeffs
        return K, D

    def __filter_visib(self,
                        points      :np.ndarray, 
                        points_3d   :np.ndarray, 
                        points_visib:Union[np.ndarray, None] == None):
        if points_visib is None or len(points_visib) == 0:
            pass
        elif points_visib.shape[0] == points_3d.shape[0]:
            points_visib = points_visib.astype(np.bool_)
            points_3d = points_3d[points_visib]
            points = points[points_visib]
        else:
            raise ValueError("PnPSolver.__filter_visib: she shape of 'points_visib' is not matched with 'points' or 'points_3d'")

        return points, points_3d

    def solvepnp(self,
                        points      :np.ndarray, 
                        points_3d   :np.ndarray,
                        points_visib:np.ndarray = None,  
                        K           :Union[np.ndarray, CameraIntr] = None,  
                        D           :np.ndarray = None,
                        *, return_posture = False):
        '''
        point_type: 'kp', 'bbox'
        '''
        K,D = self.__filter_cam_parameter(K, D)
        points, points_3d = self.__filter_visib(points, points_3d, points_visib)
        # 计算
        success, vector_R, vector_T  = cv2.solvePnP(points_3d, points, K, D, flags=cv2.SOLVEPNP_EPNP)
        if return_posture:
            return Posture(rvec=vector_R, tvec=vector_T)
        else:
            return vector_R, vector_T
        
    def calc_reproj(self, points_3d   :np.ndarray, 
                        vector_R    :np.ndarray = None, 
                        vector_T    :np.ndarray = None,
                        posture     :Posture    = None,
                        K           :np.ndarray = None, 
                        D           :np.ndarray = None) ->np.ndarray:
        '''
        * points_3d: [N,3]
        * vector_R: [3,1] or [3]
        * vector_T: [3,1] or [3]
        * posture: Posture
        * K: [3,3]
        * D: [5,1]

        return 
        -----
        * point2D: [N,2]
        '''
        assert (vector_R is not None and vector_T is not None) or posture is not None, "vector_R and vector_T or posture should be provided."
        if posture is not None:
            assert isinstance(posture, Posture), "posture should be Posture, but got {}".format(type(posture))
            vector_R = posture.rvec
            vector_T = posture.tvec
        else:
            assert isinstance(vector_R, np.ndarray) and isinstance(vector_T, np.ndarray) and\
                vector_R.size == 3 and vector_T.size == 3
        K,D = self.__filter_cam_parameter(K, D)
        point2D, _ = cv2.projectPoints(points_3d, vector_R, vector_T, K, D)
        point2D = np.squeeze(point2D, 1)
        return point2D
