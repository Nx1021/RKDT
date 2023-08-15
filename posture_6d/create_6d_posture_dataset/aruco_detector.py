# -- coding: utf-8 --
"""
compute_gt_poses.py
---------------

Main Function for registering (aligning) colored point clouds with ICP/aruco marker 
matching as well as pose graph optimizating, output transforms.npy in each directory

"""
from open3d import *
import numpy as np
import cv2
import os
import glob
# from utils.camera import *
# from registration import icp, feature_registration, match_ransac, rigid_transform_3D
# from open3d import pipelines
# registration = pipelines.registration
from tqdm import trange
import time
import sys
# from config.registrationParameters import *
import json
import png
# from excude_pipeline import *
from MyLib.posture_6d.posture import Posture
# from utils.plane import findplane, fitplane, point_to_plane, findplane_wo_outliers
import matplotlib.pyplot as plt
from sko.GA import GA

import cv2.aruco as aruco
from . import JsonIO, Posture, CameraIntr
from typing import Union

from .utils.camera_sys import convert_depth_frame_to_pointcloud
from .utils.plane import findplane_wo_outliers
from .utils import homo_pad




class ArucoDetector():
     '''
     detect aruco marker and compute its pose
     '''
     def __init__(self, aruco_floor:Union[str, dict, np.ndarray], *, long_side_real_size = None) -> None:
          if isinstance(aruco_floor, str):
               if aruco_floor :    
                    suffix = aruco_floor.split('.')[-1]
                    if suffix == 'json':
                         self.C0_aruco_3d_dict = JsonIO.load_json(aruco_floor)
                    elif suffix == "png" or suffix == "bmp":
                         assert long_side_real_size is not None, "long_side_real_size must be specified"                         
                         self.C0_aruco_3d_dict = self.get_C0_aruco_3d_dict_from_image(aruco_floor, long_side_real_size=long_side_real_size)
                         raise ValueError("long_side_real_size must be specified")
               else:
                    pass
          elif isinstance(aruco_floor, dict):
               # check the dict 
               for k, v in aruco_floor.items():
                    assert isinstance(k, int), "the key of aruco_floor must be int"
                    assert isinstance(v, np.ndarray), "the value of aruco_floor must be ndarray"
                    assert v.shape[0] == 4 and v.shape[1] == 3, "the shape of aruco_floor must be [4, 3]"
                    assert np.issubdtype(v.dtype, np.floating)
               self.C0_aruco_3d_dict = aruco_floor
          elif isinstance(aruco, np.ndarray):
               assert long_side_real_size is not None, "long_side_real_size must be specified"
               self.C0_aruco_3d_dict = self.get_C0_aruco_3d_dict_from_image(aruco_floor, long_side_real_size=long_side_real_size)
          self.verify_tol = 0.01
     
     @staticmethod
     def detect_aruco_2d(image:np.ndarray)->tuple[np.ndarray, np.ndarray, np.ndarray]:
          '''
          return
          ----
          corners_src: np.ndarray, [N, 4, 2]
          ids: np.ndarray, [N]
          rejectedImgPoints: np.ndarray, [N, 1, 2]
          '''
          # if the image has 3-channels, convert it to gray scale
          if len(image.shape) == 3:
               gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
          aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
          parameters = aruco.DetectorParameters_create()
          corners_src, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)   
          if len(corners_src) > 0:
               corners_src = np.squeeze(np.array(corners_src), axis=1) # [N, 4, 2]
               ids = np.array(ids).squeeze(axis=-1) # [N]
          else:
               corners_src = np.zeros((0,4,2)) # [0, 4, 2]
               ids = np.array([]) # [0]
          return corners_src, ids, rejectedImgPoints

     @staticmethod
     def detect_aruco_3d(color, depth, camera_intrinsics:Union[np.ndarray, CameraIntr], corners_src = None, ids = None):
          '''
          brief
          ----
          detect aruco marker in 2d image and compute its 3d pose by depth image

          params
          ----
          color: np.ndarray, [H, W, 3]
          depth: np.ndarray, [H, W] np.uint16
          camera_intrinsics: np.ndarray, [3, 3] or CameraIntr
          corners_src: Optional, np.ndarray, [N, 4, 2], to avoid repeatly detect aruco marker if detect_aruco_2d has been called in context
          ids: Optional, np.ndarray, [N], to avoid repeatly detect aruco marker if detect_aruco_2d has been called in context
          '''
          camera_intrinsics = CameraIntr(camera_intrinsics)
          if corners_src is None or ids is None:
               corners_src, ids, rejectedImgPoints = ArucoDetector.detect_aruco_2d(color)
          depth = convert_depth_frame_to_pointcloud(depth, camera_intrinsics)
          scene_aruco = np.round(corners_src).astype(np.int32)
          scene_aruco_3d = depth[scene_aruco[:, :, 1], scene_aruco[:, :, 0]]   #[N,4,3]
          return scene_aruco_3d, ids

     @staticmethod
     def get_C0_aruco_3d_dict_from_image(image:Union[str, np.ndarray], long_side_real_size:float):
          '''
          params
          ----
          image: np.ndarray, [H, W, 3]
          long_side_real_size: float, the real size of the long side of the image, unit: mm
          '''
          assert isinstance(image, str) or isinstance(image, np.ndarray), "image must be a path or np.ndarray"
          if isinstance(image, str):
               assert os.path.exists(image), "image not exists"
               af_image = cv2.imread(image) # aruco floor image
          af_image:np.ndarray = np.pad(af_image,[(100,100),(100,100)],'constant',constant_values=255)
          # 粗测
          zoom_1 = 600 / np.max(af_image.shape)
          coarse_arcuo_floor = cv2.resize(af_image, (-1,-1), fx=zoom_1, fy=zoom_1)
          corners_src, ids, rejectedImgPoints = ArucoDetector.detect_aruco_2d(coarse_arcuo_floor)
          
          # 精测
          refine_corners_src_list = []
          for cs in corners_src:
               refine_corner_list = []
               for corner in cs: #[2]
                    crop_rect_min = np.clip(corner - 5, [0,0], coarse_arcuo_floor.shape[::-1]) 
                    crop_rect_max = np.clip(corner + 5, [0,0], coarse_arcuo_floor.shape[::-1]) 
                    crop_rect_min = np.round(crop_rect_min / zoom_1).astype(np.int32)
                    crop_rect_max = np.round(crop_rect_max / zoom_1).astype(np.int32)
                    local = af_image[crop_rect_min[1]: crop_rect_max[1], crop_rect_min[0]: crop_rect_max[0]]
                    refine_corner = np.squeeze(cv2.goodFeaturesToTrack(local, 1, 0.05, 0)) + crop_rect_min
                    refine_corner_list.append(refine_corner)
               refine_corners_src_list.append(refine_corner_list)
          refine_corners_src = np.array(refine_corners_src_list) # [N, 4, 2]

          refine_corners_src = np.reshape(refine_corners_src, (-1, 2)) # [N*4, 2]
          ### map refine_corners_src to 0~long_side_real_size
          refine_corners_src = refine_corners_src - np.min(refine_corners_src, axis=0)
          refine_corners_src = refine_corners_src / np.max(af_image.shape) * long_side_real_size
          # stack z axis, the value of z is 0
          refine_corners_src = np.hstack((refine_corners_src, np.zeros((refine_corners_src.shape[0], 1)))) # [N*4, 3]
          refine_corners_src = np.reshape(refine_corners_src, (-1, 4, 3)) # [N, 4, (y, x, z)]
          # swap axis
          refine_corners_src_3d = refine_corners_src[:,:,(1,0,2)] #[N, 4, (x, y, z)]
          # zip dict
          C0_aruco_3d_dict = dict(zip(ids.flatten().tolist(), refine_corners_src_3d))

          return C0_aruco_3d_dict
          # min_pos = np.min(refine_corners_src, axis=0) #最小范围
          # refine_corners_src = refine_corners_src - min_pos
          # max_pos = np.max(refine_corners_src, axis=0) #最大范围
          # refine_corners_src = refine_corners_src * long_side_real_size / np.max(max_pos)
          # # refine_corners_src = np.swapaxes(refine_corners_src, 0, 1)
          # refine_corners_src = np.hstack((refine_corners_src, np.zeros((refine_corners_src.shape[0], 1)))) # [N*4, 3]
          # refine_corners_src = np.reshape(refine_corners_src, (-1, 4, 3)) # [N, 4, 3]
          # refine_corners_src = refine_corners_src[:,:,(1,0,2)]
          # predef_arcuo_SCS = {}
          # for id, pos in zip(ids, refine_corners_src):
          #      predef_arcuo_SCS.update({str(int(id)): pos.tolist()})
          # dump_int_ndarray_as_json(predef_arcuo_SCS, os.path.join(self.directory, ARUCO_FLOOR + ".json"))

     @staticmethod
     def project(camera_intrinsics:Union[np.ndarray, CameraIntr], points_C):
          '''
          points_C: [N, 3]
          '''
          camera_intrinsics = CameraIntr(camera_intrinsics)
          points_I = camera_intrinsics * points_C
          return points_I #[N, 2]

     @staticmethod
     def restore(camera_intrinsics:Union[np.ndarray, CameraIntr], points_I):
          '''
          points_I: [N, 2]
          '''
          camera_intrinsics = CameraIntr(camera_intrinsics)
          points_I = homo_pad(points_I)
          K = camera_intrinsics.intr_M
          points_C = np.linalg.inv(K).dot(points_I.T) #[3, N]
          return points_C.T #[N, 3]          

     def collect_if_id_in_C0_aruco(self, ids:np.ndarray, *values):
          assert isinstance(ids, np.ndarray)
          assert np.issubdtype(ids.dtype, np.integer)
          assert all([len(ids) == len(v) for v in values]), "all the values must has the same size as ids"
          collectors = [[] for _ in range(len(values))]
          C0_aruco_3d = []
          for i, _id in enumerate(ids):
               if _id in self.C0_aruco_3d_dict:
                    C0_aruco_3d.append(self.C0_aruco_3d_dict[_id])
                    for c, v in zip(collectors, values):
                         c.append(v[i])
          return C0_aruco_3d, *collectors

     def get_T_3d(self, color, depth, camera_intrinsics, tol = 0.01, return_coords = True):
          scene_aruco_3d, ids, _ = ArucoDetector.detect_aruco_3d(color, depth, camera_intrinsics)
          C0_aruco_3d, common_scene_aruco_3d = self.collect_if_id_in_C0_aruco(ids, scene_aruco_3d)
          if len(common_scene_aruco_3d) == 0:
               transform = np.eye(4)
          else:
               C0_aruco_3d = np.array(C0_aruco_3d).reshape((-1, 4, 3))   # [N, 4, 3]
               scene_aruco_3d = np.array(common_scene_aruco_3d).reshape((-1, 4, 3))  # [N, 4, 3]
               ### 只选取近处点
               scene_aruco_3d_centers = np.mean(scene_aruco_3d, axis=1) #[N, 3]
               distances = np.linalg.norm(scene_aruco_3d_centers, axis = -1)
               argsort_idx = np.argsort(distances)[:3] #最多3个
               scene_aruco_3d = scene_aruco_3d[argsort_idx]
               C0_aruco_3d = C0_aruco_3d[argsort_idx]
               ### 
               scene_aruco_3d = np.reshape(scene_aruco_3d, (-1, 3))
               C0_aruco_3d = np.reshape(C0_aruco_3d, (-1, 3))
               sol = findplane_wo_outliers(scene_aruco_3d)
               if sol is not None:
                    sol = sol/np.linalg.norm(sol[:3])
                    distance = np.abs(np.sum(scene_aruco_3d * sol[:3], axis=-1) + sol[3])
                    if np.sum(distance < 0.005) < 4:
                         _scene_aruco_3d = np.reshape(scene_aruco_3d, (-1, 4, 3))
                         colors = plt.get_cmap('jet')(np.linspace(0, 1, scene_aruco_3d.shape[0]))
                         plt.figure(0)
                         ax = plt.axes(projection = '3d')
                         for C0,c  in zip(_scene_aruco_3d, colors):
                              ax.scatter(C0[:,0], C0[:,1], C0[:,2], s = 10, marker = 'o', color = c[:3])
                         plt.show()
                         sol = None
                    else:
                         # 对投影缩放，使得所有点在同一平面
                         distance = np.sum(scene_aruco_3d * sol[:3], axis=-1)
                         affine_ratio = -distance / sol[3]
                         ingroup_scene_aruco_3d = (scene_aruco_3d.T / affine_ratio).T
                         ingroup_C0_aruco_3d = C0_aruco_3d
               if sol is None:
                    ingroup_scene_aruco_3d = scene_aruco_3d
                    ingroup_C0_aruco_3d = C0_aruco_3d
               transform = match_ransac(ingroup_scene_aruco_3d, ingroup_C0_aruco_3d, tol = tol)
          if transform is None:
               transform = np.eye(4)
          else:
               transform = np.asarray(transform)
          if return_coords:
               re_trans = transform.dot(np.hstack((ingroup_scene_aruco_3d, np.ones((ingroup_scene_aruco_3d.shape[0], 1)))).T).T[:, :3]
               return transform, (ingroup_scene_aruco_3d, C0_aruco_3d, re_trans, ids)
          else:
               return transform

     def get_T_2d(self, color:np.ndarray, depth:np.ndarray, camera_intrinsics:CameraIntr, return_coords = True):
          assert self.C0_aruco_3d_dict is not None
          scene_aruco_2d, ids, _ = ArucoDetector.detect_aruco_2d(color) # [N, 4, 2], [N]
          (C0_aruco_3d, 
           common_scene_aruco_2d) = self.collect_if_id_in_C0_aruco(ids, scene_aruco_2d)
          if len(common_scene_aruco_2d) < 1:
               transform = np.eye(4)
          else:
               C0_aruco_3d = np.array(C0_aruco_3d).reshape((-1, 4, 3))   #[N, 3]
               common_scene_aruco_2d = np.array(common_scene_aruco_2d).reshape((-1, 4, 2)) #[N,2]
               ### 只选取近处点
               # cube_idx = np.where(np.array(common_scene_aruco_ids)<20)[0]
               # scene_aruco_3d_centers = np.mean(common_scene_aruco_3d, axis=1) #[N, 3]
               # distances = np.linalg.norm(scene_aruco_3d_centers, axis = -1)
               # distances[cube_idx] = 1e10
               # floor_idx = np.argsort(distances)[:3] #最多3个
               # argsort_idx = np.concatenate((cube_idx, floor_idx))
               # common_scene_aruco_2d = common_scene_aruco_2d[argsort_idx]
               # C0_aruco_3d = C0_aruco_3d[argsort_idx]
               ### 
               C0_aruco_3d = np.reshape(C0_aruco_3d, (-1, 3))
               common_scene_aruco_2d = np.reshape(common_scene_aruco_2d, (-1, 2))
               fx = camera_intrinsics.cam_fx
               fy = camera_intrinsics.cam_fy
               ppx = camera_intrinsics.cam_cx
               ppy = camera_intrinsics.cam_cy
               cameraMatrix = np.array([[fx, 0, ppx], [0, fy, ppy], [0,0,1]])
               distCoeffs = np.array([0.0,0,0,0,0])
               _, rvec, tvec = cv2.solvePnP(C0_aruco_3d, common_scene_aruco_2d, cameraMatrix, distCoeffs)
               posture = Posture(rvec = rvec, tvec = np.squeeze(tvec))
               transform = posture.inv_transmat
          if return_coords:
               re_trans = cv2.projectPoints(C0_aruco_3d, rvec, tvec, cameraMatrix, distCoeffs)[0]
               re_trans = np.squeeze(re_trans, axis=1)
               return transform, (scene_aruco_2d, C0_aruco_3d, re_trans, ids)
          else:
               return transform

     def verify_frame(self, cad, depth, camera_intrinsics, if_2d = True):
          if not if_2d:
               transform, (scene_aruco_3d, C0_aruco_3d, re_trans, ids) = self.get_T_3d(
                    cad, depth, camera_intrinsics, tol = self.verify_tol, return_coords=True)
               errors = np.linalg.norm(np.abs(re_trans - C0_aruco_3d), axis = -1)
               errors = errors[np.argsort(errors)][:int(len(errors)*0.8)]
               if len(ids) >= 8: 
                    if np.any(errors > 0.005) and np.mean(errors) > 0.003:
                         return False, None             
               else:
                    if np.any(errors > 0.004) and np.mean(errors) > 0.002:
                         return False, None  
               return True, transform
          else:
               # cannot be verified by 2d
               transform = self.get_T_2d(
                    cad, depth, camera_intrinsics, return_coords = False)
               return True, transform
          


# class CubePosture():
#      def __init__(self, directory) -> None:
#           self.directory = directory    
#           self.cube_aruco_3d_O_plane = {
#                4: np.array([ 1, 0, 0, -0.03501]),
#                3: np.array([-1, 0, 0, -0.03501]),
#                7: np.array([0,  1, 0, -0.03501]),
#                5: np.array([0, -1, 0, -0.03501]),
#                12: np.array([0, 0, 1, -0.03501])
#           } 
#           icp_dir = os.path.join(self.directory, ICP_DIR)
#           try:
#                os.mkdir(icp_dir)
#           except:
#                pass
#           ### 读取变换矩阵
#           self.trans_path = os.path.join(icp_dir, "7x7cube_transform.npy")
#           if os.path.exists(self.trans_path):
#                self.T_O2C0  = np.load(self.trans_path)
#           else:
#                self.T_O2C0  = None
#           ### 读取物体坐标系下aruco的位置
#           self.aruco_O_path = os.path.join(self.directory, "aruco_O_dict.json")
#           if os.path.exists(self.aruco_O_path):
#                with open(self.aruco_O_path, 'r') as jf:
#                     self.aruco_O_dict = dict_key_to_int(json.load(jf))
#           else:
#                self.aruco_O_dict = None

#      def load_cube_aruco_3d_C0_dict(self, cube_aruco_3d_C0_dict):
#           new_cube_aruco_3d_C0_dict = {}
#           cube_aruco_3d_C0_dict = dict_key_to_int(cube_aruco_3d_C0_dict)
#           for key, points_face in cube_aruco_3d_C0_dict.items():
#                key = int(key)
#                points_face = np.reshape(points_face, (-1, 3))
#                ### 进行过滤，有些点可能偏移很大
#                sol, out_idx = findplane_wo_outliers(points_face, 0.06, True)
#                if sol is not None:
#                     sol = sol/np.linalg.norm(sol[:3])
#                     distance = np.abs(np.sum(points_face * sol[:3], axis=-1) + sol[3])
#                     # outs = points_face[distance > 0.005]
#                     # # 对投影缩放，使得超出的点在同一平面
#                     # depth = np.sum(outs * sol[:3], axis=-1)
#                     # affine_ratio = -depth / sol[3]
#                     # outs = (outs.T / affine_ratio).T
#                     # points_face[distance > 0.005] = outs
#                     # # 缩放后可能仍有一些点离群，将整组排除
#                     # sol, out_idx = findplane_wo_outliers(points_face, 0.06, True)
#                     points_face = np.reshape(points_face, (-1, 4, 3))
#                     out_idx = np.unique((out_idx / 4).astype(np.int32))
#                     ok_group_idx = np.setdiff1d(np.array(range(points_face.shape[0])), out_idx)
#                     points_face = points_face[ok_group_idx]
#                     # pp = np.reshape(points_face, (-1, 3))
#                     # ax = plt.axes(projection='3d')  # 设置三维轴
#                     # ax.scatter(pp[:, 0], pp[:, 1], pp[:, 2], c = 'r')
#                     # plt.show()

#                     new_cube_aruco_3d_C0_dict.update({key: points_face})
#                if sol is None:
#                     print("!")
#           self.cube_aruco_3d_C0_dict = new_cube_aruco_3d_C0_dict

#      def solve_cube_aruco_pos_by_images(self):
#           '''
#           通过图片确定在cube上的aruco点的坐标
#           图片分为两组：第一组5张，坐标图，是5个面的的平视
#           第二组：方位图，从4个角度斜向观看
#           '''
#           image_dir = os.path.join("LINEMOD", "cube")
#           image_list = []
#           aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
#           parameters = aruco.DetectorParameters_create()
#           def onclick(event):
#                """
#                鼠标点击事件处理函数，获取鼠标所在位置的像素坐标并将其保存到列表中
#                """
#                if event.button == 1:  # 鼠标左键点击
#                     ix, iy = event.xdata, event.ydata
#                     print(f'x = {ix:.2f}, y = {iy:.2f}')
#                     points.append((ix, iy))  # 将坐标保存到列表中
#                     plt.plot(ix, iy, 'ro')  # 在图像上绘制一个红点
#                elif event.button == 3:  # 鼠标右键点击
#                     points.clear()  # 清空点坐标列表
#                     ax.clear()  # 清除图像上的所有点
#                     ax.imshow(image)  # 重新显示原始图像
#           scale_rate = 5
#           outer_size = 70
#           aruco_size = 60
#           image_size = outer_size * scale_rate
#           std_points = np.array([(0, 0),
#                                  (0, image_size), 
#                                  (image_size, image_size), 
#                                  (image_size, 0)], np.float32)
#           coord_images:list[np.ndarray] = []
#           orientation_images:list[np.ndarray] = []
#           ### 读入图像并分类：坐标图/方位图
#           for name in os.listdir(image_dir):
#                gray_src = plt.imread(os.path.join(image_dir, name), cv2.IMREAD_GRAYSCALE)
#                gray_src = np.clip((gray_src*255), 0, 255).astype(np.uint8)
#                corners_src, _ids_src, rejectedImgPoints = aruco.detectMarkers(gray_src, aruco_dict, parameters=parameters)
#                if len(corners_src) == 1:
#                     coord_images.append(gray_src)
#                elif len(corners_src) == 3:
#                     orientation_images.append(gray_src)
          
#           # 方位图
#           all_neighbors:dict[int, dict[tuple, int]] = {}
#           for image in orientation_images:
#                corners_src, _ids_src, rejectedImgPoints = aruco.detectMarkers(image, aruco_dict, parameters=parameters)
#                _ids_src = np.squeeze(_ids_src, axis=1).astype(np.int32)
#                corners_src = np.squeeze(np.array(corners_src), axis=1)
#                centers = np.mean(corners_src, axis=1)
#                for center_i in range(3):
#                     center_id = _ids_src[center_i]
#                     if center_id not in all_neighbors:
#                          all_neighbors.update({center_id: {}})
#                     corners = corners_src[center_i]
#                     other_idx = np.setdiff1d(np.array([0,1,2]), center_i)
#                     for idx in other_idx:
#                          other_id = _ids_src[idx]
#                          other_corners = corners_src[idx] #[4, 2]
#                          other_center = np.mean(other_corners, axis = 0)
#                          distances = np.linalg.norm(corners - other_center, axis=-1)
#                          min_2 = np.argsort(distances)[:2] # 最近的2个
#                          min_2 = min_2[np.argsort(min_2)]
#                          all_neighbors[center_id].update({(min_2[0], min_2[1]): other_id})
#           for n in all_neighbors.values(): #靠近底面的aruco只有3个邻，补充为-1
#                for pair in [(0,1), (1,2), (2,3), (0,3)]:
#                     if pair not in n:
#                          n.update({pair: -1})

#           aruco_O_dict = {}
#           # 坐标图
#           for image in coord_images:
#                fig, ax = plt.subplots()
#                ax.imshow(image)

#                # 初始化点坐标列表
#                points = [] #边界点

#                # 注册鼠标点击事件处理函数
#                cid = fig.canvas.mpl_connect('button_press_event', onclick)

#                # 显示图像
#                plt.ion()
#                plt.show()
#                while True:
#                     if plt.waitforbuttonpress():
#                          if plt.get_current_fig_manager().toolbar.mode == '':
#                               break
#                plt.close()
#                print(points)
#                points = np.array(points, np.float32)
#                # 计算单应矩阵
#                H, _ = cv2.findHomography(points, std_points)
#                # 对图像进行对齐变换
#                aligned_img = cv2.warpPerspective(image, H, (image_size, image_size))
#                gray_src = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)
#                ### 检测aruco角点
#                corners_src, _ids_src, rejectedImgPoints = aruco.detectMarkers(gray_src, aruco_dict, parameters=parameters)
#                corners_src = np.squeeze(corners_src)
#                aruco_A = np.hstack((corners_src[:, ::-1], np.ones((4,1))))
#                aruco_A[:, 2] = 0.03501
#                aruco_A[:, :2] -= np.array((image_size/2, image_size/2))
#                aruco_A[:, :2] = aruco_A[:, :2]/scale_rate/1000
#                # 根据方位图选择正确的方位
#                this_id = int(np.squeeze(_ids_src))
#                neighbors = all_neighbors[this_id]
#                plane = self.cube_aruco_3d_O_plane[this_id]
#                rot = np.cross([0,0,1], plane[:3]) * np.pi/2
#                base_rmat = Posture(rvec=rot).trans_mat
#                ok_pos = None
#                min_dist = 1e10
#                for r_z in np.linspace(0, 2*np.pi, 4, False):
#                     R = Posture(rvec=np.array([0,0,r_z])).trans_mat
#                     temp_pos = np.linalg.multi_dot((base_rmat, R, homo_pad(aruco_A).T)).T[:, :3]
#                     # 计算到邻面的距离
#                     dist = 0

#                     for ci, pi in neighbors.items():
#                          if pi == -1:
#                               pe = np.array([0,0,1,0.03501])
#                          else:
#                               pe = self.cube_aruco_3d_O_plane[pi]
#                          mid = np.mean(temp_pos[ci, :], axis=0)
#                          d = np.abs(np.sum(mid * pe[:3]) + pe[3])
#                          # print(d)
#                          dist += d
#                     if dist < min_dist:
#                          min_dist = dist
#                          ok_pos = temp_pos
#                aruco_O_dict.update({this_id: ok_pos})
#           self.aruco_O_dict = aruco_O_dict
#           dump_int_ndarray_as_json(aruco_O_dict, self.aruco_O_path)
#           return aruco_O_dict
          


#      def solve_cube_aruco_pos(self, cube_transform, aruco_size = 0.060):
#           cube_aruco_3d_O_plane = self.cube_aruco_3d_O_plane
#           cube_aruco_3d_C0_dict = self.cube_aruco_3d_C0_dict
#           ahs = aruco_size / 2 #aruco half size
#           aruco_A = np.array([[-ahs, -ahs, 0.03501], [-ahs, ahs, 0.03501], [ahs, ahs, 0.03501], [ahs, -ahs, 0.03501]])
#           aruco_A = homo_pad(aruco_A)
#           aruco_O_dict = {}
#           for id, points_C0 in cube_aruco_3d_C0_dict.items():
#                print(id)
#                points_C0 = np.reshape(points_C0, (-1, 3))
#                points_C0 = homo_pad(points_C0)
#                points_O = np.linalg.inv(cube_transform).dot(points_C0.T).T[:, :3] # 物体坐标系下的坐标
#                points_O = np.reshape(points_O, (-1, 4, 3))
#                plane = cube_aruco_3d_O_plane[int(id)] # 平面方程
#                rot = np.cross([0,0,1], plane[:3]) * np.pi/2
#                base_rmat = Posture(rvec=rot).trans_mat
#                def trans_mat(parameter):
#                     t = Posture(tvec=np.array([parameter[0], parameter[1], 0])).trans_mat
#                     R = Posture(rvec=np.array([0, 0, parameter[2]])).trans_mat
#                     T = np.linalg.multi_dot((base_rmat, t, R))
#                     return T
#                def func(parameter):
#                     # parameter[:2] # 沿着xy的平移
#                     # parameter[2] #在Aruco系下绕Z轴旋转的角度
#                     # t = Posture(tvec=np.array([parameter[0], parameter[1], 0]))
#                     # R = Posture(rvec=np.array([0, 0, parameter[2]]))
#                     T = trans_mat(parameter)
#                     aruco_O = np.linalg.multi_dot((T, aruco_A.T)).T[:, :3] # [4, 3]
#                     delta = np.linalg.norm(points_O - aruco_O, axis=-1)
#                     delta = np.mean(delta)
#                     return delta
#                ga = GA(func=func, n_dim=3, size_pop=1000, max_iter=100,    lb=[-0.005, -0.005, -np.pi], 
#                                                                            ub=[ 0.005,  0.005,  np.pi], 
#                                                                            precision=[ 0.0001, 0.0001, np.pi/360])
#                best_x, best_y = ga.run()   
#                print(best_y)
#                T = trans_mat(best_x)
#                aruco_O = np.linalg.multi_dot((T, aruco_A.T)).T[:, :3] # [4, 3]
#                ax = plt.axes(projection='3d')  # 设置三维轴
#                ax.scatter(aruco_O[:, 0], aruco_O[:, 1], aruco_O[:, 2], c = 'r')
#                ppo = np.reshape(points_O, (-1, 3))
#                ax.scatter(ppo[:, 0], ppo[:, 1], ppo[:, 2], c = 'b')
#                plt.show()
#                aruco_O_dict.update({id: aruco_O.tolist()})
#           self.aruco_O_dict = aruco_O_dict
#           with open(self.aruco_O_path, 'w') as f:
#                json.dump(aruco_O_dict, f)
#           return aruco_O_dict

#      def solve_cube_pos(self):
#           cube_aruco_3d_O_plane = self.cube_aruco_3d_O_plane
#           cube_aruco_3d_C0_dict = self.cube_aruco_3d_C0_dict
#           points = []
#           laws = []
#           ds = []
#           for key, points_face in cube_aruco_3d_C0_dict.items():
#                key = int(key)
#                points_face = np.reshape(points_face, (-1, 3))
#                if key == 12:
#                     base_center = np.mean(points_face, axis=0)[0:3]
#                     base_center[2] = 0.03501
#                else:
#                     PN = points_face.shape[0]
#                     points.append(points_face)
#                     law = cube_aruco_3d_O_plane[key][:3]
#                     laws.append(np.tile(np.expand_dims(np.array(law), axis=0), (PN, 1)))
#                     d = cube_aruco_3d_O_plane[key][3]
#                     ds.append(np.tile(np.array([d]), (PN)))
#           # 使点数量均衡
#           min_num = min([len(x) for x in ds])
#           for i in range(len(points)):
#                points[i] = points[i][:min_num]
#                laws[i] = laws[i][:min_num]
#                ds[i] = ds[i][:min_num]
#           points = np.vstack(points)
#           ax = plt.axes(projection='3d')  # 设置三维轴
#           ax.scatter(points[:, 0], points[:, 1], points[:, 2], c = 'r')
#           plt.show()
#           # points = homo_pad(points) #[N, 4]
#           laws = np.vstack(laws)
#           ds = np.hstack(ds)

#           # 求解
#           def trans(posture):
#                z_angle = posture[2]
#                rot_mat = np.array([   [np.cos(z_angle),     -np.sin(z_angle),   0],
#                                         [np.sin(z_angle),     np.cos(z_angle),    0],
#                                         [0,                   0,                  1]])               
#                return rot_mat.dot((points - base_center).T).T + np.array([posture[0], posture[1], 0])
                                                                  
#           def func(posture):
#                res = (trans(posture)
#                        * laws).sum(axis=-1) + ds
#                res = np.mean(np.abs(res))
#                return res * 1000
#           # set constraint on T to be full rank
#           ga = GA(func=func, n_dim=3, size_pop=1000, max_iter=100,    lb=[-0.01, -0.01, -np.pi], 
#                                                                       ub=[ 0.01,  0.01,  np.pi], 
#                                                                       precision=[ 0.0001, 0.0001, np.pi/360])
#           best_x, best_y = ga.run()
#           print(best_y)
#           t1 = Posture(tvec= - base_center)
#           r = Posture(rvec=np.array([0,0,best_x[2]]))
#           t2 = Posture(tvec= np.array([best_x[0], best_x[1], 0]))
#           posture = np.linalg.multi_dot((t2.trans_mat, r.trans_mat, t1.trans_mat))
#           transform = np.linalg.inv(posture)
#           self.T_O2C0  = transform
#           np.save(self.trans_path, transform)

#      def get_cube_aruco_C0_dict(self):
#           cube_aruco_C0_dict = {}
#           for id, points in self.aruco_O_dict.items():
#                # if id == 12:
#                #      continue
#                pC0 = homo_pad(np.array(points))
#                pO = self.T_O2C0.dot(pC0.T).T[:, :3]
#                cube_aruco_C0_dict.update({id: pO})
#           return cube_aruco_C0_dict

# class GtPostureComputer():
#      def __init__(self, directory) -> None:
#           self.directory = directory
#           self.predef_arcuo_SCS = self.get_predef_arcuo_SCS(0.694733)
#           try:
#                self.cali_camera_intr = CameraIntr(os.path.join(directory, CALI_INTR_FILE))
#           except FileNotFoundError:
#                self.cali_camera_intr = None
#           try:
#                self.data_camera_intr = CameraIntr(os.path.join(directory, DATA_INTR_FILE))
#           except FileNotFoundError:
#                self.data_camera_intr = None
#           self.verify_tol = 0.01

#      @staticmethod
#      def __cross_point(line1, line2):  # 计算交点函数
#           x1 = line1[0]  # 取直线1的第一个点坐标
#           y1 = line1[1]
#           x2 = line1[2]  # 取直线1的第二个点坐标
#           y2 = line1[3]
          
#           x3 = line2[0]  # 取直线2的第一个点坐标
#           y3 = line2[1]
#           x4 = line2[2]  # 取直线2的第二个点坐标
#           y4 = line2[3]
          
#           if x2 - x1 == 0:  # L1 直线斜率不存在
#                k1 = None
#                b1 = 0
#           else:
#                k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
#                b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键
          
#           if (x4 - x3) == 0:  # L2直线斜率不存在操作
#                k2 = None
#                b2 = 0
#           else:
#                k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在操作
#                b2 = y3 * 1.0 - x3 * k2 * 1.0
          
#           if k1 is None and k2 is None:  # L1与L2直线斜率都不存在，两条直线均与y轴平行
#                if x1 == x3:  # 两条直线实际为同一直线
#                     return [x1, y1]  # 均为交点，返回任意一个点
#                else:
#                     return None  # 平行线无交点
#           elif k1 is not None and k2 is None:  # 若L2与y轴平行，L1为一般直线，交点横坐标为L2的x坐标
#                x = x3
#                y = k1 * x * 1.0 + b1 * 1.0
#           elif k1 is None and k2 is not None:  # 若L1与y轴平行，L2为一般直线，交点横坐标为L1的x坐标
#                x = x1
#                y = k2 * x * 1.0 + b2 * 1.0
#           else:  # 两条一般直线
#                if k1 == k2:  # 两直线斜率相同
#                     if b1 == b2:  # 截距相同，说明两直线为同一直线，返回任一点
#                          return [x1, y1]
#                     else:  # 截距不同，两直线平行，无交点
#                          return None
#                else:  # 两直线不平行，必然存在交点
#                     x = (b2 - b1) * 1.0 / (k1 - k2)
#                     y = k1 * x * 1.0 + b1 * 1.0
#           return [x, y]

#      def __refine_aruco(self, corners_src, gray):
#           refined_corners_src = []
#           for corners in corners_src:
#                corners = corners[0].astype(np.int32)
#                #[4,2]
#                crop_min = np.min(corners, axis=0) - 4
#                crop_min[0] = max(crop_min[0], 0)
#                crop_min[1] = max(crop_min[1], 0)
#                crop_max = np.max(corners, axis=0) + 4
#                crop_max[0] = min(crop_max[0], gray.shape[1])
#                crop_max[1] = min(crop_max[1], gray.shape[0])

#                pad_mask = np.zeros(gray.shape, np.uint8)
#                pad_mask = cv2.fillPoly(pad_mask, [corners], 1)
#                pad_mask = cv2.morphologyEx(pad_mask, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations=2)
#                gray_copy = gray.copy()
#                gray_copy[pad_mask.astype(np.bool_)] = 30
#                # gray_copy = cv2.threshold(gray_copy,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]
#                # gray_copy = 255 - gray_copy
#                croped = gray_copy[crop_min[1]:crop_max[1], crop_min[0]:crop_max[0]]

#                ### 用两种方法检测直线，求直线间的交点
#                # 1
#                line_groups = []
#                cross_point_groups = []
#                fld = cv2.ximgproc.createFastLineDetector()
#                detect = fld.detect(croped)          
#                if detect is not None:
#                     line_groups.append(np.squeeze(detect, axis=1))
#                # 2
#                lsd = cv2.createLineSegmentDetector(0)
#                line_groups.append(np.squeeze(lsd.detect(croped)[0], axis=1))
#                for dlines in line_groups:
#                     if len(dlines) != 4:
#                          continue
#                     # 直线排序
#                     dlines_ = np.reshape(dlines, (-1, 2, 2))
#                     centers = np.mean(dlines_, axis=1)
#                     angles = np.arctan2(centers[:, 1], centers[:, 0])
#                     angles[angles<0] += 2*np.pi
#                     dlines = dlines[np.argsort(angles)]
#                     dlines = np.vstack((dlines, dlines[0]))
#                     cross_points = []
#                     for i in range(4):
#                          cross_points.append(self.__cross_point(dlines[i], dlines[i + 1]))
#                     cross_points = np.array(cross_points)
#                     cross_point_groups.append(cross_points)
#                if len(cross_point_groups) == 0:
#                     refined_corners = corners
#                else:               
#                     refined_corners = np.array(cross_point_groups) + crop_min #[2, 4, 2]
#                     refined_corners = np.mean(refined_corners, 0)
#                     angles = np.arctan2(refined_corners[:, 1], refined_corners[:, 0])
#                     angles[angles<0] += 2*np.pi
#                     sort_new = np.argsort(angles)
#                     angles = np.arctan2(corners[:, 1], corners[:, 0])
#                     angles[angles<0] += 2*np.pi
#                     sort_org = np.argsort(angles)
#                     inv_sort_org = np.array([int(np.where(sort_org == 0)[0]), int(np.where(sort_org == 1)[0]), 
#                                              int(np.where(sort_org == 2)[0]), int(np.where(sort_org == 3)[0])])
#                     refined_corners = refined_corners[sort_new][inv_sort_org]
                    
#                     delta = np.abs(refined_corners - corners)
#                     if np.any(delta > 2):
#                          refined_corners = corners
#                # 按原有顺序排序：

#                refined_corners_src.append(refined_corners)

#                # for dline in dlines[0]:
#                #      x0 = int(round(dline[0][0]))
#                #      y0 = int(round(dline[0][1]))
#                #      x1 = int(round(dline[0][2]))
#                #      y1 = int(round(dline[0][3]))
#                #      plt.plot((x0,x1), (y0, y1))
#                # lsd = cv2.createLineSegmentDetector(0)
#                # # 执行检测结果
#                # dlines = lsd.detect(croped)
#                # for dline in dlines[0]:
#                #      x0 = int(round(dline[0][0]))
#                #      y0 = int(round(dline[0][1]))
#                #      x1 = int(round(dline[0][2]))
#                #      y1 = int(round(dline[0][3]))
#                #      plt.plot((x0,x1), (y0, y1))
#                # plt.imshow(gray[crop_min[1]:crop_max[1], crop_min[0]:crop_max[0]])
#                # plt.scatter(refined_corners[:,0]- crop_min[0], refined_corners[:,1] - crop_min[1], c = 'r')
#                # plt.scatter(corners[:,0] - crop_min[0], corners[:,1] - crop_min[1], c = 'b')
#                # plt.show()
#                # plt.clf()
#           return np.array(refined_corners_src)

#      def get_images(self, Filenum, directory = None):
#           if self is not None:
#                directory = self.directory
#           Filename = str(Filenum).rjust(6, "0")
#           img_file = os.path.join(directory, RGB_DIR, '{}.jpg'.format(Filename))
#           # mask = cv2.imread(img_file, 0)   
#           cad = cv2.imread(img_file)
#           cad = cv2.cvtColor(cad, cv2.COLOR_BGR2RGB)

#           depth_file = os.path.join(directory, DEPTH_DIR, '{}.png'.format(Filename))
#           try:
#                reader = png.Reader(depth_file)
#                pngdata = reader.read()
#                depth:np.ndarray = np.array(tuple(map(np.uint16, pngdata[2])))
#           except FileNotFoundError:
#                depth = None
#           return cad, depth

#      @staticmethod
#      def get_aruco_coord(cad, depth, camera_intrinsics, get3d = True):
#           aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
#           parameters = aruco.DetectorParameters_create()
#           gray = cv2.cvtColor(cad, cv2.COLOR_RGB2GRAY)
#           corners_src, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
#           if len(corners_src) > 0:
#                corners_src = np.squeeze(np.array(corners_src), axis=1)     
#                ids = np.array(ids).squeeze(axis=-1)
#           else:
#                corners_src = np.zeros((0,4,2))
#                ids = np.array([])
#           if get3d:
#                depth = convert_depth_frame_to_pointcloud(depth, camera_intrinsics)
#                scene_aruco = np.round(corners_src).astype(np.int32)
#                scene_aruco_3d = depth[scene_aruco[:, :, 1], scene_aruco[:, :, 0]]   #[N,4,3]
#                return scene_aruco_3d, ids
#           else:
#                return corners_src, ids
#           np.random.random()

#      @staticmethod
#      def project_from_C0(trans_mat_Cn2C0, camera_intrinsics, C0_corners):
#           C0_corners = homo_pad(C0_corners)
#           Cn_corners = np.linalg.inv(trans_mat_Cn2C0).dot(C0_corners.T).T[:, :3]
#           fx = camera_intrinsics["fx"]
#           fy = camera_intrinsics["fy"]
#           ppx = camera_intrinsics["ppx"]
#           ppy = camera_intrinsics["ppy"]
#           K = np.array([[fx, 0, ppx],[0, fy, ppy],[0, 0, 1]])
#           I_corners = K.dot(Cn_corners.T) #[3, N]
#           I_corners = I_corners / I_corners[2]
#           return I_corners[:2].T #[N, 2]
     
#      @staticmethod
#      def project(camera_intrinsics, points_C):
#           '''
#           points_C: [N, 3]
#           '''
#           fx = camera_intrinsics["fx"]
#           fy = camera_intrinsics["fy"]
#           ppx = camera_intrinsics["ppx"]
#           ppy = camera_intrinsics["ppy"]
#           K = np.array([[fx, 0, ppx],[0, fy, ppy],[0, 0, 1]])
#           points_I = K.dot(points_C.T) #[3, N]
#           points_I = points_I / points_I[2]
#           return points_I[:2].T #[N, 2]

#      @staticmethod
#      def restore(camera_intrinsics, points_I):
#           '''
#           points_I: [N, 2]
#           '''
#           fx = camera_intrinsics["fx"]
#           fy = camera_intrinsics["fy"]
#           ppx = camera_intrinsics["ppx"]
#           ppy = camera_intrinsics["ppy"]
#           points_I = homo_pad(points_I)
#           K = np.array([[fx, 0, ppx],[0, fy, ppy],[0, 0, 1]])
#           points_C = np.linalg.inv(K).dot(points_I.T) #[3, N]
#           return points_C.T #[N, 3]

#      @staticmethod
#      def get_T_3d(cad, depth, camera_intrinsics, C0_aruco_3d_dict:dict, tol = 0.01, return_coords = True):
#           scene_aruco_3d, ids = GtPostureComputer.get_aruco_coord(cad, depth, camera_intrinsics)
#           C0_aruco_3d = []
#           common_scene_aruco_3d = []
#           for id, sa in zip(ids, scene_aruco_3d):
#                try:
#                     C0_aruco_3d.append(C0_aruco_3d_dict[int(id)])
#                     common_scene_aruco_3d.append(sa)
#                except:
#                     pass
#           if len(common_scene_aruco_3d) == 0:
#                transform = np.eye(4)
#           else:
#                C0_aruco_3d = np.array(C0_aruco_3d)   #[N, 3]
#                scene_aruco_3d = np.array(common_scene_aruco_3d)   #[N, 3]
#                scene_aruco_3d = np.reshape(scene_aruco_3d, (-1, 4, 3))
#                C0_aruco_3d = np.reshape(C0_aruco_3d, (-1, 4, 3))
#                ### 只选取近处点
#                scene_aruco_3d_centers = np.mean(scene_aruco_3d, axis=1) #[N, 3]
#                distances = np.linalg.norm(scene_aruco_3d_centers, axis = -1)
#                argsort_idx = np.argsort(distances)[:3] #最多3个
#                scene_aruco_3d = scene_aruco_3d[argsort_idx]
#                C0_aruco_3d = C0_aruco_3d[argsort_idx]
#                ### 
#                scene_aruco_3d = np.reshape(scene_aruco_3d, (-1, 3))
#                C0_aruco_3d = np.reshape(C0_aruco_3d, (-1, 3))
#                sol = findplane_wo_outliers(scene_aruco_3d)
#                if sol is not None:
#                     sol = sol/np.linalg.norm(sol[:3])
#                     distance = np.abs(np.sum(scene_aruco_3d * sol[:3], axis=-1) + sol[3])
#                     if np.sum(distance < 0.005) < 4:
#                          _scene_aruco_3d = np.reshape(scene_aruco_3d, (-1, 4, 3))
#                          colors = plt.get_cmap('jet')(np.linspace(0, 1, scene_aruco_3d.shape[0]))
#                          plt.figure(0)
#                          ax = plt.axes(projection = '3d')
#                          for C0,c  in zip(_scene_aruco_3d, colors):
#                               ax.scatter(C0[:,0], C0[:,1], C0[:,2], s = 10, marker = 'o', color = c[:3])
#                          plt.show()
#                          sol = None
#                     else:
#                          # 对投影缩放，使得所有点在同一平面
#                          distance = np.sum(scene_aruco_3d * sol[:3], axis=-1)
#                          affine_ratio = -distance / sol[3]
#                          ingroup_scene_aruco_3d = (scene_aruco_3d.T / affine_ratio).T
#                          ingroup_C0_aruco_3d = C0_aruco_3d
#                if sol is None:
#                     ingroup_scene_aruco_3d = scene_aruco_3d
#                     ingroup_C0_aruco_3d = C0_aruco_3d
#                transform = match_ransac(ingroup_scene_aruco_3d, ingroup_C0_aruco_3d, tol = tol)
#           if transform is None:
#                transform = np.eye(4)
#           else:
#                transform = np.asarray(transform)
#           if return_coords:
#                re_trans = transform.dot(np.hstack((ingroup_scene_aruco_3d, np.ones((ingroup_scene_aruco_3d.shape[0], 1)))).T).T[:, :3]
#                return transform, (ingroup_scene_aruco_3d, C0_aruco_3d, re_trans, ids)
#           else:
#                return transform

#      @staticmethod
#      def get_T_2d(cad, depth, camera_intrinsics, C0_aruco_3d_dict:dict, tol = 0.01, return_coords = True):
#           scene_aruco_2d, ids = GtPostureComputer.get_aruco_coord(cad, depth, camera_intrinsics, False) #[N,4,2], [N]
#           scene_aruco_3d, ids = GtPostureComputer.get_aruco_coord(cad, depth, camera_intrinsics)
#           C0_aruco_3d = []
#           common_scene_aruco_ids = []
#           common_scene_aruco_2d = []
#           common_scene_aruco_3d = []
#           for id, sa_2d, sa_3d in zip(ids, scene_aruco_2d, scene_aruco_3d):
#                try:
#                     C0_aruco_3d.append(C0_aruco_3d_dict[int(id)])
#                     common_scene_aruco_ids.append(id)
#                     common_scene_aruco_2d.append(sa_2d)
#                     common_scene_aruco_3d.append(sa_3d)
#                except:
#                     pass
#           if len(common_scene_aruco_3d) < 1:
#                transform = np.eye(4)
#           else:
#                C0_aruco_3d = np.array(C0_aruco_3d).reshape((-1, 4, 3))   #[N, 3]
#                common_scene_aruco_2d = np.array(common_scene_aruco_2d).reshape((-1, 4, 2)) #[N,2]
#                ### 只选取近处点
#                # cube_idx = np.where(np.array(common_scene_aruco_ids)<20)[0]
#                # scene_aruco_3d_centers = np.mean(common_scene_aruco_3d, axis=1) #[N, 3]
#                # distances = np.linalg.norm(scene_aruco_3d_centers, axis = -1)
#                # distances[cube_idx] = 1e10
#                # floor_idx = np.argsort(distances)[:3] #最多3个
#                # argsort_idx = np.concatenate((cube_idx, floor_idx))
#                # common_scene_aruco_2d = common_scene_aruco_2d[argsort_idx]
#                # C0_aruco_3d = C0_aruco_3d[argsort_idx]
#                ### 
#                C0_aruco_3d = np.reshape(C0_aruco_3d, (-1, 3))
#                common_scene_aruco_2d = np.reshape(common_scene_aruco_2d, (-1, 2))
#                fx = camera_intrinsics["fx"]
#                fy = camera_intrinsics["fy"]
#                ppx = camera_intrinsics["ppx"]
#                ppy = camera_intrinsics["ppy"]
#                cameraMatrix = np.array([[fx, 0, ppx], [0, fy, ppy], [0,0,1]])
#                distCoeffs = np.array([0.0,0,0,0,0])
#                _, rvec, tvec = cv2.solvePnP(C0_aruco_3d, common_scene_aruco_2d, cameraMatrix, distCoeffs)
#                posture = Posture(rvec = rvec, tvec = np.squeeze(tvec))
#                transform = posture.inv_transmat
#           if return_coords:
#                re_trans =cv2.projectPoints(C0_aruco_3d, rvec, tvec, cameraMatrix, distCoeffs)[0]
#                re_trans = np.squeeze(re_trans, axis=1)
#                return transform, (scene_aruco_2d, C0_aruco_3d, re_trans, ids)
#           else:
#                return transform

#      @staticmethod
#      def get_arucos_from_dict(corner_dict: dict[int, np.ndarray], ids: np.ndarray = None) -> list[np.ndarray]:
#           corners = []
#           if ids is None:
#                ids = list(corner_dict.keys())
#           for id in ids:
#                try:
#                     corners.append(corner_dict[id])
#                except:
#                     pass
#           return corners

#      def verify_frame(self, cad, depth, camera_intrinsics, if_2d = True):
#           C0_aruco_3d_dict = self.C0_aruco_3d_dict
#           if not if_2d:
#                transform, (scene_aruco_3d, C0_aruco_3d, re_trans, ids) = self.get_T_3d(
#                     cad, depth, camera_intrinsics, C0_aruco_3d_dict, tol = self.verify_tol, return_coords=True)
#                errors = np.linalg.norm(np.abs(re_trans - C0_aruco_3d), axis = -1)
#                errors = errors[np.argsort(errors)][:int(len(errors)*0.8)]
#                if len(ids) >= 8: 
#                     if np.any(errors > 0.005) and np.mean(errors) > 0.003:
#                          return False, None             
#                else:
#                     if np.any(errors > 0.004) and np.mean(errors) > 0.002:
#                          return False, None  
#                return True, transform
#           else:
#                transform = self.get_T_2d(
#                     cad, depth, camera_intrinsics, C0_aruco_3d_dict, return_coords = False)
#                return True, transform
             
#      def get_predef_arcuo_SCS(self, long_side_real_size):
#           try:
#                with open(os.path.join(self.directory, ARUCO_FLOOR + ".json"), 'r') as f:
#                     predef_arcuo_SCS = json.load(f)
#           except FileNotFoundError:
#                pass
#                arcuo_floor = cv2.imread(os.path.join(self.directory, ARUCO_FLOOR + ".png"), cv2.IMREAD_GRAYSCALE)
#                arcuo_floor = np.pad(arcuo_floor,[(100,100),(100,100)],'constant',constant_values=255)
#                # zoom = 4000 / np.max(arcuo_floor.shape)
#                # arcuo_floor = cv2.resize(arcuo_floor, (-1,-1), fx=zoom, fy=zoom)
#                # 粗测
#                zoom_1 = 600 / np.max(arcuo_floor.shape)
#                coarse_arcuo_floor = cv2.resize(arcuo_floor, (-1,-1), fx=zoom_1, fy=zoom_1)
#                aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
#                parameters = aruco.DetectorParameters_create()
#                corners_src, ids, rejectedImgPoints = aruco.detectMarkers(coarse_arcuo_floor, aruco_dict, parameters=parameters)
#                corners_src = np.squeeze(np.array(corners_src), axis=1)    # [N, 4, 2]
#                refine_corners_src_list = []
#                # 精测
#                for cs in corners_src:
#                     refine_corner_list = []
#                     for corner in cs: #[2]
#                          crop_rect_min = np.clip(corner - 5, [0,0], coarse_arcuo_floor.shape[::-1]) 
#                          crop_rect_max = np.clip(corner + 5, [0,0], coarse_arcuo_floor.shape[::-1]) 
#                          crop_rect_min = np.round(crop_rect_min / zoom_1).astype(np.int32)
#                          crop_rect_max = np.round(crop_rect_max / zoom_1).astype(np.int32)
#                          local = arcuo_floor[crop_rect_min[1]: crop_rect_max[1], crop_rect_min[0]: crop_rect_max[0]]
#                          refine_corner = np.squeeze(cv2.goodFeaturesToTrack(local, 1, 0.05, 0)) + crop_rect_min
#                          refine_corner_list.append(refine_corner)
#                     refine_corners_src_list.append(refine_corner_list)
#                     # crop_rect_min = np.clip(np.min(cs, axis=0) - 5, [0,0], coarse_arcuo_floor.shape[::-1]) 
#                     # crop_rect_max = np.clip(np.max(cs, axis=0) + 5, [0,0], coarse_arcuo_floor.shape[::-1]) 
#                     # crop_rect_min = np.round(crop_rect_min / zoom_1).astype(np.int32)
#                     # crop_rect_max = np.round(crop_rect_max / zoom_1).astype(np.int32)
#                     # local = arcuo_floor[crop_rect_min[1]: crop_rect_max[1], crop_rect_min[0]: crop_rect_max[0]]
#                     # zoom_2 = 600 / np.max(local.shape) if np.max(local.shape) > 600 else 1.0
#                     # local = cv2.resize(local, (-1,-1), fx=zoom_2, fy=zoom_2)
#                     # refine_corners_src, _, rejectedImgPoints = aruco.detectMarkers(local, aruco_dict, parameters=parameters)
#                     # refine_corners_src = np.squeeze(np.array(refine_corners_src))    # [4, 2]
#                     # refine_corners_src = refine_corners_src / zoom_2 + crop_rect_min
#                     # refine_corners_src_list.append(refine_corners_src)
#                refine_corners_src = np.array(refine_corners_src_list) # [N, 4, 2]

#                refine_corners_src = np.reshape(refine_corners_src, (-1, 2))
#                min_pos = np.min(refine_corners_src, axis=0)
#                refine_corners_src = refine_corners_src - min_pos
#                max_pos = np.max(refine_corners_src, axis=0) #最大范围
#                refine_corners_src = refine_corners_src * long_side_real_size / np.max(max_pos)
#                # refine_corners_src = np.swapaxes(refine_corners_src, 0, 1)
#                refine_corners_src = np.hstack((refine_corners_src, np.zeros((refine_corners_src.shape[0], 1)))) # [N*4, 3]
#                refine_corners_src = np.reshape(refine_corners_src, (-1, 4, 3)) # [N, 4, 3]
#                refine_corners_src = refine_corners_src[:,:,(1,0,2)]
#                predef_arcuo_SCS = {}
#                for id, pos in zip(ids, refine_corners_src):
#                     predef_arcuo_SCS.update({str(int(id)): pos.tolist()})
#                dump_int_ndarray_as_json(predef_arcuo_SCS, os.path.join(self.directory, ARUCO_FLOOR + ".json"))

#           self.predef_arcuo_SCS = predef_arcuo_SCS
#           self.C0_aruco_3d_dict, ids  = self.calc_C0_aruco_3d_dict(None)
#           return predef_arcuo_SCS

#      def calc_C0_aruco_3d_dict(self, model_index_dict):
#           if self.predef_arcuo_SCS is None:
#                C0_aruco_3d_dict = {}
#                C0_aruco_3d_dict_temp = {}
#                # 第一帧
#                cad, depth = self.get_images(0)
#                scene_aruco_3d, ids = self.get_aruco_coord(cad, depth, 0, self.cali_camera_intr.dict)
#                for id, a3d in zip(ids, scene_aruco_3d):
#                     C0_aruco_3d_dict.update({int(id): a3d})
#                     C0_aruco_3d_dict_temp.update({int(id): [a3d]})     
#                # 其他基准帧
#                for Filenum in range(1, model_index_dict["local_base_frames"][-1]):
#                     cad, depth = self.get_images(Filenum)
#                     transform, (scene_aruco_3d, C0_aruco_3d, re_trans, ids) = self.get_T_3d(cad, depth, self.cali_camera_intr.dict, C0_aruco_3d_dict, tol = 0.02, return_coords=True)
#                     C0_aruco_3d = np.reshape(C0_aruco_3d, (-1, 4, 3))
#                     re_trans = np.reshape(re_trans, (-1, 4, 3))
                         
#                     errors = np.linalg.norm(re_trans - C0_aruco_3d, axis = -1)
#                     # errors = errors[np.argsort(errors)][:int(len(errors)*0.8)]
#                     print(np.mean(errors))
#                     # re_trans = np.reshape(re_trans, (-1, 4, 3))
#                     for id, a3d in zip(ids, re_trans):
#                          try:
#                               C0_aruco_3d_dict_temp[int(id)].append(a3d)
#                          except KeyError:
#                               C0_aruco_3d_dict_temp.update({int(id): [a3d]})
#                     C0_aruco_3d_dict = {}
#                     for k, l_v in C0_aruco_3d_dict_temp.items():
#                          C0_aruco_3d_dict.update({k: np.mean(l_v, axis=0)})
#                for id, a3ds in C0_aruco_3d_dict_temp.items():
#                     C0_aruco_3d_dict[int(id)] = np.mean(a3ds, axis=0).tolist()
#           else:
#                C0_aruco_3d_dict = {}
#                for id, a3d in self.predef_arcuo_SCS.items():
#                     C0_aruco_3d_dict.update({int(id): a3d})
#                ids = [int(x) for x in list(self.predef_arcuo_SCS.keys())]       
#           return C0_aruco_3d_dict,  ids  

#      def compute_cube_pos(self, cube:CubePosture):
#           if_2d = True

#           C0_aruco_3d_dict = {}
#           model_index_dict = read_json_as_int_ndarray(os.path.join(self.directory, "category_idx_range.json"), np.int32)
#           C0_aruco_3d_dict, ids  = self.calc_C0_aruco_3d_dict(model_index_dict)

#           data_recorder = Recorder(self.directory, os.path.join(os.path.abspath(os.path.join(self.directory, "..")), "models"))
#           all_not_verified_index = []

#           # 记录每个面的坐标
#           i_range = model_index_dict["7x7cube"]
#           cube_aruco_3d_C0_dict = {}
#           for Filenum in range(*i_range):
#                cad, depth = self.get_images(Filenum)
#                print("\r{:>4d}/{:>4d}".format(Filenum+1, i_range[1] - i_range[0]), end="")
#                # 根據幀類型選擇内參
#                cali_camera_intr = self.cali_camera_intr.dict
#                # 计算变换矩阵
#                if if_2d:
#                     transform, _ = GtPostureComputer.get_T_2d(cad, depth, cali_camera_intr, C0_aruco_3d_dict)
#                     # 计算cube在C0的坐标
#                     scene_aruco_2d, ids = self.get_aruco_coord(cad, depth, cali_camera_intr, get3d = not if_2d)
#                     cube_aruco_2d = np.reshape(scene_aruco_2d[ids <= 12], (-1, 2))
#                     cube_id = ids[ids <= 12]
#                     # 计算cube在Cn的位姿矩阵
#                     aruco_3d_O = self.get_arucos_from_dict(cube.aruco_O_dict, cube_id)
#                     aruco_3d_O = np.reshape(aruco_3d_O, (-1, 3))
#                     cube_aruco_2d = np.reshape(cube_aruco_2d, (-1, 2))
#                     fx = cali_camera_intr["fx"]
#                     fy = cali_camera_intr["fy"]
#                     ppx = cali_camera_intr["ppx"]
#                     ppy = cali_camera_intr["ppy"]
#                     cameraMatrix = np.array([[fx, 0, ppx], [0, fy, ppy], [0,0,1]])
#                     distCoeffs = np.array([0.0,0,0,0,0])
#                     _, rvec, tvec = cv2.solvePnP(aruco_3d_O, cube_aruco_2d, cameraMatrix, distCoeffs)
#                     posture = Posture(rvec = rvec, tvec = np.squeeze(tvec))
#                     T_cube_O2Cn = posture.trans_mat
#                     # 变换到C0，并记录
#                     cube_aruco_3d_C0 = np.linalg.multi_dot((transform, T_cube_O2Cn, homo_pad(aruco_3d_O).T)).T[:, :3]
#                     cube_aruco_3d_C0 = np.reshape(cube_aruco_3d_C0, (-1, 4, 3))
#                     for corners, id in zip(cube_aruco_3d_C0, cube_id):
#                          try:
#                               cube_aruco_3d_C0_dict[id].append(corners.tolist())
#                          except KeyError:
#                               cube_aruco_3d_C0_dict.update({int(id): []})
#                               cube_aruco_3d_C0_dict[id].append(corners.tolist())
#                else:
#                     transform, (scene_aruco_3d, C0_aruco_3d, re_trans, ids) = GtPostureComputer.get_T_3d(cad, depth, cali_camera_intr, C0_aruco_3d_dict, tol = 0.02, return_coords=True)
#                     errors = np.linalg.norm(np.abs(re_trans - C0_aruco_3d), axis = -1)
#                     errors = errors[np.argsort(errors)][:int(len(errors)*0.8)]
#                     if np.any(errors > 0.004) and np.mean(errors) > 0.002:
#                          continue
#                     # 计算cube在C0的坐标
#                     scene_aruco_3d, ids = self.get_aruco_coord(cad, depth, cali_camera_intr, get3d = True)
#                     cube_aruco_3d = np.reshape(scene_aruco_3d[ids <= 12], (-1, 3))
#                     cube_id = ids[ids <= 12]
#                     cube_aruco_3d_C0 = transform.dot(homo_pad(cube_aruco_3d).T).T[:, :3]
#                     cube_aruco_3d_C0 = np.reshape(cube_aruco_3d_C0, (-1, 4, 3))
#                     for corners, id in zip(cube_aruco_3d_C0, cube_id):
#                          try:
#                               cube_aruco_3d_C0_dict[id].append(corners.tolist())
#                          except KeyError:
#                               cube_aruco_3d_C0_dict.update({int(id): []})
#                               cube_aruco_3d_C0_dict[id].append(corners.tolist())
#           # cube_aruco_3d_C0_dict_json = {}
#           # for k, v in cube_aruco_3d_C0_dict.items():
#           #      cube_aruco_3d_C0_dict_json.update({k: v.tolist()})
#           dump_int_ndarray_as_json(cube_aruco_3d_C0_dict, os.path.join(self.directory, "cube_aruco_3d_C0_dict.json"))
#           # 求解
#           cube = CubePosture(self.directory)
#           cube.load_cube_aruco_3d_C0_dict(cube_aruco_3d_C0_dict)
#           T = cube.solve_cube_pos()
#           return T
#           # T= self.solve_cube_pos(cube_aruco_3d_C0_dict)

#      def update_base_marker(self, cube_arucos):
#           for k, v in cube_arucos.items():
#                self.C0_aruco_3d_dict.update({k: v.tolist()})

#      def registration_with_marker(self):
#           if_2d = True

#           C0_aruco = []
#           C0_ids = []
#           C0_aruco_3d_dict = {}
#           Ts = []
#           frames_aruco_3d = []
#           model_index_dict = read_json_as_int_ndarray(os.path.join(self.directory, "category_idx_range.json"), np.int32)
#           # C0_aruco_3d_dict, ids  = self.calc_C0_aruco_3d_dict(model_index_dict)
#           C0_aruco_3d_dict = self.C0_aruco_3d_dict
#           data_recorder = Recorder(self.directory, os.path.join(os.path.abspath(os.path.join(self.directory, "..")), "models"))
#           all_not_verified_index = []
#           all_file_num = int(len(glob.glob1(self.directory+RGB_DIR,"*.jpg")))

#           frames_max_error = []
#           frames_mean_error = []
#           aruco_used_nums = {}
#           # for Filenum in file_gen:
#           for name, i_range in model_index_dict.items():
#                part_aun = {} #aruco 被每个零件使用的次数
#                for Filenum in range(*i_range):
#                     cad, depth = self.get_images(Filenum)
#                     print("\r{:>4d}/{:>4d}".format(Filenum+1, all_file_num), end="")
#                     # 根據幀類型選擇内參
#                     if name != FRAMETYPE_DATA:
#                          cali_camera_intr = self.cali_camera_intr.dict
#                     else:
#                          cali_camera_intr = self.data_camera_intr.dict
#                     # 计算变换矩阵
#                     if if_2d:
#                          transform, (_, _, _, ids) = GtPostureComputer.get_T_2d(cad, depth, cali_camera_intr, C0_aruco_3d_dict)
#                     else:
#                          transform, (scene_aruco_3d, C0_aruco_3d, re_trans, ids) = GtPostureComputer.get_T_3d(cad, depth, cali_camera_intr, C0_aruco_3d_dict, tol = 0.02, return_coords=True)
#                          ### 投影到rgb上观阅 ###
#                          # CAM_WID, CAM_HGT    = camera_intrinsics["width"], camera_intrinsics["height"] # 重投影到的深度图尺寸
#                          # CAM_FX, CAM_FY      = camera_intrinsics["fx"], camera_intrinsics["fy"]  # fx/fy
#                          # CAM_CX, CAM_CY      = camera_intrinsics["ppx"], camera_intrinsics["ppy"]  # cx/cy
#                          # EPS = 1e-16
#                          # MAX_DEPTH = 4.0
#                          # depth_images = []
#                          # extr_trans_list = []
#                          # pc = homo_pad(scene_aruco_3d.reshape((-1, 3)))
#                          # pc_std = np.linalg.inv(transform).dot(homo_pad(C0_aruco_3d).T).T[:, :3]
#                          # z = pc[:, 2]
#                          # # 点云反向映射到像素坐标位置
#                          # u,v = FrameMeta.proj(CAM_FX, CAM_FY, CAM_CX, CAM_CY, pc, z)
#                          # z_std = pc_std[:, 2]
#                          # u_std, v_std = FrameMeta.proj(CAM_FX, CAM_FY, CAM_CX, CAM_CY, pc_std, z_std)
#                          # rgb = cv2.imread(os.path.join(directory, RGB_DIR, str(Filenum).rjust(6, "0") + ".jpg"))
#                          # plt.clf()
#                          # plt.imshow(rgb)
#                          # plt.scatter(u, v, c = 'r')
#                          # plt.scatter(u_std, v_std, c = 'b')
#                          # plt.show()
#                          ### 结束 ###
#                          errors = np.linalg.norm(np.abs(re_trans - C0_aruco_3d), axis = -1)
#                          errors = errors[np.argsort(errors)][:int(len(errors)*0.8)]
#                          if Filenum < model_index_dict["local_base_frames"][1]: 
#                               if np.any(errors > 0.005) and np.mean(errors) > 0.003:
#                                    all_not_verified_index.append(Filenum)                    
#                          else:
#                               if np.any(errors > 0.004) and np.mean(errors) > 0.002:
#                                    all_not_verified_index.append(Filenum)  
#                          frames_max_error.append(np.max(errors))     
#                          frames_mean_error.append(np.mean(errors))                
#                     for id in ids:
#                          try:
#                               part_aun[id] += 1
#                          except KeyError:
#                               part_aun.update({int(id): 1})
#                     Ts.append(transform)
#                aruco_used_nums.update({name: part_aun})
#           frames_max_error = np.array(frames_max_error)
#           frames_mean_error = np.array(frames_mean_error)
#           array_anvi = np.array(all_not_verified_index)
#           all_verified_index = np.setdiff1d(np.arange(len(Ts)), all_not_verified_index)
#           print()
#           base_frames_end = model_index_dict["local_base_frames"][1]
#           if not if_2d:
#                print("未通过核验数量：{}/{}".format(len(all_not_verified_index), all_file_num))
#                print("基准帧最大误差：", ", ".join(["{:>0.4f}".format(x) for x in frames_max_error[:base_frames_end]]))
#                print("基准帧平均误差：", ", ".join(["{:>0.4f}".format(x) for x in frames_max_error[:base_frames_end]]))
#                print("物体帧最大误差：", "{:>0.4f}".format(np.max(frames_max_error[all_verified_index])))
#                print("物体帧平均误差：", "{:>0.4f}".format(np.mean(frames_mean_error[all_verified_index])))
#                print("未通过核验帧：")
               
#                for name, i_range in model_index_dict.items():
#                     print("{:<20s}".format(name+':'), end="")
#                     anvi = array_anvi[np.where((array_anvi<i_range[1]) * (array_anvi>=i_range[0]))]
#                     print("{:<10s}".format("{}/{}".format(len(anvi), i_range[1] - i_range[0])), end="")
#                     print(",".join(["{:>4d}".format(x) for x in anvi]))
#           else:
#                pass
#           ok = input("请确认是否继续删除未通过核验的帧，该操作无法恢复，确认请输入y, 不删除而继续保存输入c， 放弃输入n：")
#           if ok == 'y' or ok == 'c':
#                dump_int_ndarray_as_json(C0_aruco_3d_dict, os.path.join(self.directory, "refined_arcuo_points.json"))
#                Ts = np.array(Ts)
#                if ok == 'y':
#                     Ts = Ts[all_verified_index]
#                     data_recorder.delete(all_not_verified_index)
#                ffr_0, ffr_1 = data_recorder.model_index_dict["global_base_frames"]
#                base_frames_mean_error = frames_mean_error[:base_frames_end]
#                if ffr_1 - ffr_0 > 1 and not if_2d:
#                     base_frames_mean_error = np.array(base_frames_mean_error)
#                     gbf_errors_error = base_frames_mean_error[ffr_0: ffr_1]
#                     min_error_index = int(np.argmin(gbf_errors_error))
#                     data_recorder.rename_all(exchange_pair=[(min_error_index, 0)])
#                     Ts[min_error_index], Ts[0] = Ts[0], Ts[min_error_index]
#                else:
#                     pass

#                filename = os.path.join(self.directory, 'transforms.npy')
#                np.save(filename, Ts)
#                print("Transforms saved")
#                dump_int_ndarray_as_json(aruco_used_nums, (os.path.join(self.directory, "aruco_used_times.json")))
#                return Ts
#           else:
#                pass

#      def save_Ts_as_files(self, Ts):
#           folder_path = os.path.join(self.directory, TRANS_DIR)
#           for filename in os.listdir(folder_path):
#                file_path = os.path.join(folder_path, filename)
#                try:
#                     if os.path.isfile(file_path) or os.path.islink(file_path):
#                          os.remove(file_path)
#                     elif os.path.isdir(file_path):
#                          pass
#                except Exception as e:
#                     pass
#           for data_i, T in enumerate(Ts):
#                filename = os.path.join(self.directory, TRANS_DIR, str(data_i).rjust(6, "0")+".npy")
#                np.save(filename, T)

