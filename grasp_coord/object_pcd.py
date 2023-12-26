from posture_6d.core.posture import Posture
from posture_6d.data.mesh_manager import MeshMeta, MeshManager

import numpy as np
import open3d as o3d
import os
import time
import matplotlib.pyplot as plt 

from . import MODELS_DIR, PCD_MODELS
from .gripper import Gripper, MyThreeFingerGripper

def create_ObjectPcd_from_file(class_id):
    mm = MeshManager(MODELS_DIR, PCD_MODELS)
    return ObjectPcd(mm.export_meta(class_id))

class ObjectPcd(MeshMeta):
    '''
    物体点云
    '''
    def __init__(self, 
                 modelinfo:MeshMeta,
                 posture:Posture = None) -> None:
        super().__init__(   modelinfo.mesh, 
                            modelinfo.bbox_3d, 
                            modelinfo.symmetries, 
                            modelinfo.diameter, 
                            modelinfo.ldmk_3d,
                            modelinfo.name,
                            modelinfo.class_id)
        self.posture_WCS: Posture = posture if posture is not None else Posture()
        self.read_candidate_coord()

    def render(self, gripper:MyThreeFingerGripper = None, gripper_posture_O:Posture = None, u = None):
        '''
        brief
        -----
        渲染，生成TriangleMesh对象用于显示

        parameter
        -----
        gripper: 夹持器对象

        gripper_posture_O: 夹持器姿态
        
        u: 夹持参数

        如果传入了gripper，则必须指定gripper_posture_O和u
        '''
        assert (gripper and gripper_posture_O and u) or not gripper # 检查输入

        showing_geometrys = []
        obj_geo = self.transform(self.posture_WCS)
        if len(obj_geo.mesh.triangle_normals) == 0:
            obj_geo.mesh.compute_triangle_normals()
        sence_center_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=40, origin=self.mesh.get_center())  #场景中心标架
        sence_robot_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=40, origin=[0,0,0])                  #原点标架
        showing_geometrys +=  [obj_geo.mesh, sence_center_frame, sence_robot_frame]

        if gripper:
            gripper_posture = self.posture_WCS * gripper_posture_O # * Posture(tvec=[0,0,-gripper.finger_gripping_bottom[2]])
            showing_geometrys += gripper.render(gripper_posture, u)

        return showing_geometrys

    def draw_all(self, gripper):
        '''
        绘制所有夹持姿态
        '''
        for ggp in self.candi_coord_parameter:
            print(ggp[7])
            self.draw(gripper, Posture(rvec=ggp[:3], tvec=ggp[3:6]), ggp[6])

    def draw(self, gripper:MyThreeFingerGripper = None, gripper_posture_O:Posture = None, u = None):
        '''
        绘制一个夹持姿态
        '''
        showing_geometrys = self.render(gripper, gripper_posture_O, u)
        o3d.visualization.draw_geometries(showing_geometrys, width=1280, height=720)

    @property
    def pcd_WCS(self):
        '''
        世界坐标系(WCS)下的点云
        '''
        pcd = self.points_array
        return self.posture_WCS * pcd

    @property
    def pointcloud_size(self):
        pcd = self.points_array
        return np.max(pcd[:,:3], 0) - np.min(pcd[:,:3], 0)
    
    def read_candidate_coord(self):
        '''
        从硬盘读取候选夹持坐标
        '''
        try:
            path = os.path.join(MODELS_DIR, self.name + "_candi_grasp_posture.npy")
            self.candi_coord_parameter = np.load(path)
        except FileNotFoundError:
            self.candi_coord_parameter = np.zeros((0,8))
            print("Can not find {}".format(path))
    
    def write_candidate_coord(self):
        '''
        将候选夹持坐标写入硬盘
        '''
        path = os.path.join(MODELS_DIR, self.name + "_candi_grasp_posture.npy")
        np.save(path, self.candi_coord_parameter)

    def add_candidate_coord(self, rvec, tvec, u, score, *, replace = None):
        '''
        添加候选夹持坐标
        '''
        rvec = np.array(rvec).reshape((1,3))
        tvec = np.array(tvec).reshape((1,3))
        u = np.array(u).reshape((1,1))
        score = np.array(score).reshape((1,1))
        if isinstance(replace, int):
            self.candi_coord_parameter[replace] = np.hstack([rvec, tvec, u, score]).squeeze(0)
        else:
            self.candi_coord_parameter = np.vstack([self.candi_coord_parameter, np.hstack([rvec, tvec, u, score])])

    def parse_candidate_coord(self, idx = None):
        '''
        解析夹持坐标
        '''
        if idx is None:
            candi_coord_parameter = self.candi_coord_parameter
        elif isinstance(idx, int):
            candi_coord_parameter = self.candi_coord_parameter[idx:idx+1, :] #[1, 8]
        else:
            raise TypeError("None or int")

        rvec = candi_coord_parameter[:, 0:3]
        tvec = candi_coord_parameter[:, 3:6]
        u = candi_coord_parameter[:, 6]
        score = candi_coord_parameter[:, 7]

        rposture = Posture(rvec=self.posture_WCS.rvec)
        direction = np.stack([rposture * Posture(rvec=rvec[i]) * np.array([0,0,1]) for i in range(len(rvec))])
        return (rvec, tvec, u, score, direction)