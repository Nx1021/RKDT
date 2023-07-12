from posture_6d.posture import Posture
from posture_6d.mesh_manager import MeshMeta, MeshManager

import numpy as np
import open3d as o3d
import os
import time
import matplotlib.pyplot as plt 

from utils.yaml import yaml_load
from grasp_coord import MODELS_DIR, PCD_MODELS
from grasp_coord.gripper import Gripper, MyThreeFingerGripper

def create_ObjectPcd_from_file(class_id):
    mm = MeshManager(MODELS_DIR, PCD_MODELS)
    return ObjectPcd(mm.export_meta(class_id))

class ObjectPcd(MeshMeta):
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
        assert (gripper and gripper_posture_O and u) or not gripper

        showing_geometrys = []
        obj_geo = self.transform(self.posture_WCS)
        if len(obj_geo.mesh.triangle_normals) == 0:
            obj_geo.mesh.compute_triangle_normals()
        sence_center_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=40, origin=self.mesh.get_center())
        sence_robot_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=40, origin=[0,0,0])
        showing_geometrys +=  [obj_geo.mesh, sence_center_frame, sence_robot_frame]

        if gripper:
            gripper_posture = self.posture_WCS * gripper_posture_O # * Posture(tvec=[0,0,-gripper.finger_gripping_bottom[2]])
            showing_geometrys += gripper.render(gripper_posture, u)

        return showing_geometrys

    def draw_all(self, gripper):
        for ggp in self.candi_coord_parameter:
            self.draw(gripper, Posture(rvec=ggp[:3], tvec=ggp[3:6]), ggp[6])

    def draw(self, gripper:MyThreeFingerGripper = None, gripper_posture_O:Posture = None, u = None):
        showing_geometrys = self.render(gripper, gripper_posture_O, u)
        o3d.visualization.draw_geometries(showing_geometrys, width=1280, height=720)

    def read_local_grasp_coord(self):
        pass

    @property
    def pcd_WCS(self):
        pcd = self.pcd
        return self.posture_WCS * pcd

    @property
    def pointcloud_size(self):
        pcd = self.pcd
        return np.max(pcd[:,:3], 0) - np.min(pcd[:,:3], 0)
    
    def read_candidate_coord(self):
        try:
            path = os.path.join(MODELS_DIR, self.name + "_candi_grasp_posture.npy")
            self.candi_coord_parameter = np.load(path)
        except FileNotFoundError:
            self.candi_coord_parameter = np.zeros((0,8))
            print("Can not find {}".format(path))

    def parse_candidate_coord(self):
        candi_coord_parameter = self.candi_coord_parameter
        rvec = candi_coord_parameter[:, 0:3]
        tvec = candi_coord_parameter[:, 3:6]
        u = candi_coord_parameter[:, 6]
        score = candi_coord_parameter[:, 7]

        rposture = Posture(rvec=self.posture_WCS.rvec)
        direction = np.stack([rposture * Posture(rvec=rvec[i]) * np.array([0,0,1]) for i in range(len(rvec))])
        return (rvec, tvec, u, score, direction)