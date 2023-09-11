from .gripper import MyThreeFingerGripper
from .object_pcd import ObjectPcd, create_ObjectPcd_from_file

from posture_6d.core.posture import Posture
import numpy as np
import open3d as o3d
import copy
import time

from models.results import ImagePosture
from utils.match_with_threshold import perform_matching
from utils.yaml import load_yaml
from scipy.spatial import distance


class ObjectPostureHistory():
    record_num = 5
    translate_max_tolerance = 40
    def __init__(self) -> None:
        self.obj_id = -1
        self.obj:ObjectPcd = None
        self.history = []
    
    def update(self, posture_paras):
        mean_value = self.mean()
        if np.linalg.norm(posture_paras[3:6] - mean_value[3:6]) > ObjectPostureHistory.translate_max_tolerance:
            self.history.clear()
        self.history.append(posture_paras)
        if len(self.history) > ObjectPostureHistory.record_num:
            self.history.pop(0) 

    def mean(self):
        return np.mean(self.history, axis=0)


class Scene:
    '''
    场景类，管理所有物体的点云
    '''    
    def __init__(self, cfg) -> None:
        self.gripper = MyThreeFingerGripper()
        self.object_list:list[ObjectPcd] = []
        self.scene_center = np.array([400,0,0])
        self.cfg = load_yaml(cfg)
        # self.GCS = Posture(rvec = np.array(0,0,0), tvec = np.array(400,0,0)) #抓取坐标系
        
    def set_gripper(self, gripper):
        self.gripper = gripper
    
    def add_object(self, object:ObjectPcd, posture:Posture):
        object.posture_WCS = posture
        self.object_list.append(object)
    
    def remove_object(self, object:ObjectPcd):
        if isinstance(object, ObjectPcd):
            try:
                self.object_list.remove(object)
            except ValueError:
                pass
        elif isinstance(object, str):
            pass

    def random_place_objects(self, x_range = (-100,100), y_range = (-100,100), z_range = (-100, 100), random_rot = True):
        pass

    def clear_all_object(self):
        self.object_list.clear()
        
    def choose_grasp_posture(self, obj:ObjectPcd):
        '''
        parameter:
        ------
        obj:ObjectPcd

        return
        -----
        success: bool
        grasp_posture:Posture
        '''
        angle_threshold = 30
        if obj.candi_coord_parameter is None:
            return False, np.zeros((0,7))
        ### filter the directions
        directions = obj.parse_candidate_coord()[4]
        rmat_RCS = obj.posture_WCS.rmat
        directions_RCS = rmat_RCS.dot(directions.T).T
        grasp_vec = np.array([0,0,-1])
        angles = np.arccos(np.sum(directions_RCS * grasp_vec, axis=-1))
        ok_index = np.where(angles < angle_threshold * np.pi / 180)[0]
        if ok_index.size == 0:
            return False, np.zeros((0,7))
        else:
            score_argsort = np.argsort(obj.parse_candidate_coord()[3][ok_index])
            parameter = obj.candi_coord_parameter[ok_index[score_argsort]]
            return True, parameter[:, :7]

    def calc_grasp_posture(self, selected_obj: ObjectPcd = None):
        '''
        计算抓取位置，默认从全局开始抓取，也可以抓取指定的物体
        若不存在合适的抓取位置，将返回None
        '''
        if selected_obj is None:
            ### 按照到中心位置排序
            sorted_object_list = sorted(self.object_list, key = lambda x:np.linalg.norm(x.posture_WCS.tvec - self.scene_center), reverse = True) #从大到小
        elif isinstance(selected_obj, ObjectPcd):
            sorted_object_list = [selected_obj]
        else:
            return False, None, None
        ### 以各初始位置计算新位置
        obj_centers = np.array([o.posture_WCS.tvec for o in self.object_list])
        obj_model_diameters = np.array([o.diameter for o in self.object_list])
        for obj in sorted_object_list:
            success, posture_para_list = self.choose_grasp_posture(obj)
            ### 选取附近的物体
            gripper_r = self.gripper.finger_width + self.gripper.finger_gripping_width + self.gripper.finger_gripping_bottom[0]            
            obj_center_distance = np.linalg.norm(obj_centers - obj.posture_WCS.tvec, axis=-1)
            near_index = np.where(obj_center_distance < obj_model_diameters + gripper_r)[0]
            near_surf_pointcloud = np.vstack([self.object_list[i].pcd_WCS for i in near_index])            
            if success:
                for paras in posture_para_list:
                    rvec = paras[:3]
                    tvec = paras[3:6]
                    u = paras[6]
                    self.gripper.set_u(u)
                    posture_GinO = Posture(rvec=rvec, tvec=tvec)
                    # near_surf_pointcloud = near_surf_pointcloud[np.linalg.norm(near_surf_pointcloud[:, :3] - grasp_center_R[:3], axis=-1) < gripper_r]
                    # grasp_posture = Posture(homomat =  obj.posture_RCS.trans_mat.dot(Posture(rvec=rvec, tvec=tvec).trans_mat))
                    # self.gripper.posture_WCS = grasp_posture
                    # self.show_scene()
                    interference_array = self.gripper.in_interference_region(
                        posture_GinO, obj.posture_WCS, near_surf_pointcloud)
                    interference_num = np.sum(interference_array)
                    if interference_num > 5:
                        continue
                    else:
                        grasp_posture = Posture(homomat =  obj.posture_WCS.trans_mat.dot(Posture(rvec=rvec, tvec=tvec).trans_mat))
                        return True, grasp_posture, u
        return False, None, None 

    def show_scene(self):
        showing_geometrys = []
        for obj in self.object_list:
            showing_geometrys += obj.render()
        showing_geometrys += self.gripper.render()
        o3d.visualization.draw_geometries(showing_geometrys, width=1280, height=720)

    def update_from_prediction(self, prediction: ImagePosture):
        '''
        从网络的预测结果更新场景
        '''
        new:dict[int, list[Posture]] = {}
        old:dict[int, list[ObjectPcd]] = {}
        ### 收集
        for obj in prediction.obj_list:
            if obj.posture is not None:
                new.setdefault(obj.class_id, []).append(obj.posture)
        for obj in self.object_list:
            old.setdefault(obj.class_id, []).append(obj)
        union_class_id = set(new.keys()).union(set(old.keys()))
        for class_id in union_class_id:
            new.setdefault(class_id, [])
            old.setdefault(class_id, [])
            A = np.stack([x.posture_WCS.tvec for x in old[class_id]])
            B = np.stack([x.tvec for x in new[class_id]])

            (matched_pairs,
                unmatched_A,
                unmatched_B) = perform_matching(A, B, 100, distance.cdist)
            
            # 修改匹配成功的
            for mp in matched_pairs:
                i:int = mp[0]
                j:int = mp[1]
                old[class_id][i].posture_WCS = new[class_id][j]
            # 删除old中没有匹配成功的
            for um in unmatched_A:
                self.remove_object(old[class_id][um])
            # 添加new中没有匹配成功的
            for um in unmatched_B:
                self.add_object(create_ObjectPcd_from_file(class_id), new[class_id][um])