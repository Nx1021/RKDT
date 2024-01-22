from .gripper import MyThreeFingerGripper
from .object_pcd import ObjectPcd, create_ObjectPcd_from_file

from posture_6d.core.posture import Posture
from posture_6d.core.utils import JsonIO
import numpy as np
import open3d as o3d
import copy
import time

from typing import Union, Any, Optional

from models.results import ImagePosture, ObjPosture
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
    def __init__(self, cfg = None) -> None:
        self.gripper = MyThreeFingerGripper()
        self.object_list:list[ObjectPcd] = []
        self.scene_center = np.array([400,0,0])
        self.cfg = load_yaml(cfg) if cfg is not None else {}

        # self.GCS = Posture(rvec = np.array(0,0,0), tvec = np.array(400,0,0)) #抓取坐标系
        self.log_update = False
        self.logs = {}
        
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
        angle_threshold = 45
        if obj.candi_coord_parameter is None:
            return False, np.zeros((0,7))
        ### filter the directions
        directions = obj.parse_candidate_coord()[4]
        rmat_RCS = obj.posture_WCS.rmat
        directions_RCS = np.eye(3).dot(directions.T).T # np.linalg.inv(rmat_RCS).dot(directions.T).T
        grasp_vec = np.array([0,0,-1])
        angles = np.arccos(np.sum(directions_RCS * grasp_vec, axis=-1))
        ok_index = np.where(angles < angle_threshold * np.pi / 180)[0]
        if ok_index.size == 0:
            return False, np.zeros((0,7))
        else:
            # score_argsort = np.argsort(obj.parse_candidate_coord()[3][ok_index])
            score_argsort = np.argsort(angles[ok_index])
            parameter = obj.candi_coord_parameter[ok_index[score_argsort]]
            return True, parameter[:, :7]

    def calc_grasp_posture(self, selected_obj: Optional[ObjectPcd] = None, return_near_surf_pointcloud_dict = False):
        '''
        计算抓取位置，默认从全局开始抓取，也可以抓取指定的物体
        若不存在合适的抓取位置，将返回None
        '''
        extra_info:dict[str, Any] = {}

        if selected_obj is None:
            ### 按照到中心位置排序
            sorted_object_list = sorted(self.object_list, key = lambda x:np.linalg.norm(x.posture_WCS.tvec - self.scene_center), reverse = True) #从大到小
        elif isinstance(selected_obj, ObjectPcd):
            sorted_object_list = [selected_obj]
        else:
            return False, None, None, extra_info
        ### 以各初始位置计算新位置
        other_object_list = [o for o in self.object_list if o is not selected_obj]
        obj_centers = np.array([o.posture_WCS.tvec for o in other_object_list]).reshape(-1, 3)
        obj_model_diameters = np.array([o.diameter for o in other_object_list]).reshape(-1)
        for obj in sorted_object_list:
            success, posture_para_list = self.choose_grasp_posture(obj)
            if len(obj_centers) == 0:
                if success:
                    paras = posture_para_list[0]
                    rvec = paras[:3]
                    tvec = paras[3:6]
                    u = paras[6]
                    grasp_posture = Posture(homomat =  obj.posture_WCS.trans_mat.dot(Posture(rvec=rvec, tvec=tvec).trans_mat))
                    if return_near_surf_pointcloud_dict:
                        extra_info["near_surf_pointcloud_dict"] = {}

                    return True, grasp_posture, u, extra_info
                else:
                    continue
            ### 选取附近的物体
            gripper_r = self.gripper.finger_width + self.gripper.finger_gripping_width + self.gripper.finger_gripping_bottom[0]            
            obj_center_distance = np.linalg.norm(obj_centers - obj.posture_WCS.tvec, axis=-1)
            near_index = np.where(obj_center_distance < obj_model_diameters + gripper_r)[0]
            if len(near_index )== 0:
                near_surf_pointcloud = np.zeros((0,3))
            else:
                near_surf_pointcloud = np.vstack([other_object_list[i].pcd_WCS for i in near_index])            
            if success:
                grasp_posture_list:list[Posture] = [None for _ in range(len(posture_para_list))]
                _scores:list[float] = [0.0 for _ in range(len(posture_para_list))]
                if return_near_surf_pointcloud_dict:
                    extra_info["near_surf_pointcloud_dict"] = {other_object_list[i].class_id: other_object_list[i].pcd_WCS for i in near_index}
                for para_i, paras in enumerate(posture_para_list):
                    rvec = paras[:3]
                    tvec = paras[3:6]
                    u:float = paras[6]
                    self.gripper.set_u(u)
                    posture_GinO = Posture(rvec=rvec, tvec=tvec)
                    # near_surf_pointcloud = near_surf_pointcloud[np.linalg.norm(near_surf_pointcloud[:, :3] - grasp_center_R[:3], axis=-1) < gripper_r]
                    # grasp_posture = Posture(homomat =  obj.posture_RCS.trans_mat.dot(Posture(rvec=rvec, tvec=tvec).trans_mat))
                    # self.gripper.posture_WCS = grasp_posture
                    # self.show_scene()
                    interference_array = self.gripper.in_interference_region(
                        posture_GinO, obj.posture_WCS, near_surf_pointcloud)
                    interference_num = np.sum(interference_array)
                    if interference_num/len(interference_array) > 0.01 and interference_num > 5:
                        continue
                    else:
                        # 夹持点距离其他点的距离
                        grasp_posture = Posture(homomat =  obj.posture_WCS.trans_mat.dot(Posture(rvec=rvec, tvec=tvec).trans_mat))
                        grasp_posture_list[para_i] = grasp_posture
                        center = self.gripper.get_grasp_center(grasp_posture)
                        dist = np.linalg.norm(near_surf_pointcloud - center, axis=-1)
                        min_dist = np.min(dist) if len(dist) > 0 else 0
                        _scores[para_i] += np.clip(min_dist, 0, 100) / 100
                grasp_posture = grasp_posture_list[np.argmax(_scores)]
                if grasp_posture is None:
                    return False, None, None, extra_info
                else:
                    return True, grasp_posture, u, extra_info
        return False, None, None, extra_info

    def show_scene(self):
        showing_geometrys = []
        for obj in self.object_list:
            showing_geometrys += obj.render()
        showing_geometrys += self.gripper.render()
        o3d.visualization.draw_geometries(showing_geometrys, width=1280, height=720)

    def update_from_prediction(self, prediction: Union[ImagePosture, list[ObjPosture]]):
        '''
        从网络的预测结果更新场景,
        '''
        new:dict[int, list[Posture]] = {}
        old:dict[int, list[ObjectPcd]] = {}
        ### 收集
        if isinstance(prediction, ImagePosture):
            prediction = prediction.obj_list

        for obj in prediction:
            if obj.posture is not None:
                new.setdefault(obj.class_id, []).append(obj.posture)
        for obj in self.object_list:
            old.setdefault(obj.class_id, []).append(obj)
        union_class_id = set(new.keys()).union(set(old.keys()))
        for class_id in union_class_id:
            new.setdefault(class_id, [])
            old.setdefault(class_id, [])
            if len(new[class_id]) > 0 and len(old[class_id]) > 0:
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
            else:
                for obj in old[class_id]:
                    self.remove_object(obj)
                for obj in new[class_id]:
                    self.add_object(create_ObjectPcd_from_file(class_id), obj)
    
    def log_scene(self, grasp:tuple[np.ndarray, float, int] = None):
        log_i = len(self.logs)

        object_info = {}

        for obj in self.object_list:
            id_ = obj.class_id
            posture = obj.posture_WCS.trans_mat
            object_info[id_] = posture
        
        if grasp is None:
            self.logs[log_i] = {"obj_list": object_info, "grasp_posture": None, "grasp_u": None, "grasp_obj_id": None}
        else:
            self.logs[log_i] = {"obj_list": object_info, "grasp_posture": grasp[0], "grasp_u": grasp[1], "grasp_obj_id": grasp[2]}

    def save_log(self, path):
        JsonIO.dump_json(path, self.logs)

    def load_log(self, path):
        self.logs = JsonIO.load_json(path)


class GraspManipulator():
    def __init__(self, scene:Scene) -> None:
        self.scene = scene
        self.gripper_mesh:list = []
        self.gripper_frame_mesh = None
        # self.base_frame_points = np.array([[0,0,0,1], [1,0,0,1], [0,1,0,1], [0,0,1,1]], np.float32)
        # self.gripper_frame_points = np.array([[0,0,0,1], [1,0,0,1], [0,1,0,1], [0,0,1,1]], np.float32)
        self.cur_gripper_posture = Posture()

        self.object_mesh = None
        self.object:ObjectPcd = None
        self.u = 1.0

        self.step = 1.0
        self.is_refine_mode = True
        self.cur_grasp_idx = -1

    def trans_with_G(self, T_center:Posture, vis):
        '''
        以物体中心的旋转
        '''
        ####
        translate = self.cur_gripper_posture #Posture(rvec = , tvec=center)
        T = np.linalg.multi_dot((translate.trans_mat, 
                                 T_center.trans_mat, 
                                 translate.inv_transmat))      
        self.gripper_transform(vis, T)
    
    def gripper_transform(self, vis, T):
        self.cur_gripper_posture = Posture(homomat=T) * self.cur_gripper_posture
        for m in self.gripper_mesh:
            m.transform(T)  
        self.gripper_frame_mesh.transform(T)  
        # self.gripper_frame_points = self.gripper_frame_points.dot(T.T) #self.gripper_frame_points.dot(np.linalg.inv(T))
        if vis is not None:
            for m in self.gripper_mesh:
                vis.update_geometry(m)
            vis.update_geometry(self.gripper_frame_mesh)

    def rotate_X_inc(self, vis):
        rotate = Posture(rvec=np.array([self.step * np.pi/180, 0, 0]))
        self.trans_with_G(rotate, vis)
    
    def rotate_Y_inc(self, vis):
        rotate = Posture(rvec=np.array([0, self.step * np.pi/180, 0]))
        self.trans_with_G(rotate, vis)

    def rotate_Z_inc(self, vis):
        rotate = Posture(rvec=np.array([0, 0, self.step * np.pi/180]))
        self.trans_with_G(rotate, vis)
    
    def rotate_X_dec(self, vis):
        rotate = Posture(rvec=np.array([-self.step * np.pi/180, 0, 0]))
        self.trans_with_G(rotate, vis)
    
    def rotate_Y_dec(self, vis):
        rotate = Posture(rvec=np.array([0, -self.step * np.pi/180, 0]))
        self.trans_with_G(rotate, vis)

    def rotate_Z_dec(self, vis):
        rotate = Posture(rvec=np.array([0, 0, -self.step * np.pi/180]))
        self.trans_with_G(rotate, vis)

    def translate_X_inc(self, vis):
        posture = Posture(tvec=np.array([self.step * 5, 0, 0]))
        self.trans_with_G(posture, vis)

    def translate_Y_inc(self, vis):
        posture = Posture(tvec=np.array([0, self.step * 5, 0]))
        self.trans_with_G(posture, vis)
    
    def translate_Z_inc(self, vis):
        posture = Posture(tvec=np.array([0, 0, self.step * 5]))
        self.trans_with_G(posture, vis)

    def translate_X_dec(self, vis):
        posture = Posture(tvec=np.array([-self.step * 5, 0, 0]))
        self.trans_with_G(posture, vis)

    def translate_Y_dec(self, vis):
        posture = Posture(tvec=np.array([0, -self.step * 5, 0]))
        self.trans_with_G(posture, vis)
    
    def translate_Z_dec(self, vis):
        posture = Posture(tvec=np.array([0, 0, -self.step * 5]))
        self.trans_with_G(posture, vis)

    def make_straight_down(self, vis):
        # new_posture = Posture(rvec=np.array([0, 0, 0]))
        ### 转换为旋转向量
        src_vec:np.ndarray = self.cur_gripper_posture.rmat.dot(np.array([0, 0, 1]))
        dst_vec:np.ndarray = np.array([0, 0, -1])
        # src_vec = np.tile(src_vec, [dst_vec.shape[0],1]).astype(np.float32)
        times = np.sum( dst_vec * src_vec)
        angle = np.arccos(times) #旋转角度
        rot = np.cross(src_vec, dst_vec)
        if np.linalg.norm(rot) == 0:
            rot = np.array([1,0,0])
        rvec = rot * np.tile(np.expand_dims(angle/ np.linalg.norm(rot), -1), [1,3])
        
        ti_p = Posture(rvec=rvec)
        self.gripper_transform(vis, ti_p.trans_mat)

    def _modify_u(self, vis, u):
        new_u = float(np.clip(u, 0.0, 1.0))
        P_list = self.scene.gripper._get_transmit_mat_for_u(self.u, new_u)
        self.u = new_u
        for i in range(6):
            P = P_list[i]
            T = np.linalg.multi_dot([self.cur_gripper_posture.trans_mat,
                                     P,
                                    self.cur_gripper_posture.inv_transmat])
            self.gripper_mesh[i].transform(T)
        if vis is not None:
            for m in self.gripper_mesh:
                vis.update_geometry(m)

    def inc_u(self, vis):
        self._modify_u(vis, self.u + 0.1)

    def dec_u(self, vis):
        self._modify_u(vis, self.u - 0.1)

    def refine_mode(self, vis):
        if not self.is_refine_mode:
            self.step = 0.1
            self.is_refine_mode = True
        else:
            self.step = 1.0
            self.is_refine_mode = False

    def switch_grasp(self, vis):
        self.cur_grasp_idx += 1
        if (self.cur_grasp_idx == len(self.object.candi_coord_parameter)):
            self.cur_grasp_idx = -1
        else:
            rvec, tvec, u, score, d = self.object.parse_candidate_coord(self.cur_grasp_idx)
            # transitionmatrix = Posture(rvec=rvec, tvec=tvec) * (self.object.posture_WCS.inv() * self.cur_gripper_posture).inv()
            transitionmatrix = self.object.posture_WCS * Posture(rvec=rvec, tvec=tvec) * self.cur_gripper_posture.inv()
            self.gripper_transform(vis, transitionmatrix.trans_mat)
            self._modify_u(vis, u)
        print("cur_grasp_idx:", self.cur_grasp_idx)

    def delete_grasp(self, vis):
        if self.cur_grasp_idx == -1:
            return
        else:
            self.object.candi_coord_parameter = np.delete(self.object.candi_coord_parameter, self.cur_grasp_idx, axis=0)
            self.object.write_candidate_coord()
            self.cur_grasp_idx -= 1
            self.switch_grasp(vis)
            print("delete_grasp:", self.cur_grasp_idx, "grasp left:", len(self.object.candi_coord_parameter))

    def confirm(self, vis):
        # 计算当前的姿态
        
        # P_inG = P_inB * T_BtoG
        # T_B2G = np.linalg.inv(self.base_frame_points) * self.gripper_frame_points
        Posture_B2G = self.cur_gripper_posture

        Posture_B2O = self.object.posture_WCS

        Posture_GinO = Posture_B2O.inv() * Posture_B2G  #PostureO2G

        if self.cur_grasp_idx == -1:
            self.object.add_candidate_coord(Posture_GinO.rvec, Posture_GinO.tvec, self.u, 1.0)
            print("add candidate coord: ", Posture_GinO.rvec, Posture_GinO.tvec, self.u, 1.0)
        else:
            self.object.add_candidate_coord(Posture_GinO.rvec, Posture_GinO.tvec, self.u, 1.0, replace=self.cur_grasp_idx)
            print(f"replace candidate coord at {self.cur_grasp_idx}: ", Posture_GinO.rvec, Posture_GinO.tvec, self.u, 1.0)

        self.object.write_candidate_coord()

    def start_manipulate(self, obj_id):
        '''
        开始人工抓取，选取的抓取姿态将被记录并添加到物体的抓取姿态列表中
        '''
        # 获取场景内的物体
        for obj in self.scene.object_list:
            if obj_id == obj.class_id:
                self.object = obj
                break
        self.scene.gripper.set_u(self.u)
        meshes = self.scene.gripper.render()
        self.gripper_mesh = meshes[:-1]
        self.gripper_frame_mesh = meshes[-1]
        self.gripper_transform(None, self.scene.gripper.posture_WCS.trans_mat)
        obj_mesh = self.object.mesh.transform(self.object.posture_WCS.trans_mat)
        obj_mesh.compute_vertex_normals()
        # 新建一个在原点的坐标系
        frame_B = o3d.geometry.TriangleMesh.create_coordinate_frame(size=60, origin=[0,0,0])

        showing_geometrys = [obj_mesh, *self.gripper_mesh, self.gripper_frame_mesh, frame_B]

        # 注册回调函数
        key_to_callback = {}
        key_to_callback[ord("Q")] = self.rotate_X_inc
        key_to_callback[ord("W")] = self.rotate_X_dec
        key_to_callback[ord("A")] = self.rotate_Y_inc
        key_to_callback[ord("S")] = self.rotate_Y_dec
        key_to_callback[ord("Z")] = self.rotate_Z_inc
        key_to_callback[ord("X")] = self.rotate_Z_dec
        key_to_callback[ord("E")] = self.translate_X_inc
        key_to_callback[ord("R")] = self.translate_X_dec       
        key_to_callback[ord("D")] = self.translate_Y_inc
        key_to_callback[ord("F")] = self.translate_Y_dec       
        key_to_callback[ord("C")] = self.translate_Z_inc
        key_to_callback[ord("V")] = self.translate_Z_dec

        key_to_callback[ord("T")] = self.make_straight_down

        key_to_callback[ord("U")] = self.inc_u
        key_to_callback[ord("I")] = self.dec_u
        key_to_callback[ord("1")] = self.refine_mode

        key_to_callback[ord("N")] = self.confirm
        key_to_callback[ord("J")] = self.switch_grasp
        key_to_callback[ord("K")] = self.delete_grasp
        

        o3d.visualization.draw_geometries_with_key_callbacks(showing_geometrys, 
                    key_to_callback ,width =640, height=480)