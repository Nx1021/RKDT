import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
from posture_6d.core.posture import Posture

from sko.GA import GA

class Gripper():
    def __init__(self, finger_num = 3, finger_length = 40, finger_width = 10, finger_gripping_length = 20, finger_gripping_width = 10, finger_thickness = 3) -> None:
        self.posture_WCS = Posture()
        
        self.finger_num = finger_num #手指数量
        self.finger_length = finger_length
        self.finger_width = finger_width
        self.finger_gripping_length = finger_gripping_length
        self.finger_gripping_width = finger_gripping_width
        self.finger_thickness = finger_thickness
        
        self.finger_trans_mat:np.ndarray = np.eye(4)
        self.u = 0
        self.set_u(self.u)
        self.u_list = np.linspace(0,1,51)
        self.r_list = []
        for u in self.u_list:
            self.set_u(u)
            self.r_list.append(abs(self.finger_gripping_center[0]))
        self.r_list = np.expand_dims(np.array(self.r_list), -1)

    @property
    def finger_gripping_center(self):
        point = np.array([-self.finger_gripping_width, 0, -self.finger_gripping_length/2, 1])
        return self.finger_trans_mat.dot(point)

    @property
    def finger_gripping_bottom(self):
        point = np.array([-self.finger_gripping_width, 0, 0, 1])
        return self.finger_trans_mat.dot(point)

    @property
    def max_grasp_depth(self):
        return self.finger_length

    def calc_finger_trans_mat(self, u):
        return np.eye(4)

    def get_u(self, r):
        r_list = np.tile(self.r_list, (1, len(r)))
        ok_index = np.sum(r_list < r, axis=0)
        # if ok_index.size == 0:
        #     u_index = -1
        # else:
        #     u_index = min(np.where(self.r_list > r)[0])
        return self.u_list[ok_index]

    def set_u(self, u):
        self.u = u
        self.finger_trans_mat = self.calc_finger_trans_mat(u)

class MyThreeFingerGripper(Gripper):
    '''
    夹爪空间，通过定义若干参数来表示夹爪的运行空间，包括：夹持区域、干涉区域、无关区域
    坐标系与机器人末端坐标系重合
    '''
    def __init__(self, finger_num=3, finger_length=40, finger_width=10, finger_gripping_length=20, finger_gripping_width=10, finger_thickness=3) -> None:
        self.rod = 60
        self.center_distance = 15   
        self.z_offset = 213.64     
        finger_length = 47
        finger_width = 1
        finger_gripping_length = 33.5
        finger_gripping_width = 18
        super().__init__(finger_num, finger_length, finger_width, finger_gripping_length, finger_gripping_width, finger_thickness)

    @property
    def max_grasp_depth(self):
        angle = self.u * np.pi/3
        extra_h = self.finger_gripping_width / np.math.tan(angle)
        max_grasp_depth = self.finger_length + extra_h
        return min(max_grasp_depth, self.finger_gripping_center[2])

    def get_grasp_center(self, object_trans_mat, gripper_trans_mat):
        center_G = np.array([0, 0, self.finger_gripping_bottom[2], 1])
        center_R = np.linalg.multi_dot((object_trans_mat, gripper_trans_mat, center_G))
        return center_R

    def calc_finger_trans_mat(self, u):
        angle = u * np.pi/3
        posture_1 = Posture(tvec=[self.center_distance, 0, 0])
        posture_2 = Posture(tvec=[self.rod*np.math.sin(angle), 0, self.rod*np.math.cos(angle)])
        posture_3 = Posture(tvec=[0, 0, self.finger_length + self.z_offset])
        return np.linalg.multi_dot((posture_1.trans_mat, posture_2.trans_mat, posture_3.trans_mat))

    def in_grasp_region(self, pointcloud):
        '''
        brief
        -----
        判断所有点是否在抓取区域内

        parameter
        -----
        pointcloud: np.ndarray [N, 3]

        return
        ------
        in_array: np.ndarray(bool), [N]
        '''
        ### 根据u计算夹持区域半径、位置
        r, z = self._grasp_r, self._grasp_z
        points_r:np.ndarray = np.linalg.norm(pointcloud[:,0:2], axis = -1) #[N]
        r_in = points_r < r

        z_min = - self.claw_length/2
        z_max = self.claw_length/2
        z_in = (pointcloud[:,2] >  z_min) * (pointcloud[:,2] < z_max)

        in_array = r_in * z_in
        return in_array

    def in_interference_region(self, gripper_posture:Posture, object_posture:Posture, pointcloud_R, include_claw = True):
        '''
        brief
        -----
        判断所有点是否在干涉区域内
        干涉区域分为2个部分：
        1.z>0
        2.夹爪的运动空间

        parameter
        -----
        gripper_trans_mat: np.ndarray [N, 4] 夹爪在物体坐标系下的姿态:
        object_trans_mat:  np.ndarray [N, 4] 物体在机器人坐标系下的姿态:
        pointcloud: np.ndarray [N, 3]

        return
        ------
        in_array: np.ndarray(bool), [N]
        '''
        in_array = np.zeros(pointcloud_R.shape[0], np.bool_)

        pointcloud_O = (object_posture.inv()* pointcloud_R)
        pointcloud_Gpr = (gripper_posture.inv() * pointcloud_O)
        
        global_min_z = self.z_offset
        outof_z = pointcloud_Gpr[:,2] < global_min_z
        in_array = in_array + outof_z
        # exclude those point that is obviously not interference with the gripper
        global_max_r = self.finger_gripping_bottom[0] + self.finger_width + self.finger_gripping_width
        global_max_z = self.finger_trans_mat[2,3]
        possible_itrfrnc_index = np.where((np.linalg.norm(pointcloud_Gpr[:,:2], axis=-1) < global_max_r) * \
                                        (pointcloud_Gpr[:,2] < global_max_z) * \
                                        (pointcloud_Gpr[:,2] > global_min_z))[0]
        pointcloud_itfrc = pointcloud_Gpr[possible_itrfrnc_index, :]
        # 手指均布，包含了二指和三指的情况
        for phi in np.linspace(0, np.pi*2, self.finger_num, False):
            finger_rot_trans = Posture(rvec = np.array([0,0,phi])).trans_mat
            finger_trans = np.dot(finger_rot_trans, self.finger_trans_mat)
            finger_trans_posture = Posture(homomat=finger_trans)
            pointcloud_Finger = finger_trans_posture.inv() * pointcloud_itfrc
            y_min = -self.finger_thickness / 2
            y_max = self.finger_thickness / 2
            in_y_local_index = np.where((pointcloud_Finger[:,1] < y_max) *( pointcloud_Finger[:,1] > y_min))[0]
            in_y_array = pointcloud_Finger[in_y_local_index]
            finger_x_min = 0
            finger_x_max = self.finger_width
            finger_z_min = -self.finger_length
            finger_z_max = 0
            finger_gripping_x_min = -self.finger_gripping_width
            finger_gripping_x_max = 0
            finger_gripping_z_min = -self.finger_gripping_length
            finger_gripping_z_max = 0
            in_finger = np.all(in_y_array[:,(0,2)] < np.array([finger_x_max, finger_z_max]), axis=-1) * \
                        np.all(in_y_array[:,(0,2)] > np.array([finger_x_min, finger_z_min]), axis=-1)
            in_finger_gripping =    np.all(in_y_array[:,(0,2)] < np.array([finger_gripping_x_max, finger_gripping_z_max]), axis=-1) * \
                                    np.all(in_y_array[:,(0,2)] > np.array([finger_gripping_x_min, finger_gripping_z_min]), axis=-1)
            global_idx = possible_itrfrnc_index[in_y_local_index]
            in_array[global_idx] = in_array[global_idx] + in_finger + in_finger_gripping
            # in_array[possible_itfrc_index] = in_array[possible_itfrc_index]
        return in_array

    def in_notcare_region(self, pointcloud):
        '''
        brief
        -----
        无关区域，限定的无关区域是夹持区域上方，直径为中心距，z<0的区域

        parameter
        -----
        pointcloud: np.ndarray [N, 3]

        return
        ------
        in_array: np.ndarray(bool), [N]
        '''
        ### 根据u计算夹持区域半径、位置
        _, z = self._grasp_r, self._grasp_z
        r = self.finger_center_dis/2
        points_r:np.ndarray = np.linalg.norm(pointcloud[:,0:2], axis = -1) #[N]
        r_in = points_r < r

        z_min = z + self.claw_length
        z_max = 0
        z_in = np.all((pointcloud[:,2] >  z_min,
                        pointcloud[:,2] < z_max))

        in_array = r_in * z_in
        return in_array

    def get_trans_mat(self, trans_parameter):
        trans_mat = np.eye(4,4)
        rot = cv2.Rodrigues(trans_parameter[0:3])[0]
        trans_mat[0:3,0:3] = rot
        trans_mat[0:3,3] = np.array(trans_parameter[3:-1])        
        return trans_mat

    def trans_pointcloud(self, trans_mat, pointcloud):
        N = pointcloud.shape[0]
        pointcloud = np.concatenate((pointcloud, np.ones((N,1))), axis=-1)
        trans_pointcloud = np.dot(pointcloud, trans_mat.T)
        return trans_pointcloud[:,:3]

    def expand_one(self, pointcloud):
        N = pointcloud.shape[0]
        pointcloud = np.concatenate((pointcloud, np.ones((N,1))), axis=-1)
        return    pointcloud     
        
    def evaluate_posture_func(self, target_pointcloud, other_pointcloud, search_zangle = False, 
                              base_rotvec:np.ndarray = np.array([0,0,0]), base_tvec:np.ndarray = np.array([0,0,0]), base_u = 0.0,
                              ifprint:bool = False):
        '''
        brief
        -----
        评估姿态

        parameter
        -----
        trans_mat: 夹持器的变化矩阵的逆（改为让点云变换）
        target_pointcloud: 目标抓取物的点云
        other_pointcloud: 其他物体的点云
        
        base_rotvec = [brx, bry, brz]: trans_parameter应当是长度为6的向量，[rot_theta, rot_rho, δx, δy, δz, δu]\n
        其中 [cos(rot_theta) * rot_rho, sin(rot_theta) * rot_rho, 0] 构成一个旋转向量，rot_theta∈[0,2pi]，通过约束rot_rho的值，形成一个锥形的空间。
        点云先做上述旋转，再做base_rotvec的旋转，实现在指定角度范围内的搜索。
        

        return
        -----
        grasp_score: 抓取分数
        '''
        # 目标物体
        N = target_pointcloud.shape[0]
        if target_pointcloud.shape[-1] == 3:
            target_pointcloud = np.concatenate((target_pointcloud, np.ones((N,1))), axis=-1)
        barycenter = np.mean(target_pointcloud, axis=0, keepdims= True) #重心，放置在最后一行
        target_pointcloud = np.concatenate((target_pointcloud, barycenter), axis = 0)
        N = other_pointcloud.shape[0]
        if other_pointcloud.shape[-1] == 3:
            other_pointcloud = np.concatenate((other_pointcloud, np.ones((N,1))), axis=-1)
        def score_func(trans_parameter):
            '''
            brief
            -----
            打分函数，根据输入参数评估姿态的分数。
            根据是否搜索z有2种模式：
            * mode1: search_zangle == False\n
            trans_parameter应当是长度为6的向量
            * mode2: search_zangle == True\n
            trans_parameter应当是长度为7的向量
            base_rotvec = [brx, bry, brz], base_tvec = [btx, bty, btz]: trans_parameter应当是长度为1的向量，[angle_z]\n
            在变换基础上找合适的z
            
            * 其他输入为非法 TODO：输入合法性检测

            parameter
            -----
            trans_parameter
            '''
            # if search_zangle:
            #     z_rot_posture = Posture(Posture.POSTURE_VEC, 
            #                             np.array([0, 0, trans_parameter[-1]]), np.array([0, 0, 0]))
            #     base_posture = Posture(Posture.POSTURE_VEC, 
            #                             base_rotvec, base_tvec)
            #     trans_mat = base_posture.trans_mat
            #     u = base_u
            # else:
            #     z_rot_posture = Posture(Posture.POSTURE_VEC, 
            #                             np.zeros(3), np.zeros(3))
            #     trans_mat, u = self.parse_trans_parameter(trans_parameter, base_rotvec, base_tvec, base_u)
            
            trans_mat, u = self.parse_trans_parameter(trans_parameter, base_rotvec, base_tvec, base_u)
            self.set_u(u)
            # trans_mat = np.dot(trans_mat, z_rot_posture.trans_mat)
            # rot_vec = tmp_trnsprmt[0:2]
            # rot_vec = np.append(rot_vec, 0)
            # base_trans_mat = self.parse_trans_parameter(rot_vec, tmp_trnsprmt[2:-1]) #基础变换矩阵
            # u = tmp_trnsprmt[-1]
            # self.set_u(u) #设置u

            # # 在新坐标系的z轴上旋转
            # if search_zangle:
            #     rot_vec_new = np.array([0,0,trans_parameter[0]])
            #     new_trans_mat = self.parse_trans_parameter(rot_vec_new) #z旋转矩阵
            #     trans_mat = np.dot(new_trans_mat, base_trans_mat) #在原有变换矩阵基础上额外进行Z的旋转
            # else:
            #     trans_mat = base_trans_mat #基础变换
            # print(trans_mat)
            # 本应是夹持器左乘变换矩阵，改为点云右乘变换矩阵，得到在变换后的夹持器坐标系下的位置
            trans_target_pointcloud = np.linalg.inv(trans_mat).dot(target_pointcloud.T).T
            trans_other_pointcloud  = np.linalg.inv(trans_mat).dot(other_pointcloud.T).T

            grasp_region_array = self.in_grasp_region(trans_target_pointcloud)
            target_interference_region_array = self.in_interference_region(trans_target_pointcloud[:,:3], True)
            other_interference_region_array = self.in_interference_region(trans_other_pointcloud[:,:3], True)

            target_interference_region_array = target_interference_region_array * np.logical_not(grasp_region_array)

            grasp_num = np.sum(grasp_region_array)
            interface_num = np.sum(target_interference_region_array) + np.sum(other_interference_region_array)

            grasp_score = 100 * grasp_num / self._grasp_r**2
            interface_score = - interface_num**2

            # ### 还要考虑重心，重心应该尽量贴近夹爪中心线
            # trans_barycenter = trans_target_pointcloud[-1]
            # center_score = - 0.1 * np.linalg.norm(trans_barycenter[0:2], axis=-1)

            score = grasp_score + interface_score
            if ifprint:
                print(grasp_num, interface_num, grasp_score, interface_score, score)
            return -score
        return score_func

    def get_gripper_o3d_geo(self):
        fingers = [o3d.geometry.TriangleMesh.create_box(width=self.finger_width,
                                                height=self.finger_thickness,
                                                depth=self.finger_length) for i in range(3)]
        finger_grippings = [o3d.geometry.TriangleMesh.create_box(width=self.finger_gripping_width,
                                                height=self.finger_thickness,
                                                depth=self.finger_gripping_length) for i in range(3)]
        gripper_o3d_geos =  fingers + finger_grippings
        
        trans_t_fingers = Posture(tvec=np.array([0, -self.finger_thickness/2, -self.finger_length]))
        trans_t_fingers_grippings = Posture(tvec=np.array([-self.finger_gripping_width, -self.finger_thickness/2, -self.finger_gripping_length]))
        finger_trans_mat = [np.linalg.multi_dot((   Posture(rvec=np.array([0, 0, np.pi/3*2*i])).trans_mat, 
                                                    self.finger_trans_mat,
                                                    trans_t_fingers.trans_mat)) for i in range(3)]
        finger_grippings_trans_mat = [np.linalg.multi_dot((Posture(rvec=np.array([0, 0, np.pi/3*2*i])).trans_mat,                                                     
                                                    self.finger_trans_mat,
                                                    trans_t_fingers_grippings.trans_mat)) for i in range(3)] 
        gripper_o3d_geos_trans_mat = finger_trans_mat + finger_grippings_trans_mat                                            
        for f, mat in zip(gripper_o3d_geos, gripper_o3d_geos_trans_mat):
            f.compute_vertex_normals()    
            f.paint_uniform_color([1, 0, 0])     
            f.transform(mat)       
       
        return gripper_o3d_geos, gripper_o3d_geos_trans_mat

    def render(self, posture_WCS:Posture = None, u = None):
        posture_WCS = self.posture_WCS if posture_WCS is None else posture_WCS
        u = self.u if u is None else u
        self.set_u(u)
        gripper_o3d_geo, gripper_trans_mat = self.get_gripper_o3d_geo()
        for geo, mat in zip(gripper_o3d_geo, gripper_trans_mat):
            # mat = np.dot(self.posture_WCS.trans_mat, mat)
            geo.transform(posture_WCS.trans_mat)    
        gripper_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=40, origin=[0,0,0])
        gripper_frame.transform(posture_WCS.trans_mat)
        return gripper_o3d_geo + [gripper_frame]

if __name__ == "__main__":
    gripper = MyThreeFingerGripper() 
    # trans_mat, u = gripper.parse_trans_parameter([0, InitAngle.CONE_ANGLE, 0, 0, 0, 0.5, 0], np.array([0,0,0]), -np.array([0,0,0]), 0)
    parameter_mat = np.load(r".\graspmodel\ape_candi_grasp_posture.npy")
    for parameter in parameter_mat:
        posture = Posture(rvec=parameter[:3], tvec=parameter[3:6])
        u = parameter[6]
        gripper.set_u(u)
        gripper.show_claw(posture.trans_mat)
        print()