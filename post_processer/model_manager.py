import os
import json
import numpy as np
import open3d as o3d
from utils.yaml import yaml_load


class ModelManager:
    '''
    单例类
    '''
    _instance = None

    def __new__(cls, landmark_info_path, models_info_path, model_dirs: dict[int, str]):
        if not cls._instance:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.model_dirs = model_dirs
            cls._instance.model_pointcloud = {}
            cls._instance.model_bbox_3d = {}
            cls._instance.model_symmetries = {} #"symmetries_continuous": "symmetries_discrete": 
            cls._instance.model_diameter = {}
            cls._instance.model_ldmk_3d = {}
            cls._instance.landmark_info_path = landmark_info_path
            cls._instance.models_info_path = models_info_path
            cls._instance.load_landmarks(landmark_info_path)
            cls._instance.load_models_info(models_info_path)
        return cls._instance

    @staticmethod
    def farthest_point_sample(point_cloud, npoint): 
        """
        Input:
            point_cloud: pointcloud data, [N, 3]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [B, npoint]
        """

        N = point_cloud.shape[0]
        centroids = np.zeros((npoint, 3))    # 采样点矩阵
        farthest_idx_array = np.zeros(npoint)
        distance = np.ones((N)) * 1e10    # 采样点到所有点距离（npoint, N）    
        
        #计算重心
        center = np.mean(point_cloud, 0) # [3]
        # center = np.array([0,0,0])
        # 计算距离重心最远的点
        dist = np.sum((point_cloud - center) ** 2, -1)
        farthest_idx = np.argmax(dist)                                     #将距离重心最远的点作为第一个点，这里跟torch.max不一样
        for i in range(npoint):
            # print("-------------------------------------------------------")
            # print("The %d farthest point %s " % (i, farthest_idx))
            centroid = point_cloud[farthest_idx, :]             # 取出这个最远点的xyz坐标
            centroids[i, :] = centroid                          # 更新第i个最远点
            farthest_idx_array[i] = farthest_idx
            dist = np.sum((point_cloud - centroid) ** 2, -1)                     # 计算点集中的所有点到这个最远点的欧式距离，-1消掉了xyz那个维度
            # print("dist    : ", dist)
            mask = dist < distance
            # print("mask %i : %s" % (i,mask))
            distance[mask] = dist[mask]                                     # 更新distance，记录样本中每个点距离所有已出现的采样点（已采样集合中的点）的最小距离
            # print("distance: ", distance)

            farthest_idx = np.argmax(distance)                           # 返回最远点索引

        return centroids, farthest_idx_array

    def load_model(self, model_id:int):
        # 模型路径下的名称目录
        pcd = o3d.io.read_point_cloud(self.model_dirs[model_id])
        # 获取点云的顶点位置
        vertices = np.asarray(pcd.points)
        if vertices.max() - vertices.min() > 10:
            vertices = vertices / 1000 # 转换单位为m

        self.model_pointcloud.update({model_id: vertices})    

    def load_models_info(self, models_info_path: str):
        '''
          1_______7
         /|      /|         Z
        3_______5 |         |__Y 
        | 0_____|_6        /
        |/      |/        X
        2_______4        
        '''
        if os.path.exists(models_info_path):
            try:
                self.model_diameter:dict[int,float] = {} ### 模型直径
                self.model_bbox_3d:dict[int,np.ndarray] = {} ### 模型包围盒
                self.model_symmetries:dict[int, dict] = {}
                with open(models_info_path, 'r') as MI:
                    info = json.load(MI)
                    for k,v in info.items():
                        k = int(k)
                        self.model_diameter.update({k: v["diameter"]})
                        min_x =  info[str(k)]["min_x"]
                        min_y =  info[str(k)]["min_y"]
                        min_z =  info[str(k)]["min_z"]
                        size_x = info[str(k)]["size_x"]
                        size_y = info[str(k)]["size_y"]
                        size_z = info[str(k)]["size_z"]
                        # 计算顶点坐标并以ndarray返回
                        max_x = min_x + size_x
                        max_y = min_y + size_y
                        max_z = min_z + size_z
                        # 计算顶点坐标并以ndarray返回
                        #   0,  1,  2,  3,  4,  5,  6,  7
                        x =np.array([-1,-1, 1, 1, 1, 1,-1,-1]) * max_x
                        y =np.array([-1,-1,-1,-1, 1, 1, 1, 1]) * max_y
                        z =np.array([-1, 1,-1, 1,-1, 1,-1, 1]) * max_z
                        vertex = np.vstack((x, y, z)).T
                        self.model_bbox_3d.update({k: vertex}) #[8, 3]
                        # 对称属性
                        for symm in ["symmetries_continuous", "symmetries_discrete"]:
                            if symm in info[str(k)]:
                                self.model_symmetries.update({k: info[str(k)][symm]})
            except:
                print("Error: PnPSolver cannot set_models_info_path")
            self.models_info_path = models_info_path
        else:
            print("Warning: models_info_path doesn't exist")

    def load_landmarks(self, landmark_info_path: str):
        if os.path.exists(landmark_info_path):
            try:
                self.model_ldmk_3d:dict[int,np.ndarray]  = {} ### 模型关键点
                with open(landmark_info_path, 'r') as ldmkf:
                    ldmks_info = json.load(ldmkf)
                    for k,v in ldmks_info.items():
                        v = np.reshape(np.array(v), (-1, 3))
                        self.model_ldmk_3d.update({int(k): v})
            except:
                print("Error: PnPSolver cannot set_landmark_info_path")
            self.landmark_info_path = landmark_info_path
        else:
            print("Warning: landmark_info_path doesn't exist")

    def get_bbox_3d(self, class_id:int):
        bbox_3d = self.model_bbox_3d[class_id].copy()
        return bbox_3d

    def get_ldmk_3d(self, class_id:int):
        ldmk_3d = self.model_ldmk_3d[class_id].copy()
        return ldmk_3d

    def get_model_pcd(self, class_id:int) -> np.ndarray:
        if class_id not in self.model_pointcloud:
            self.load_model(class_id)
        pcd = self.model_pointcloud[class_id].copy()
        return pcd
    
    def get_model_diameter(self, class_id:int):
        diameter = self.model_diameter[class_id]
        return diameter

    def get_model_symmetries(self, class_id:int):
        if class_id not in self.model_symmetries:
            return None
        else:
            return self.model_symmetries[class_id].copy()
    
def create_model_manager(cfg) -> ModelManager:
    cfg_paras = yaml_load(cfg)
    model_manager = ModelManager(cfg_paras["landmarks"],
                                 cfg_paras["models_info"],
                                 cfg_paras["pcd_models"])
    return model_manager

