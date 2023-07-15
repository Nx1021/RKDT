import os
import numpy as np
import open3d as o3d
from .posture import Posture
from .utils import JsonIO, modify_class_id, get_meta_dict


class MeshMeta:
    def __init__(self,
                 mesh, 
                 bbox_3d: np.ndarray = None, 
                 symmetries:dict = None, 
                 diameter:float = None,  
                 ldmk_3d: np.ndarray = None,
                 name = "",
                 class_id = -1) -> None:
        self.mesh = mesh
        self.bbox_3d: np.ndarray    = bbox_3d
        self.symmetries:dict        = symmetries #"symmetries_continuous": "symmetries_discrete": 
        self.diameter: float        = diameter
        self.ldmk_3d: np.ndarray    = ldmk_3d

        self.name = name
        self.class_id = class_id

    @property
    def pcd(self):
        return np.asarray(self.mesh.vertices)
    
    @property
    def normals(self):
        return np.asarray(self.mesh.vertex_normals)

    @property
    def tris(self):
        return np.asarray(self.mesh.triangles)

    def transform(self, posture:Posture, copy = True):
        new_mesh = o3d.geometry.TriangleMesh(self.mesh)
        new_mesh = new_mesh.transform(posture.trans_mat)

        new_bbox = posture * self.bbox_3d
        new_ldmk = posture * self.ldmk_3d

        if copy:
            return MeshMeta(new_mesh, new_bbox, self.symmetries, self.diameter, new_ldmk, self.name, self.class_id)
        else:
            self.mesh = new_mesh
            self.bbox_3d = new_bbox
            self.ldmk_3d = new_ldmk
            return self


class MeshManager:
    _instance = None

    def __new__(cls, *arg, **kw):
        if not cls._instance:
            cls._instance = super(MeshManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, root, model_names: dict[int, str], load_all = False, modify_class_id_pairs:list[tuple[int]]=[]) -> None:
        self.model_names: dict[int, str] = model_names
        self.model_dirs : dict[int, str]      = {}
        for id_, name in self.model_names.items():
            # 模型名称
            dir_ = os.path.join(root, name)
            self.model_dirs.update({id_: dir_}) 
            name = os.path.splitext((os.path.split(name)[-1]))[0]
            self.model_names[id_] = name
        self.model_meshes       = {}
        self.model_bbox_3d          = {}
        self.model_symmetries = {} #"symmetries_continuous": "symmetries_discrete": 
        self.model_diameter = {}
        self.model_ldmk_3d = {}
        self.landmark_info_path = os.path.join(root, "landmarks.json")
        self.models_info_path = os.path.join(root, "models_info.json")
        self.load_landmarks(self.landmark_info_path)
        self.load_models_info(self.models_info_path)
        if load_all or len(modify_class_id_pairs)>0:
            for key in self.model_dirs:
                if key not in self.model_meshes:
                    self.load_model(key)
        if len(modify_class_id_pairs)>0:
            self.modify_class_id(modify_class_id_pairs)

    def modify_class_id(self, modify_class_id_pairs):
        orig_dict_list = get_meta_dict(self)
        modify_class_id(orig_dict_list, modify_class_id_pairs)

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
        mesh = o3d.io.read_triangle_mesh(self.model_dirs[model_id])
        mesh.normalize_normals()
  
        self.model_meshes.update({model_id: mesh})   

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
            self.model_diameter:dict[int,float] = {} ### 模型直径
            self.model_bbox_3d:dict[int,np.ndarray] = {} ### 模型包围盒
            self.model_symmetries:dict[int, dict] = {}
            # with open(models_info_path, 'r') as MI:
            #     info = json.load(MI)
            info = JsonIO.load_json(models_info_path)
            for k,v in info.items():
                self.model_diameter.update({k: v["diameter"]})
                min_x =  info[k]["min_x"]
                min_y =  info[k]["min_y"]
                min_z =  info[k]["min_z"]
                size_x = info[k]["size_x"]
                size_y = info[k]["size_y"]
                size_z = info[k]["size_z"]
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
                    if symm in info[k]:
                        self.model_symmetries.update({k: info[k][symm]})
            self.models_info_path = models_info_path
        else:
            print("Warning: models_info_path doesn't exist")

    def load_landmarks(self, landmark_info_path: str):
        recalc = True
        if os.path.exists(landmark_info_path):
            self.landmark_info_path = landmark_info_path
            self.model_ldmk_3d:dict[int,np.ndarray] = JsonIO.load_json(landmark_info_path)
            if set(self.model_ldmk_3d.keys()) == set(self.model_dirs.keys()):
                recalc = False
        if recalc:
            for class_id in self.model_dirs:
                points = self.get_model_pcd(class_id)
                centroids, farthest_idx_array = self.farthest_point_sample(points, 24)
                self.model_ldmk_3d.update({class_id: centroids})
            JsonIO.dump_json(landmark_info_path, self.model_ldmk_3d)
            print("Warning: landmark_info_path doesn't exist, calculated")

    def get_bbox_3d(self, class_id:int):
        bbox_3d = self.model_bbox_3d[class_id].copy()
        return bbox_3d

    def get_ldmk_3d(self, class_id:int):
        ldmk_3d = self.model_ldmk_3d[class_id].copy()
        return ldmk_3d

    def get_model_name(self, class_id:int) -> str:
        if class_id not in self.model_names:
            self.load_model(class_id)
        name = self.model_names[class_id]
        return name

    def get_model_pcd(self, class_id:int) -> np.ndarray:
        if class_id not in self.model_meshes:
            self.load_model(class_id)
        pcd = np.asarray(self.model_meshes[class_id].vertices)
        return pcd

    def get_model_normal(self, class_id:int) -> np.ndarray:
        if class_id not in self.model_meshes:
            self.load_model(class_id)
        normal = np.asarray(self.model_meshes[class_id].vertex_normals)
        return normal

    def get_model_mesh(self, class_id:int) -> np.ndarray:
        if class_id not in self.model_meshes:
            self.load_model(class_id)
        mesh = o3d.geometry.TriangleMesh(self.model_meshes[class_id])
        return mesh

    def get_model_diameter(self, class_id:int):
        diameter = self.model_diameter[class_id]
        return diameter

    def get_model_symmetries(self, class_id:int):
        if class_id not in self.model_symmetries:
            return None
        else:
            return self.model_symmetries[class_id].copy()
    
    def export_meta(self, class_id:int):
        mesh                = self.get_model_mesh(class_id)
        bbox_3d     = self.get_bbox_3d(class_id)
        symmetries  = self.get_model_symmetries(class_id) #"symmetries_continuous": "symmetries_discrete": 
        diameter: float    = self.get_model_diameter(class_id)
        ldmk_3d     = self.get_ldmk_3d(class_id)

        name = self.get_model_name(class_id)

        return MeshMeta(mesh, bbox_3d, symmetries, diameter, ldmk_3d, name = name, class_id = class_id)
    
    def get_meta_dict(self):
        meta_dict = {}
        for key in self.model_dirs:
            meta = self.export_meta(key)
            meta_dict[key] = meta
        return meta_dict
