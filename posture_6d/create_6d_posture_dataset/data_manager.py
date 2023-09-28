from . import DatasetFormat, DST, ClusterNotRecommendWarning, FRAMETYPE_DATA
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import open3d as o3d
from typing import Union, Callable, Generator, Any
from warnings import warn


from . import DatasetFormatMode, DatasetFormat, Elements, ViewMeta, JsonDict, FileCluster, JsonIO
from . import RGB_DIR, DEPTH_DIR, TRANS_DIR, ARUCO_FLOOR

class FrameMeta():
    def __init__(self, trans_mat_Cn2C0, rgb = None, depth = None, intr_M = None) -> None:
        self.trans_mat_Cn2C0:np.ndarray = trans_mat_Cn2C0
        self.color:np.ndarray = rgb
        self.depth:np.ndarray = depth
        self.intr_M:np.ndarray = intr_M

class CommonData(DatasetFormat[DST]):
    def __init__(self, directory, clear_incomplete=False, init_mode=DatasetFormatMode.NORMAL) -> None:
        super().__init__(directory, clear_incomplete, init_mode)

    def _init_clusters(self):
        super()._init_clusters()
        self.std_meshes = EnumElements(self, "../std_meshes", register=False,
                                read_func=o3d.io.read_triangle_mesh,
                                write_func=o3d.io.write_triangle_mesh,
                                suffix='.ply')
        self.std_meshes_image = EnumElements(self, "../std_meshes", register=False,
                                read_func=cv2.imread,
                                write_func=cv2.imwrite,
                                suffix='.jpg')
        self.aruco_floor_json       = FileCluster(self, "../", True, name="aruco_floor_json",
                                                  singlefile_list = [FileCluster.SingleFile(ARUCO_FLOOR + ".json", 
                                                                            JsonIO.load_json, 
                                                                            JsonIO.dump_json)])
        self.aruco_floor_png        = FileCluster(self, "../", True, name="aruco_floor_png",
                                                  singlefile_list = [FileCluster.SingleFile(ARUCO_FLOOR + ".png", 
                                                                            cv2.imread, 
                                                                            cv2.imwrite),
                                                  FileCluster.SingleFile(ARUCO_FLOOR + "_long_side.txt",
                                                                            np.loadtxt,
                                                                            np.savetxt)])

        self.std_meshes_dir = self.std_meshes.directory
        self.std_meshes_names:list = []
        for i in range(len(self.std_meshes)):
            self.std_meshes_names.append(self.std_meshes.auto_path(i, return_app=True)[-1])

        self.imu_calibration        = FileCluster(self, "../", True, name="imu_calibration",
                                                  singlefile_list = [FileCluster.SingleFile("imu_calibration.json", 
                                                                            JsonIO.load_json, 
                                                                            JsonIO.dump_json)])
        
        self.barycenter_dict        = JsonDict(self, "../std_meshes/barycenter.json", False)
        self.barycenter_dict.save_mode = JsonDict.SAVE_IMMIDIATELY


class EnumElements(Elements[CommonData, Any]):
    # def __init__(self, dataset_node: "ModelManager", sub_dir, register=True, read_func = ..., write_func = ..., suffix: str = '.txt', filllen=6, fillchar='0') -> None:
    #     super().__init__(dataset_node, sub_dir, register, read_func, write_func, suffix, filllen, fillchar)
    #     self.dataset_node:ModelManager = dataset_node    

    @property
    def enums(self):
        return self.dataset_node.std_meshes_names

    def format_path(self, enum:Union[str, int], subdir="", appname="", **kw):
        if isinstance(enum, (np.intc, np.integer)):
            enum = int(enum)
        if not appname:
            if isinstance(enum, int):
                appname = self.dulmap_id_name(enum)
                data_i = enum
            elif isinstance(enum, str):
                # get key by value
                appname = enum
                data_i = self.dulmap_id_name(enum)
        else:
            assert isinstance(enum, int), "enum must be int when appname is not empty"
            data_i = enum
        return super().format_path(data_i, subdir, appname, **kw)
    
    def dulmap_id_name(self, enum:Union[str, int]):
        if isinstance(enum, int):
            return self.enums[enum]
        elif isinstance(enum, str):
            return self.enums.index(enum)

class DataRecorder(CommonData[FrameMeta]):    
    @property
    def category_idx_range(self):
        return self._category_idx_range
            
    @property
    def current_category_index(self):
        return self.__categroy_index
    
    @property
    def current_category_name(self):
        return self.category_names[self.__categroy_index]
    
    @property
    def current_categroy_num(self):
        return len(self.category_idx_range[self.current_category_name])

    @property
    def is_all_recorded(self):
        return self.__categroy_index == len(self.category_names)

    def _update_dataset(self, data_i = None):
        super()._update_dataset(data_i)
        if data_i is None:
            self.category_idx_range.clear()
            for name in self.category_names:
                self.category_idx_range.setdefault(name, [])
            for data_i in self.rgb_elements.keys():
                cate_i = self.get_category_idx(data_i)
                self.category_idx_range.setdefault(self.category_names[cate_i], []).append(data_i)
        else:
            self.category_idx_range[self.current_category_name].append(data_i)

    def _init_clusters(self):
        super()._init_clusters()
        self.close_all()
        self.rgb_elements   = ElementsWithCategory(self,      RGB_DIR,
                                       read_func=cv2.imread,                                    
                                       write_func=cv2.imwrite, 
                                       suffix='.png')
        self.depth_elements = ElementsWithCategory(self,      DEPTH_DIR,    
                                       read_func=lambda x:cv2.imread(x, cv2.IMREAD_ANYDEPTH),   
                                       write_func=cv2.imwrite, 
                                       suffix='.png')
        self.trans_elements = ElementsWithCategory(self,    TRANS_DIR,
                                       read_func=np.load,
                                       write_func=np.save,
                                       suffix='.npy')
        
        self.intr_0_file = FileCluster(self, "./", True, name="intr_0_file",
                                    singlefile_list = [FileCluster.SingleFile("intrinsics_0.json",
                                                            JsonIO.load_json,
                                                            JsonIO.dump_json)])
        self.intr_1_file = FileCluster(self, "./", True, name="intr_1_file",
                                    singlefile_list = [FileCluster.SingleFile("intrinsics_1.json",
                                                            JsonIO.load_json,
                                                            JsonIO.dump_json)])

        self.close_all(False)
        self.set_all_readonly(False)

        self.category_names = self.std_meshes_names.copy()
        self.category_names.insert(0, "global_base_frames") # 在标准模型列表的第一位插入"global_base_frames"
        self.category_names.insert(1, "local_base_frames") # 在标准模型列表的第二位插入"local_base_frames"
        self.category_names.append("dataset_frames") # 在标准模型列表的最后一位插入"dataset_frames"

        self.__categroy_index = 0
        self.AddNum = 0 # 当前标准模型已采集的增量帧数
        self.skip_segs = []

        self._category_idx_range = {}

    def inc_idx(self):
        self.__categroy_index += 1
        self.__categroy_index = min(self.__categroy_index, len(self.category_names))
        self.AddNum = 0

    def dec_idx(self):
        self.__categroy_index -= 1
        self.__categroy_index = max(self.__categroy_index, 0)
        self.AddNum = 0

    def clear_skip_segs(self):
        self.skip_segs.clear()

    def add_skip_seg(self, seg):
        assert isinstance(seg, int), "seg must be int"
        if seg > 0 and seg <= len(self.category_names):
            self.skip_segs.append(seg)
        if seg < 0 and seg > -len(self.category_names):
            self.skip_segs.append(len(self.category_names) + seg)

    def skip_to_seg(self):
        '''
        跳过，直到分段点
        '''
        try:
            skip_to = self.skip_segs.pop(0)
            skip_to = min(skip_to, len(self.category_names))
            skip_to = max(skip_to, self.__categroy_index)
        except:
            skip_to = len(self.category_names)
        while self.__categroy_index < skip_to:
            self.inc_idx()
            
    def get_category_idx(self, data_i):
        subdir = self.rgb_elements.auto_path(data_i, return_app=True)[-2]
        return self.category_names.index(subdir)

    def read_one(self, data_i) -> FrameMeta:
        super().read_one(data_i)

        rgb = self.rgb_elements.read(data_i)
        depth = self.depth_elements.read(data_i)
        trans = self.trans_elements.read(data_i)

        if data_i in self.category_idx_range[FRAMETYPE_DATA]:
            intr_M = self.intr_1_file.read(0)
        else:
            intr_M = self.intr_0_file.read(0)

        return FrameMeta(trans_mat_Cn2C0=trans, rgb=rgb, depth=depth, intr_M=intr_M)

    def read_from_disk(self) -> Generator[FrameMeta, Any, None]:
        return super().read_from_disk()
    
    def read_in_category_range(self, start, end):
        valid_category = list(range(len(self.category_names)))[start:end]
        for i in range(self.data_num):
            category_idx:int = self.get_category_idx(i)
            if category_idx in valid_category:
                framemeta:FrameMeta = self.read_one(i)
                yield category_idx, framemeta

    def _write_elements(self, data_i: int, framemeta: FrameMeta):
        subdir = self.category_names[self.__categroy_index]
        self.rgb_elements.write(data_i,    framemeta.color,   subdir=subdir)
        self.depth_elements.write(data_i,  framemeta.depth, subdir=subdir)
        self.trans_elements.write(data_i,  framemeta.trans_mat_Cn2C0, subdir=subdir)
        ### ADD    
        self.AddNum += 1

    def save_frames(self, c, d, t):
        framemeta = FrameMeta(t, c, d)
        data_i = self.data_i_upper
        self.write_one(data_i, framemeta)

    def remove(self, remove_list:list, change_file = True):
        pass

    def insert(self, insert_list:list, change_file = True):
        pass

    def rename_all(self, exchange_pair = []):
        pass

    def make_directories(self):
        pass
        # if os.path.exists(self.rgb_dir):
        #     return
        # else:
        #     for d in [self.rgb_dir, self.depth_dir, self.trans_dir]:
        #         try:
        #             shutil.rmtree(d)
        #         except:
        #             pass
        #         os.makedirs(d)
        #     with open(self.directory+'category_idx_range.json', 'w') as fp:
        #         json.dump(self.model_index_dict, fp)

class ElementsWithCategory(Elements[DataRecorder, np.ndarray]):
    @property
    def current_category_range(self):
        return self.dataset_node.category_idx_range[self.dataset_node.current_category_name]

    def in_current_category(self):
        try:
            _range = self.current_category_range
        except:
            _range = [] 
        for data_i in _range:
            yield self.read(data_i)

class ModelManager(CommonData):

    ARUCO_USED_TIMES = "aruco_used_times"
    ARUCO_CENTERS = "aruco_centers"
    PLANE_EQUATION = "plane_equation"
    TRANS_MAT_C0_2_SCS = "trans_mat_C0_2_SCS"
    VOR_POLYS_COORD = "vor_polys_coord"
    FLOOR_COLOR = "floor_color"

    def _init_clusters(self):
        super()._init_clusters()
        self.registerd_pcd = EnumElements(self, "registerd_pcd", False,
                                      read_func=o3d.io.read_point_cloud,
                                      write_func=o3d.io.write_point_cloud,
                                      suffix='.ply')
        self.voronoi_segpcd = EnumElements(self, "voronoi_segpcd", False,
                                        read_func=o3d.io.read_point_cloud,
                                        write_func=o3d.io.write_point_cloud,
                                        suffix='.ply')
        self.extracted_mesh = EnumElements(self, "extracted_mesh", False,
                                        read_func=o3d.io.read_triangle_mesh,
                                        write_func=o3d.io.write_triangle_mesh,
                                        suffix='.ply')
        self.icp_trans      = EnumElements(self, "icp_trans", False,
                                        read_func=np.load,
                                        write_func=np.save,
                                        suffix='.npy')
        self.icp_std_mesh = EnumElements(self, "icp_std_mesh", False,
                                        read_func = o3d.io.read_triangle_mesh,
                                        write_func= o3d.io.write_triangle_mesh,
                                        suffix='.ply')
        self.icp_unf_pcd = EnumElements(self, "icp_unf_pcd", False,
                                        read_func=o3d.io.read_point_cloud,
                                        write_func=o3d.io.write_point_cloud,
                                        suffix='.ply')        
        
        self.merged_regist_pcd_file = FileCluster(self, "", False, name="merged_regist_pcd_file",
                                                    singlefile_list = [FileCluster.SingleFile("merged.ply",
                                                                                   read_func = o3d.io.read_point_cloud,
                                                                                   write_func= o3d.io.write_point_cloud)])

        self.process_data = ProcessData(self, register = False)
    
    def read_one(self, data_i, subdir="", appname="") -> ViewMeta:
        warn("the file might be too large to read, only the paths are returned", ClusterNotRecommendWarning)
        return self.get_element_paths_of_one(data_i, subdir, appname)
    
    def write_one(self, data_i, data: Any, *arg, **kwargs):
        warn("can not write", ClusterNotRecommendWarning)

class ProcessData(JsonDict[ModelManager, dict[str, np.ndarray]]):

    ARUCO_USED_TIMES = "aruco_used_times"
    ARUCO_CENTERS = "aruco_centers"
    PLANE_EQUATION = "plane_equation"
    TRANS_MAT_C0_2_SCS = "trans_mat_C0_2_SCS"
    VOR_POLYS_COORD = "vor_polys_coord"
    FLOOR_COLOR = "floor_color"

    def __init__(self, dataset_node: DatasetFormat, sub_dir: str = "process_data.json", register=False) -> None:
        super().__init__(dataset_node, sub_dir, register)

    def check_key(self, key):
        super().check_key(key)
        if key not in vars(ProcessData).values() or callable(key):
            return False
        else:
            return True
        
    # def __init__(self, directory) -> None:
    #     self.directory = directory
    #     self.process_file = os.path.join(self.directory, "pcd_creator_process_data.json")
    #     self.load_process_data()

    # def load_process_data(self):
    #     self.process_data = JsonDict()
    #     if not os.path.exists(self.process_file):
    #         self.process_data = {}
    #         JsonIO.dump_json(self.process_file, self.process_data)
    #         return
    #     self.process_data = JsonIO.load_json(self.process_file)

    # def dump_process_data(self):
    #     JsonIO.dump_json(self.process_file, self.process_data)
