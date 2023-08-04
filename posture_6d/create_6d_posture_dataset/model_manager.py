import os
from typing import Callable
from posture_6d.dataset_format import DatasetFormat
from posture_6d.viewmeta import ViewMeta
from . import dataset_format, Elements, DatasetFormat, JsonDict
import numpy as np
import open3d as o3d
from typing import Union
from warnings import warn


class EnumElements(dataset_format.Elements):
    def __init__(self, format_obj: "ModelManager", sub_dir, register=True, read_func = ..., write_func = ..., suffix: str = '.txt', filllen=6, fillchar='0') -> None:
        super().__init__(format_obj, sub_dir, register, read_func, write_func, suffix, filllen, fillchar)
        self.format_obj:ModelManager = format_obj    

    @property
    def enums(self):
        return self.format_obj.model_names

    def read(self, data_i, appdir="", appname="", **kw):
        ''''
        data_i: int or str
        appdir: str
        appname: str(unused in this function)
        '''
        return super().read(data_i, appdir, appname, **kw)
    
    def write(self, data_i, element, appdir="", appname="", **kw):
        '''
        data_i: int or str
        element: Object
        appdir: str
        appname: str(unused in this function)
        '''
        return super().write(data_i, element, appdir, appname, **kw)

    def format_path(self, enum:Union[str, int], element, appdir="", appname="", **kw):
        if isinstance(enum, int):
            appname = self.dulmap_id_name(enum)
            data_i = enum
        elif isinstance(enum, str):
            # get key by value
            appname = enum
            data_i = self.dulmap_id_name(enum)
        return super().format_path(data_i, element, appdir, appname, **kw)
    
    def dulmap_id_name(self, enum:Union[str, int]):
        if isinstance(enum, int):
            return self.enums[enum]
        elif isinstance(enum, str):
            return self.enums.index(enum)

class ModelManager(DatasetFormat):
    def _init_clusters(self):
        self.std_meshes = EnumElements(self, "../std_meshes", register=False,
                                    read_func=o3d.io.read_triangle_mesh,
                                    write_func=o3d.io.write_triangle_mesh,
                                    suffix='.ply')
        self.registerd_pcd = EnumElements(self, "registerd_pcd",
                                      read_func=o3d.io.read_point_cloud,
                                      write_func=o3d.io.write_point_cloud,
                                      suffix='.ply')
        self.voronoi_segpcd = EnumElements(self, "voronoi_segpcd",
                                        read_func=o3d.io.read_point_cloud,
                                        write_func=o3d.io.write_point_cloud,
                                        suffix='.ply')
        self.extracted_mesh = EnumElements(self, "extracted_mesh",
                                        read_func=o3d.io.read_triangle_mesh,
                                        write_func=o3d.io.write_triangle_mesh,
                                        suffix='.ply')
        self.icp_trans = EnumElements(self, "icp_trans", 
                                  read_func=np.load,
                                  write_func=np.save,
                                  suffix='.npy')
        self.icp_std_mesh = EnumElements(self, "icp_std_mesh",
                                     read_func = o3d.io.read_triangle_mesh,
                                     write_func= o3d.io.write_triangle_mesh,
                                     suffix='.ply')
        self.icp_unf_pcd = EnumElements(self, "icp_unf_pcd",
                                    read_func=o3d.io.read_point_cloud,
                                    write_func=o3d.io.write_point_cloud,
                                    suffix='.ply')        
        
        self.merged_regist_pcd_file = os.path.join(self.directory, "merged.ply")
        self.model_names = JsonDict(self, "model_names.json")
    
    def read_one(self, data_i, appdir="", appname="") -> ViewMeta:
        warn("the file might be too large to read, only the paths are returned")
        return self.get_element_paths_of_one(data_i, appdir, appname)
    
    def write_element(self, obj, data_i: int, appdir="", appname=""):
        warn("please write by the elements")
        return None
    
    def write_merged_registered_pcd(self, pcd):
        o3d.io.write_point_cloud(self.merged_regist_pcd_file, pcd)

    def read_merged_registered_pcd(self, pcd):
        return o3d.io.read_point_cloud(self.merged_regist_pcd_file)
    
