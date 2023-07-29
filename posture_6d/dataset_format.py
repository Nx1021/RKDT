# from compute_gt_poses import GtPostureComputer

# from toolfunc import *
import matplotlib.pyplot as plt
import numpy as np
from open3d import geometry, utility, io
import os
import shutil
import pickle
import cv2
import time
import warnings

from abc import ABC, abstractmethod
from typing import Union, Callable

from .viewmeta import ViewMeta, serialize_image_container, deserialize_image_container
from .posture import Posture
from .mesh_manager import MeshMeta
from .utils import JsonIO, JSONDecodeError


class _DataCluster():
    def __init__(self, format_obj:"DatasetFormat", sub_dir:str, register = True, *args, **kwargs) -> None:
        self.format_obj = format_obj
        self.sub_dir = sub_dir
        self.register = register  
        self.incomplete = self.format_obj.incomplete     
        self._closed = True
        self._read_only = True
        self._updated = False
        self.directory = os.path.join(format_obj.directory, self.sub_dir)         
        self._init_attr(*args, **kwargs)
        self.open()

    @property
    def cover_write(self):
        return self.format_obj.cover_write

    def _init_attr(self, *args, **kwargs):
        pass

    def open(self):
        if self.incomplete:
            self.clear()
            self.incomplete = False
        self._closed = False

    def close(self):
        self._updated = False
        self._closed = True  

    def is_close(self):
        return self._closed

    def set_read_only(self, read_only = True):
        self._read_only = read_only

    def is_read_only(self):
        return self._read_only

    @staticmethod
    def cluster_closed_decorator(is_write_func):
        def write_func_wrapper(func):
            def wrapper(self:"_DataCluster", *args, **kwargs):
                if self._closed:
                    warnings.warn(f"{self.__class__.__name__} is closed, any io operation will not be executed.", Warning)
                    return None
                elif self._read_only and is_write_func:
                    warnings.warn(f"{self.__class__.__name__} is read only, any write operation will not be executed.", Warning)
                    return None
                else:
                    rlt = func(self, *args, **kwargs)
                    self._updated = is_write_func
                    return rlt
            return wrapper
        return write_func_wrapper
    
    def clear(self):
        pass

class BaseJsonDict(dict, _DataCluster):
    '''
    dict for base json
    ----
    Returns None if accessing an key that does not exist

    attr
    ----
    * self.format_obj: DatasetFormat
    * self.closed: bool, Control the shielding of reading and writing, 
        if it is true, the instance will not write, and the read will get None
    '''
    def __init__(self, format_obj: "DatasetFormat", sub_dir: str, register = True) -> None:
        _DataCluster.__init__(self, format_obj, sub_dir, register)

    def _init_attr(self, *args, **kwargs):
        _DataCluster._init_attr(self, *args, **kwargs)
        if os.path.exists(self.directory):
            try:
                value = JsonIO.load_json(self.directory)
            except JSONDecodeError:
                value = {}
        else:
            value = {}
        dict.__init__(self, value)
        if self.register:
            if self.directory not in self.format_obj.base_json:
                self.format_obj.base_json.update({self.directory: self})
            else:
                self.format_obj.base_json[self.directory].update(self) 

    @_DataCluster.cluster_closed_decorator(False)
    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            return None
        
    @_DataCluster.cluster_closed_decorator(True)
    def __setitem__(self, key, value):
        if (self.cover_write or key not in self):
            super().__setitem__(key, value)

    @_DataCluster.cluster_closed_decorator(True)
    def update(self, *arg, **kw):
        super().update(*arg, **kw)

    def clear(self) -> None:
        with open(self.directory, 'w'):
            pass
        return super().clear()

class Elements(_DataCluster):
    '''
    elements manager
    ----
    Returns None if accessing an data_id that does not exist
    will not write if the element is None

    attr
    ----
    * self.format_obj: DatasetFormat
    * self.dir_: str, the directory to store datas
    * self.readfunc:  Callable, how to read one element from disk
    * self.writefunc: Callable, how to write one element to disk
    * filllen: int, to generate the store path
    * fillchar: str, to generate the store path
    * self.closed: bool, Control the shielding of reading and writing, 
        if it is true, the instance will not write, and the read will get None
    '''
    def __init__(self, 
                format_obj:"DatasetFormat",
                sub_dir,
                register = True,
                read_func:Callable = lambda x: None, 
                write_func:Callable = lambda x,y: None, 
                suffix:str = '.txt', 
                filllen = 6, 
                fillchar = '0') -> None:
        super().__init__(format_obj, sub_dir, register, read_func, write_func, suffix, filllen, fillchar)

    def _init_attr(self, read_func, write_func, suffix, filllen, fillchar, *args, **kwargs):
        super()._init_attr(*args, **kwargs)
        self.filllen    = filllen
        self.fillchar   = fillchar
        self.suffix     = suffix
        if not self.suffix.startswith('.'):
            self.suffix = '.' + self.suffix
        self.read_func  = read_func
        self.write_func = write_func
        os.makedirs(self.directory, exist_ok=True)
        self._data_i_dir_map = {}
        self._index = 0
        self._max_idx = 0

        if self.register:
            self.format_obj.elements[self.directory] = self # register to format_obj

    def __len__(self):
        '''
        Count the total number of files in the directory
        '''
        count = 0
        for root, dirs, files in os.walk(self.directory):
            count += len(files)
        return count

    def _init_data_i_dir_map(self):
        self._data_i_dir_map = {}
        for root, dirs, files in os.walk(self.directory):
            for f in files:
                name = os.path.splitext(f)[0][:self.filllen]
                self._data_i_dir_map[int(name)] = os.path.relpath(root, self.directory)
                self._max_idx = max(self._max_idx, int(name))        

    @_DataCluster.cluster_closed_decorator(False)
    def __iter__(self):
        if len(self._data_i_dir_map) != len(self):
            self._init_data_i_dir_map()
        self._index = 0
        return self

    def __next__(self):
        if self._index < self._max_idx:
            value = None
            while value is None:
                idx = self._index
                value = self.read(self._index, self._data_i_dir_map[self._index])
                self._index += 1
            return idx, value
        else:
            raise StopIteration

    @_DataCluster.cluster_closed_decorator(False)
    def read(self, data_i, appdir = "", appname = ""):
        path = self.path_format(data_i, appdir=appdir, appname=appname)
        if not os.path.exists(path):
            if self._updated or len(self._data_i_dir_map) == 0:
                self._init_data_i_dir_map()
            path = self.path_format(data_i, appdir=self._data_i_dir_map[data_i], appname=appname)
        if os.path.exists(path):
            return self.read_func(path)
        else:
            return None

    @_DataCluster.cluster_closed_decorator(True)
    def write(self, data_i, element, appdir = "", appname = ""):
        path = self.path_format(data_i, appdir=appdir, appname=appname)
        dir_ = os.path.split(path)[0]
        os.makedirs(dir_, exist_ok=True)
        if element is not None:
            self.write_func(path, element)

    def path_format(self, data_i, appdir = "", appname = ""):
        return os.path.join(self.directory, appdir, "{}{}{}".format(str(data_i).rjust(6, "0"), appname, self.suffix))

    @_DataCluster.cluster_closed_decorator(True)
    def clear(self, ignore_warning = False):
        '''
        brief
        -----
        clear all data, defalut to ask before executing
        '''
        if not ignore_warning:
            y = input("All files in {} will be deleted, please enter 'y' to confirm".format(self.directory))
        else:
            y = 'y'
        if y == 'y':
            shutil.rmtree(self.directory)
            os.makedirs(self.directory)

class CacheElements(Elements):
    def __init__(self, format_obj: "DatasetFormat", sub_dir, filllen=6, fillchar='0') -> None:
        super().__init__(format_obj, sub_dir, False, None, None, ".npy", filllen, fillchar)
        self.read_func = self._read_func
        self.write_func = self._write_func
    
    def _init_attr(self, *args, **kwargs):
        super()._init_attr(*args, **kwargs)
        self.read_func = self._read_func
        self.write_func = self._write_func

    def get_from_meta(self, meta:dict, name):
        value = meta[name]
        return value
        # if not np.issubdtype(value.dtype, np.number):
        #     return value.item()
        # else:
        #     return value

    def _read_func(self, path):
        meta = np.load(path,  allow_pickle= True).item()
        ids         = self.get_from_meta(meta, 'ids')
        rgb         = self.get_from_meta(meta, 'rgb')
        depth       = self.get_from_meta(meta, 'depth')
        mask_dict   = self.decompress_mask(ids, self.get_from_meta(meta, 'cprsd_mask'))
        extr_vecs   = self.zip_dict(ids, self.get_from_meta(meta, 'extr_vecs'))
        intr        = self.get_from_meta(meta, 'intr')
        depth_scale = self.get_from_meta(meta, 'depth_scale')
        bbox_3d     = self.zip_dict(ids, self.get_from_meta(meta, 'bbox_3d'))
        landmarks   = self.zip_dict(ids, self.get_from_meta(meta, 'landmarks'))
        visib_fract = self.zip_dict(ids, self.get_from_meta(meta, 'visib_fract'))

        viewmeta = ViewMeta(rgb, depth, mask_dict, extr_vecs, intr, depth_scale, bbox_3d, landmarks, visib_fract)
        return viewmeta

    def _write_func(self, path, viewmeta:ViewMeta):
        rgb = viewmeta.rgb
        depth = viewmeta.depth
        masks = viewmeta.masks
        ids, cprsd_mask = self.compress_mask(masks)
        ids, extr_vecs = self.split_dict(viewmeta.extr_vecs)
        intr = viewmeta.intr
        depth_scale = viewmeta.depth_scale
        ids, bbox_3d = self.split_dict(viewmeta.bbox_3d)
        ids, landmarks = self.split_dict(viewmeta.landmarks)
        ids, visib_fract = self.split_dict(viewmeta.visib_fract)
        np.save(path, 
                 {"ids":ids, 
                 "rgb":rgb, 
                 "depth":depth, 
                 "cprsd_mask":cprsd_mask, 
                 "extr_vecs":extr_vecs, "intr":intr, 
                 "depth_scale":depth_scale, "bbox_3d":bbox_3d, 
                 "landmarks":landmarks, "visib_fract":visib_fract})
        
    @staticmethod
    def zip_dict(ids:np.ndarray, array:np.ndarray) -> dict[int, np.ndarray]:
        if array is None:
            return None
        dict_ = dict(zip(ids, array))
        return dict_

    @staticmethod
    def split_dict(dict_:dict[int, np.ndarray]):
        if dict_ is None:
            return None, None
        return np.array(list(dict_.keys())), np.array(list(dict_.values()))

    @staticmethod
    def compress_mask(mask_dict: dict[int, np.ndarray]):
        if mask_dict is None:
            return None
        length = len(mask_dict)
        if length <= 8:
            dtype = np.uint8
        elif length <= 16:
            dtype = np.uint16
        elif length <= 32:
            dtype = np.uint32
        elif length <= 64:
            dtype = np.uint64
        elif length <= 128:
            dtype = np.uint128
        else:
            dtype = np.uint256
        mask_list = np.array(list(mask_dict.values()))
        mask_list = mask_list & 1
        maks_ids = np.array(list(mask_dict.keys()))
        cprsd_mask = np.zeros((mask_list[0].shape[0], mask_list[0].shape[1]), dtype=dtype)
        for shift, m in enumerate(mask_list):
            m = m << shift        
            cprsd_mask = np.bitwise_or(cprsd_mask, m)    
        return maks_ids, cprsd_mask

    @staticmethod
    def decompress_mask(ids:np.ndarray, masks:np.ndarray):
        if masks is None:
            return None
        mask_dict = {}
        for i, id in enumerate(ids):
            mask_dict[id] = (masks & (1 << i)).astype(np.bool8)
        return mask_dict


class DatasetFormatMode(enumerate):
    NORMAL = 0
    ONLY_CACHE = 1

class DatasetFormat(ABC):
    '''
    # Dataset Format
    -----
    A dataset manager for 6D pose estimation, based on the general .viewmeta.ViewMeta for data reading and writing

    built-in data types
    -----
    * _BaseJsonDict: Basic json dictionary, this data type allows data to be stored in json format. This type of data must use DatasetFormat.load_basejson
    * _Elements: element storage, this data type will save each set of data in a separate file in the directory
    * _Writer: write context manager, must call start_to_write method before writing

    virtual function
    ----
    * read_one: Read a piece of data, range ViewMeat object
    * write_element: write element

    rewritable
    ----
    * _parse_viewmeta_for_basejson

    example
    -----
    df1 = DatasetFormat(directory1) 
    
    df2 = DatasetFormat(directory2) 

    for viewmeta in self.read_from_disk(): 
        ...

    #df2.write_to_disk(viewmeta) × this is wrong 

    with df2.start_to_write():
        df2.write_to_disk(viewmeta)
    
    df2.clear()
    '''
    KW_CAM_K = "cam_K"
    KW_CAM_DS = "depth_scale"
    KW_CAM_VL = "view_level"

    class _Writer:
        '''
        write context manager
        -----
        call self.format_obj._dump_cache() when exit
        '''
        def __init__(self, format_obj:"DatasetFormat", cover = False) -> None:
            self.format_obj: DatasetFormat = format_obj
            self.format_obj.cover_write = cover

        def __enter__(self):
            print(f"start to write to {self.format_obj.directory}")
            with open(self.mark_file(self.format_obj.directory), 'w'): # create a file to mark that the DatasetFormat is writing
                pass
            self.format_obj.set_all_read_only(False)
            return self
        
        def __exit__(self, exc_type, exc_value:Exception, traceback):
            self.format_obj._dump_cache()
            if exc_type is not None:
                raise exc_type(exc_value).with_traceback(traceback)
                return False
            else:
                os.remove(self.mark_file(self.format_obj.directory))
                self.format_obj.set_all_read_only(True)
                return True
            
        @staticmethod
        def mark_file(directory):
            return os.path.join(directory, ".dfsw")
    
    def __init__(self, directory, clear_incomplete = False, init_mode = DatasetFormatMode.NORMAL) -> None:
        self.directory:str = directory
        if not init_mode == DatasetFormatMode.ONLY_CACHE:
            self.incomplete = os.path.exists(self._Writer.mark_file(self.directory))
            if self.incomplete:
                if clear_incomplete:
                    pass
                else:
                    tip = "the dataset is incomplete, if you want to clear all data, input 'y', else the program will exit: "
                    print("="*len(tip), '\n', tip, '\n', "="*len(tip))
                    y = input()
                    if y != 'y':
                        raise ValueError("the dataset is incomplete")
            self.writer:DatasetFormat._Writer = None       
            self.cover_write = True  
            self.stream_dumping_json = True

            self.base_json:dict[str, BaseJsonDict] = {}
            self.elements:dict[str, Elements] = {}
            self._streams:dict[str, JsonIO.Stream] = {}

            self._init_clusters()

            self._updata_data_num()

            if self.incomplete:
                os.remove(self._Writer.mark_file(self.directory))
                self.incomplete = False
        self.cache_elements = CacheElements(self, "cache")
        # def deserialize_viewmeta(path)->ViewMeta:
        #     return ViewMeta.from_serialize_object(deserialize_object(path))
        
        # def serialize_viewmeta(path, viewmeta:ViewMeta):
        #     se = viewmeta.serialize()
        #     serialize_object(path, se)

        # self.serialized_element = Elements(self, 
        #                                     "serialized", 
        #                                     deserialize_viewmeta, 
        #                                     serialize_viewmeta,
        #                                     '.pkl')
        # self.serialized_element.close()

    def _init_clusters(self):
        self.scene_camera_info          = BaseJsonDict(self, "scene_camera.json")
        self.scene_visib_fract_info     = BaseJsonDict(self, "scene_visib_fract.json")
        self.scene_bbox_3d_info         = BaseJsonDict(self, 'scene_bbox_3d.json')   
        self.scene_landmarks_info       = BaseJsonDict(self, 'scene_landmarks.json')   
        self.scene_trans_vector_info    = BaseJsonDict(self, 'scene_trans_vector.json')

    # def _load_basejson(self, file):
    #     '''
    #     load and bind
    #     '''
    #     if os.path.exists(file):
    #         value = BaseJsonDict(self, JsonIO.load_json(file))
    #     else:
    #         value = BaseJsonDict(self, {})
    #     if file not in self.base_json:
    #         self.base_json.update({file: value})
    #     else:
    #         self.base_json[file].update(value)
    #     return value

    def read_from_disk(self):
        '''
        brief
        ----
        *generator
        Since the amount of data may be large, return one by one
        '''
        for i in range(self.data_num):
            yield self.read_one(i)

    @abstractmethod
    def read_one(self, data_i)->ViewMeta:
        '''
        brief
        -----
        * abstractmethod
        read one piece of data

        parameter
        -----
        data_i: the index of data

        return
        ----
        ViewMeta instance
        '''
        pass

    def _read_source_info(self, data_i):
        camera      = self.scene_camera_info[data_i]
        visib_fract = self.scene_visib_fract_info[data_i]
        bbox_3d     = self.scene_bbox_3d_info[data_i]
        landmarks   = self.scene_landmarks_info[data_i]
        extr_vecs   = self.scene_trans_vector_info[data_i]
        return camera, visib_fract, bbox_3d, landmarks, extr_vecs 

    def start_to_write(self, cover = False):
        '''
        must be called before 'write_to_disk'
        '''
        if self.writer is None:
            self.writer = self._Writer(self, cover)
            return self.writer

    def _dump_cache(self):
        '''
        dump all base json dicts to disk
        automatically called when exiting the context of self.writer
        '''
        if self.stream_dumping_json:
            self._streams.clear()
            for path in self.base_json.keys():
                BaseJsonDict(self, path)
        else:
            for file, value in self.base_json.items():
                JsonIO.dump_json(file, value)
        self.writer = None

    def calc_by_base(self, mesh_dict:dict[int, MeshMeta], cover = False):
        '''
        brief
        -----
        calculate data by base data, see ViewMeta.calc_by_base
        '''
        with self.start_to_write():
            for i in range(self.data_num):
                viewmeta = self.read_one(i)
                viewmeta.calc_by_base(mesh_dict, cover=cover)
                self.write_to_disk(viewmeta, i)

    def write_to_disk(self, viewmeta:ViewMeta, data_i = -1):
        '''
        biref
        -----
        write elements immediately, write basejsoninfo to cache, they will be dumped when exiting the context of self.writer
        
        NOTE
        -----
        For DatasetFormat, the write mode has only 'append'. 
        If you need to modify, please call 'DatasetFormat.clear' to clear all data, and then write again.
        '''
        if self.writer is None:
            raise ValueError("please call 'self.start_to_write' first")
        self._updata_data_num()        
        if data_i == -1:
            data_i = self.data_num
        
        self.write_element(viewmeta, data_i)
        parse = self._parse_viewmeta_for_basejson(viewmeta, data_i)
        self._cache_source_info(parse)

    @abstractmethod
    def write_element(self, viewmeta:ViewMeta, data_i:int):
        pass

    def _parse_viewmeta_for_basejson(self, viewmeta:ViewMeta, data_i):
        parse = {
            self.scene_camera_info.directory: {data_i: {LinemodFormat.KW_CAM_K: viewmeta.intr.reshape(-1).tolist(),
                                                              LinemodFormat.KW_CAM_DS: float(viewmeta.depth_scale),
                                                              LinemodFormat.KW_CAM_VL: 1}},
            self.scene_visib_fract_info.directory: {data_i: viewmeta.visib_fract},
            #
            self.scene_bbox_3d_info.directory: {data_i: viewmeta.bbox_3d},
            #
            self.scene_landmarks_info.directory: {data_i: viewmeta.landmarks},
            #
            self.scene_trans_vector_info.directory: {data_i: viewmeta.extr_vecs},
        }
        return parse

    def _cache_source_info(self, parsed:dict):
        for path in self.base_json.keys():
            if self.stream_dumping_json and not self.base_json[path].is_close() and not self.base_json[path].is_read_only():
                if path not in self._streams:
                    self._streams[path] = JsonIO.create_stream(path)                
                self._streams[path].write(parsed[path])

                keys = parsed[path].keys()
                self.base_json[path].update(dict(zip(keys, [None for _ in keys])))
            else:
                self.base_json[path].update(parsed[path])

    def _updata_data_num(self):
        '''
        The total number of data of all types must be equal, otherwise an exception is thrown
        '''
        datas = list(self.base_json.values()) + list(self.elements.values())
        datas = [x for x in datas if not x.is_close()]
        nums = [len(x) for x in datas]
        num = np.unique(nums)
        if len(num) > 1:
            raise ValueError("Unknown error, the numbers of different datas are not equal")
        else:
            self.data_num = int(num)

    def clear(self, ignore_warning = False):
        '''
        brief
        -----
        clear all data, defalut to ask before executing
        '''
        if not ignore_warning:
            y = input("All files in {} will be deleted, please enter 'y' to confirm".format(self.directory))
        else:
            y = 'y'
        if y == 'y':
            for path in list(self.elements.keys()) + list(self.base_json.keys()):
                if os.path.exists(path):
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                        os.makedirs(path)
                    else:
                        os.remove(path)
            [x.clear() for x in self.base_json.values()]
        self._updata_data_num()

    def close_all(self, value = True):
        for obj in list(self.elements.values()) + list(self.base_json.values()):
            obj.close() if value else obj.open()

    def set_all_read_only(self, value = True):
        for obj in list(self.elements.values()) + list(self.base_json.values()):
            obj.set_read_only(value)


class LinemodFormat(DatasetFormat):
    KW_GT_R = "cam_R_m2c"
    KW_GT_t = "cam_t_m2c"
    KW_GT_ID = "obj_id"
    
    class _MasksElements(Elements):
        def id_format(self, class_id):
            id_format = "_" + str(class_id).rjust(6, "0")
            return id_format

        def read(self, data_i):
            masks = {}
            for n, scene_gt in enumerate(self.format_obj.scene_gt_info[data_i]):
                id_ = scene_gt[LinemodFormat.KW_GT_ID]
                mask = super().read(data_i, appname=self.id_format(n))
                if mask is None:
                    continue
                masks[id_] = mask
            return masks
        
        def write(self, data_i, masks:dict[int, np.ndarray]):
            for n, mask in enumerate(masks.values()):
                super().write(data_i, mask, appname=self.id_format(n))

    def _init_clusters(self):
        super()._init_clusters()
        self.rgb_elements   = Elements(self,      "rgb",
                                       read_func=cv2.imread,                                    
                                       write_func=cv2.imwrite, 
                                       suffix='.png')
        self.depth_elements = Elements(self,      "depth",    
                                       read_func=lambda x:cv2.imread(x, cv2.IMREAD_ANYDEPTH),   
                                       write_func=cv2.imwrite, 
                                       suffix='.png')
        self.masks_elements = self._MasksElements(self, "mask",     
                                                  read_func=lambda x:cv2.imread(x, cv2.IMREAD_GRAYSCALE),  
                                                  write_func=cv2.imwrite, 
                                                  suffix='.png')

        self.scene_gt_info              = BaseJsonDict(self, "scene_gt.json")

    def write_element(self, viewmeta:ViewMeta, data_i:int):
        super().write_element(viewmeta, data_i)
        self.rgb_elements.  write(data_i, viewmeta.rgb)
        self.depth_elements.write(data_i, viewmeta.depth)
        self.masks_elements.write(data_i, viewmeta.masks)

    def _parse_viewmeta_for_basejson(self, viewmeta: ViewMeta, data_i):
        parsed = super()._parse_viewmeta_for_basejson(viewmeta, data_i)
        one_info = []
        for obj_id, trans_vecs in viewmeta.extr_vecs.items():
            posture = Posture(rvec=trans_vecs[0], tvec=trans_vecs[1])
            one_info .append(
                {   LinemodFormat.KW_GT_R: posture.rmat.reshape(-1).tolist(),
                    LinemodFormat.KW_GT_t: posture.tvec.reshape(-1).tolist(),
                    LinemodFormat.KW_GT_ID: int(obj_id)})
        parsed.update({self.scene_gt_info.directory: {self.data_num: one_info}})

        return parsed    

    def __parse_one_scene_gt_info(self, data_i):
        one_scene_gt_info:list[dict[str, np.ndarray]] = self.scene_gt_info[data_i]
        extr = {}
        for info in one_scene_gt_info:
            posture = Posture(rmat=info[LinemodFormat.KW_GT_R].reshape(3,3), 
                    tvec=info[LinemodFormat.KW_GT_t])
            extr[info[LinemodFormat.KW_GT_ID]] = np.array([posture.rvec, posture.tvec])
        return extr

    def read_one(self, data_i):
        super().read_one(data_i)
        rgb     = self.rgb_elements.read(data_i)
        depth   = self.depth_elements.read(data_i)
        masks   = self.masks_elements.read(data_i)
        intr           = self.scene_camera_info[data_i][LinemodFormat.KW_CAM_K].reshape(3, 3)
        depth_scale    = self.scene_camera_info[data_i][LinemodFormat.KW_CAM_DS]
        visib_fract    = self.scene_visib_fract_info[data_i]
        bbox_3d        = self.scene_bbox_3d_info[data_i]
        landmarks      = self.scene_landmarks_info[data_i]

        extr_vecs      = self.__parse_one_scene_gt_info(data_i)

        return ViewMeta(rgb, depth, masks, 
                        extr_vecs,
                        intr,
                        depth_scale,
                        bbox_3d,
                        landmarks,
                        visib_fract)

class VocFormat(DatasetFormat):
    KW_TRAIN = "train"
    KW_VAL = "val"

    def __init__(self, directory, data_num = 0, split_rate = 0.75, clear = False) -> None:
        super().__init__(directory, clear)

        self.train_txt = os.path.join(self.directory,   VocFormat.KW_TRAIN + ".txt")
        self.val_txt = os.path.join(self.directory,     VocFormat.KW_VAL + ".txt")
        self.get_split_file(data_num, split_rate)

    def _init_clusters(self):
        super()._init_clusters()
        self.images_elements     = Elements(self, "images",       
                                            read_func=cv2.imread, 
                                            write_func=cv2.imwrite,
                                            suffix = ".jpg")
        self.depth_elements      = Elements(self, "depths",       
                                            read_func=lambda x:cv2.imread(x, cv2.IMREAD_ANYDEPTH), 
                                            write_func=cv2.imwrite, 
                                            suffix = '.png')
        self.masks_elements      = Elements(self, "masks",        
                                            read_func=lambda x: deserialize_image_container(deserialize_object(x), cv2.IMREAD_GRAYSCALE),
                                            write_func=lambda path, x: serialize_object(path, serialize_image_container(x)),  
                                            suffix = ".pkl")
        self.labels_elements     = Elements(self, "labels",       
                                            read_func=self.loadtxt_func((-1, 5)), 
                                            write_func=self.savetxt_func("%8.4f"), 
                                            suffix=".txt")
        self.bbox_3ds_elements   = Elements(self, "bbox_3ds",     
                                            read_func=self.loadtxt_func((-1, 8, 2)), 
                                            write_func=self.savetxt_func("%12.6f"), 
                                            suffix=".txt")   
        self.landmarks_elements  = Elements(self, "landmarks",    
                                            read_func=self.loadtxt_func((-1, 24, 2)), 
                                            write_func=self.savetxt_func("%12.6f"), 
                                            suffix=".txt")
        self.extr_vecs_elements  = Elements(self, "trans_vecs",   
                                            read_func=self.loadtxt_func((-1, 2, 3)), 
                                            write_func=self.savetxt_func("%12.6f"), 
                                            suffix=".txt")


    @staticmethod
    def savetxt_func(fmt=...):
        return lambda path, x: np.savetxt(path, x, fmt=fmt, delimiter='\t')

    @staticmethod
    def loadtxt_func(shape:tuple[int]):
        return lambda path: np.loadtxt(path).reshape(shape)

    def get_split_file(self, data_num, split_rate):
        create = False
        if data_num == 0:
            # 存在则读取，不存在则创建
            if os.path.exists(self.train_txt):
                self.train_idx_array = np.loadtxt(self.train_txt).astype(np.int32)
                self.val_idx_array   = np.loadtxt(self.val_txt).astype(np.int32)
            else:
                create = True
        else:
            create = True

        if create:
            data_i_list = list(range(data_num))
            np.random.shuffle(data_i_list) 
            self.train_idx_array = np.array(data_i_list[: int(data_num*split_rate)]).astype(np.int32)
            self.val_idx_array   = np.array(data_i_list[int(data_num*split_rate): ]).astype(np.int32)
            np.savetxt(self.train_txt, self.train_idx_array, fmt = "%6d")
            np.savetxt(self.val_txt, self.val_idx_array, fmt = "%6d")

    def decide_set(self, data_i):
        if data_i in self.train_idx_array:
            sub_set = VocFormat.KW_TRAIN
        elif data_i in self.val_idx_array:
            sub_set = VocFormat.KW_VAL
        else:
            raise ValueError("can't find datas of index: {}".format(data_i))
        return sub_set

    @staticmethod
    def _x1x2y1y2_2_normedcxcywh(bbox_2d, img_size):
        lt = bbox_2d[:2]
        rb = bbox_2d[2:]

        cx, cy = (lt + rb) / 2
        w, h = rb - lt
        # 归一化
        cy, h = np.array([cy, h]) / img_size[0]
        cx, w = np.array([cx, w]) / img_size[1]
        return np.array([cx, cy, w, h])

    @staticmethod
    def _normedcxcywh_2_x1x2y1y2(bbox_2d, img_size):
        cx, cy, w, h = bbox_2d
        cy, h = np.array([cy, h]) / img_size[0]
        cx, w = np.array([cx, w]) / img_size[1]

        x1 = cx - w/2
        x2 = cx + w/2
        y1 = cy - h/2
        y2 = cy + h/2

        new_bbox_2d = np.array([x1, y1, x2, y2]).astype(np.int32)
        return new_bbox_2d

    def write_element(self, viewmeta: ViewMeta, data_i: int):
        super().write_element(viewmeta, data_i)
        sub_set = self.decide_set(data_i)
        #
        self.images_elements.write(data_i, viewmeta.rgb, appdir=sub_set)
        #
        self.depth_elements.write(data_i, viewmeta.depth, appdir=sub_set)
        #
        self.masks_elements.write(data_i, viewmeta.masks, appdir=sub_set)
        viewmeta_bbox2d = viewmeta.bbox_2d
        ###
        labels = []
        landmarks_list = []
        bbox_list = []
        extr_vecs_list = []
        for id_, mask in viewmeta.masks.items():
            img_size = mask.shape
            point = np.array(np.where(mask))
            if point.size == 0:
                continue
            bbox_2d = viewmeta_bbox2d[id_]
            cx, cy, w, h = self._x1x2y1y2_2_normedcxcywh(bbox_2d, img_size)
            labels.append([id_, cx, cy, w, h])

            # 转换关键点
            ldmk = viewmeta.landmarks[id_] #
            ldmk = ldmk.reshape(-1).tolist()
            landmarks_list.append(ldmk)

            # 
            bbox_3d = viewmeta.bbox_3d[id_]
            bbox_3d = bbox_3d.reshape(-1).tolist()
            bbox_list.append(bbox_3d)

            #
            ev = viewmeta.extr_vecs[id_]
            extr_vecs_list.append(ev.reshape(-1))
        self.labels_elements.write(data_i, labels, appdir=sub_set)
        self.bbox_3ds_elements.write(data_i, bbox_list, appdir=sub_set)
        self.landmarks_elements.write(data_i, landmarks_list, appdir=sub_set)
        self.extr_vecs_elements.write(data_i, extr_vecs_list, appdir=sub_set)
    
    def read_one(self, data_i) -> ViewMeta:
        def as_dict(ids, objs):
            if objs is None:
                return None
            else:
                return dict(zip(ids, objs))
        super().read_one(data_i)
        # 判断data_i属于train或者val
        sub_set = self.decide_set(data_i)
        # 读取
        rgb = self.images_elements.read(data_i, appdir=sub_set)
        #
        depth = self.depth_elements.read(data_i, appdir=sub_set)
        #
        ids = self.labels_elements.read(data_i, appdir=sub_set)[:,0].astype(np.int32).tolist()
        bbox_2d = self.labels_elements.read(data_i, appdir=sub_set)[:,1:].astype(np.int32)
        masks_dict = self.masks_elements.read(data_i, appdir=sub_set)
        #
        extr_vecs = self.extr_vecs_elements.read(data_i, appdir=sub_set)
        extr_vecs_dict = as_dict(ids, extr_vecs)
        #
        bbox_3d = self.bbox_3ds_elements.read(data_i, appdir=sub_set)
        bbox_3d_dict = as_dict(ids, bbox_3d)
        #
        landmarks = self.landmarks_elements.read(data_i, appdir=sub_set)
        landmarks_dict = as_dict(ids, landmarks)

        camera, visib_fract, _, _, _  = self._read_source_info(data_i)
        return ViewMeta(rgb, depth, masks_dict,
                        extr_vecs_dict,
                        camera[DatasetFormat.KW_CAM_K],
                        camera[DatasetFormat.KW_CAM_DS],
                        bbox_3d_dict,
                        landmarks_dict,
                        visib_fract)

class _LinemodFormat_sub1(LinemodFormat):
    class _MasksElements(Elements):
        def id_format(self, class_id):
            id_format = "_" + str(class_id).rjust(6, "0")
            return id_format

        def read(self, data_i):
            masks = {}
            for n in range(100):
                mask = super().read(data_i, appname=self.id_format(n))
                if mask is None:
                    continue
                masks[n] = mask
            return masks
        
        def write(self, data_i, masks:dict[int, np.ndarray]):
            for id_, mask in masks.items():
                super().write(data_i, mask, appname=self.id_format(id_))

    def __init__(self, directory, clear = False) -> None:
        super().__init__(directory, clear)
        self.rgb_elements   = Elements(self, "rgb", 
                                       read_func=cv2.imread,  
                                       write_func=cv2.imwrite, 
                                       suffix='.jpg')

    def read_one(self, data_i):
        viewmeta = super().read_one(data_i)

        for k in viewmeta.bbox_3d:
            viewmeta.bbox_3d[k] = viewmeta.bbox_3d[k][:, ::-1]
        for k in viewmeta.landmarks:
            viewmeta.landmarks[k] = viewmeta.landmarks[k][:, ::-1]
        viewmeta.depth_scale *= 1000

        return viewmeta


def serialize_object(file_path, obj:dict):
    # if os.path.splitext(file_path)[1] == '.pkl':
    #     file_path = os.path.splitext(file_path)[0] + ".npz"
    # np.savez(file_path, **obj)
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)

# 从文件反序列化对象
def deserialize_object(serialized_file_path):
    with open(serialized_file_path, 'rb') as file:
        elements = pickle.load(file)
        return elements
    # if os.path.splitext(serialized_file_path)[1] == '.pkl':
    #     with open(serialized_file_path, 'rb') as file:
    #         elements = pickle.load(file)
    #         return elements
    # else:
    #     elements = dict(np.load(serialized_file_path, allow_pickle=True))
        
    #     # restored_elements = {}
    #     for key in elements:
    #         if not np.issubdtype(elements[key].dtype, np.number):
    #             elements[key] = elements[key].item()
    #     return elements
    # with open(serialized_file_path, 'rb') as file:
    #     elements = pickle.load(file)
    #     return elements