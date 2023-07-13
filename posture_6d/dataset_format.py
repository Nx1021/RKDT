# from compute_gt_poses import GtPostureComputer

# from toolfunc import *
from MyLib.posture_6d.viewmeta import ViewMeta
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from open3d import geometry, utility, io
import os
import shutil
import cv2
import time
import json
import scipy.ndimage as image
import skimage
import scipy.spatial as spt

from abc import ABC, abstractmethod
from typing import Union, Callable

from .viewmeta import ViewMeta
from .posture import Posture
from .mesh_manager import MeshMeta
from .utils import JsonIO


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
    * _write_element: write element

    example
    -----
    df1 = DatasetFormat(directory1) 
    
    df2 = DatasetFormat(directory2) 

    for viewmeta in self.read_from_disk(): 
        ...

    #df2.write_to_disk(viewmeta) ×wrong 

    with df2.start_to_write():
        df2.write_to_disk(viewmeta)
    
    df2.clear()
    '''
    KW_CAM_K = "cam_K"
    KW_CAM_DS = "depth_scale"
    KW_CAM_VL = "view_level"
    
    class _BaseJsonDict(dict):
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
        def __init__(self, format_obj:"DatasetFormat", *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.format_obj:DatasetFormat = format_obj
            self.closed = False #控制读写屏蔽

        @property
        def cover_write(self):
            return self.format_obj.cover_write

        def __getitem__(self, key):
            if self.closed:
                return None
            try:
                return super().__getitem__(key)
            except KeyError:
                return None
            
        def __setitem__(self, key, value):
            if (self.cover_write or key not in self) and not self.closed:
                super().__setitem__(key, value)

    class _Elements:
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
                    dir_,
                    read_func:Callable, 
                    write_func:Callable, 
                    suffix:str, 
                    filllen = 6, 
                    fillchar = '0') -> None:
            self.dir_ = dir_
            
            self.filllen    = filllen
            self.fillchar   = fillchar
            self.suffix     = suffix
            if not self.suffix.startswith('.'):
                self.suffix = '.' + self.suffix
            self.read_func  = read_func
            self.write_func = write_func

            self.format_obj:DatasetFormat = format_obj

            self.closed = False #控制读写屏蔽
            os.makedirs(self.dir_, exist_ok=True)

            self.format_obj.elements[dir_] = self # register to format_obj

        def __len__(self):
            '''
            Count the total number of files in the directory
            '''
            count = 0
            for root, dirs, files in os.walk(self.dir_):
                count += len(files)
            return count

        @property
        def cover_write(self):
            return self.format_obj.cover_write

        def read(self, data_i, appdir = "", appname = ""):
            path = self.path_format(data_i, appdir=appdir, appname=appname)
            if os.path.exists(path) and not self.closed:
                return self.read_func(path)
            else:
                return None

        def write(self, data_i, element, appdir = "", appname = ""):
            path = self.path_format(data_i, appdir=appdir, appname=appname)
            dir_ = os.path.split(path)[0]
            os.makedirs(dir_, exist_ok=True)
            if element is not None and not self.closed:
                self.write_func(path, element)

        def path_format(self, data_i, appdir = "", appname = ""):
            return os.path.join(self.dir_, appdir, "{}{}{}".format(str(data_i).rjust(6, "0"), appname, self.suffix))

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
            print("Entering the context")
            return self
        
        def __exit__(self, exc_type, exc_value, traceback):
            if exc_type is not None:
                return False
            self.format_obj._dump_cache()
            return True
    
    def __init__(self, directory) -> None:
        self.directory = directory
        self.writer = None       
        self.cover_write = False       

        self.base_json = {}
        self.elements = {}
        self.scene_camera_file          = os.path.join(self.directory, "scene_camera.json")
        self.scene_visib_fract_file     = os.path.join(self.directory, "scene_visib_fract.json")
        self.scene_bbox_3d_file         = os.path.join(self.directory, 'scene_bbox_3d.json')   
        self.scene_landmarks_file       = os.path.join(self.directory, 'scene_landmarks.json')   
        self.scene_trans_vector_file    = os.path.join(self.directory, 'scene_trans_vector.json')

        self.scene_camera_info          = self._load_basejson(self.scene_camera_file )
        self.scene_visib_fract_info     = self._load_basejson(self.scene_visib_fract_file )         
        self.scene_bbox_3d_info         = self._load_basejson(self.scene_bbox_3d_file   )  
        self.scene_landmarks_info       = self._load_basejson(self.scene_landmarks_file   ) 
        self.scene_trans_vector_info    = self._load_basejson(self.scene_trans_vector_file)

        self.data_num = len(self.scene_trans_vector_info)

    def _load_basejson(self, file):
        '''
        load and bind
        '''
        if os.path.exists(file):
            value = self._BaseJsonDict(self, JsonIO.load_json(file))
        else:
            value = self._BaseJsonDict(self, {})
        self.base_json.update({file: value})
        return value

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
        if data_i == -1:
            data_i = self.data_num
        self._write_element(viewmeta, data_i)
        self._cache_source_info(viewmeta, data_i)
        self._updata_data_num()

    @abstractmethod
    def _write_element(self, viewmeta:ViewMeta, data_i:int):
        pass

    def _cache_source_info(self, viewmeta:ViewMeta, data_i:int):
        #
        self.scene_camera_info.update({data_i: {LinemodFormat.KW_CAM_K: viewmeta.intr.reshape(-1).tolist(),
                                                              LinemodFormat.KW_CAM_DS: float(viewmeta.depth_scale),
                                                              LinemodFormat.KW_CAM_VL: 1}})
        #
        self.scene_visib_fract_info.update({data_i: viewmeta.visib_fract})
        #
        self.scene_bbox_3d_info.update({data_i: viewmeta.bbox_3d})
        #
        self.scene_landmarks_info.update({data_i: viewmeta.landmarks})
        #
        self.scene_trans_vector_info.update({data_i: viewmeta.landmarks})

    def _updata_data_num(self):
        '''
        The total number of data of all types must be equal, otherwise an exception is thrown
        '''
        datas = list(self.base_json.values()) + list(self.elements.values())
        datas = [x for x in datas if not x.closed]
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

class LinemodFormat(DatasetFormat):
    KW_GT_R = "cam_R_m2c"
    KW_GT_t = "cam_t_m2c"
    KW_GT_ID = "obj_id"
    
    class _MasksElements(DatasetFormat._Elements):
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

    def __init__(self, directory) -> None:
        super().__init__(directory)
        self.rgb_dir    = os.path.join(self.directory, "rgb")
        self.depth_dir  = os.path.join(self.directory, "depth")        
        self.mask_dir   = os.path.join(self.directory, "mask")

        self.rgb_elements   = self._Elements(self, self.rgb_dir,      cv2.imread,                                    cv2.imwrite, '.png')
        self.depth_elements = self._Elements(self, self.depth_dir,    lambda x:cv2.imread(x, cv2.IMREAD_ANYDEPTH),   cv2.imwrite, '.png')
        self.masks_elements = self._MasksElements(self, self.mask_dir,lambda x:cv2.imread(x, cv2.IMREAD_GRAYSCALE), cv2.imwrite, '.png')

        self.scene_gt_file              = os.path.join(self.directory, "scene_gt.json")
        self.scene_gt_info              = self._load_basejson(self.scene_gt_file)

        self.data_num = len(os.listdir(self.rgb_dir))

    def _write_element(self, viewmeta:ViewMeta, data_i:int):
        super()._write_element(viewmeta, data_i)
        self.rgb_elements.  write(data_i, viewmeta.rgb)
        self.depth_elements.write(data_i, viewmeta.depth)
        self.masks_elements.write(data_i, viewmeta.masks)

    def _cache_source_info(self, viewmeta: ViewMeta):
        super()._cache_source_info(viewmeta)
        one_info = []
        for obj_id, trans_vecs in viewmeta.extr_vecs.items():
            posture = Posture(rvec=trans_vecs[0], tvec=trans_vecs[1])
            one_info .append(
                {   LinemodFormat.KW_GT_R: posture.rmat.reshape(-1).tolist(),
                    LinemodFormat.KW_GT_t: posture.tvec.reshape(-1).tolist(),
                    LinemodFormat.KW_GT_ID: int(obj_id)})
        self.scene_gt_info.update({self.data_num: one_info})        

    def __parse_one_scene_gt_info(self, data_i):
        one_scene_gt_info:list[dict[str, np.ndarray]] = self.scene_gt_info[data_i]
        extr = {}
        for info in one_scene_gt_info:
            posture = Posture(rmat=info[LinemodFormat.KW_GT_R].reshape(3,3), 
                    tvec=info[LinemodFormat.KW_GT_t])
            if posture.tvec.max() > 10:
                posture.set_tvec(posture.tvec/1000)
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

    def __init__(self, directory, data_num = 0, split_rate = 0.75) -> None:
        super().__init__(directory)
        self.images_dir     = os.path.join(self.directory, "images")
        self.depths_dir     = os.path.join(self.directory, "depths")
        self.masks_dir      = os.path.join(self.directory, "masks")
        self.labels_dir     = os.path.join(self.directory, "labels")
        self.bbox_3ds_dir   = os.path.join(self.directory, "bbox_3ds")
        self.landmarks_dir  = os.path.join(self.directory, "landmarks")
        self.trans_vecs_dir = os.path.join(self.directory, "trans_vecs")

        self.images_elements     = self._Elements(self, self.images_dir,     cv2.imread, cv2.imwrite,                ".jpg")
        self.depth_elements      = self._Elements(self, self.depths_dir,     lambda x:cv2.imread(x, cv2.IMREAD_ANYDEPTH), cv2.imwrite, '.png')
        self.masks_elements      = self._Elements(self, self.masks_dir,      np.load,    np.save,                    ".npy")
        self.labels_elements     = self._Elements(self, self.labels_dir,     np.loadtxt, self.savetxt_func("%8.4f"), ".txt")
        self.bbox_3ds_elements   = self._Elements(self, self.bbox_3ds_dir,   np.loadtxt, self.savetxt_func("%12.6f"), ".txt")   
        self.landmarks_elements  = self._Elements(self, self.landmarks_dir,  np.loadtxt, self.savetxt_func("%12.6f"), ".txt")
        self.extr_vecs_elements  = self._Elements(self, self.trans_vecs_dir, np.loadtxt, self.savetxt_func("%12.6f"), ".txt")

        self.train_txt = os.path.join(self.directory,   VocFormat.KW_TRAIN + ".txt")
        self.val_txt = os.path.join(self.directory,     VocFormat.KW_VAL + ".txt")
        self.get_split_file(data_num, split_rate)

    @staticmethod
    def savetxt_func(fmt=...):
        return lambda path, x: np.savetxt(path, x, fmt=fmt, delimiter='\t')

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

    def _write_element(self, viewmeta: ViewMeta, data_i: int):
        super()._write_element(viewmeta, data_i)
        sub_set = self.decide_set(data_i)
        #
        self.images_elements.write(data_i, viewmeta.rgb, appdir=sub_set)
        #
        self.depth_elements.write(data_i, viewmeta.depth, appdir=sub_set)
        #
        self.masks_elements.write(data_i, np.stack(list(viewmeta.masks.values())), appdir=sub_set)
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
            lt = np.min(point, axis = -1)
            rb = np.max(point, axis = 1)

            cy, cx = (lt + rb) / 2
            h, w = rb - lt 
            # 归一化
            cy, h = np.array([cy, h]) / img_size[0]
            cx, w = np.array([cx, w]) / img_size[1]
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
        super().read_one(data_i)
        # 判断data_i属于train或者val
        sub_set = self.decide_set(data_i)
        # 读取
        rgb = self.images_elements.read(data_i, appdir=sub_set)
        #
        depth = self.depth_elements.read(data_i, appdir=sub_set)
        #
        ids = self.labels_elements.read(data_i, appdir=sub_set)[:,0].astype(np.int32).tolist()
        masks = self.masks_elements.read(data_i, appdir=sub_set)
        masks_dict:dict[int, np.ndarray] = dict(zip(ids, masks))
        #
        extr_vecs = self.extr_vecs_elements.read(data_i, appdir=sub_set)
        extr_vecs_dict = dict(zip(ids, extr_vecs))
        #
        landmarks = self.landmarks_elements.read(data_i, appdir=sub_set)
        landmarks_dict = dict(zip(ids, landmarks))

        camera, visib_fract, bbox_3d, _, _  = self._read_source_info(data_i)
        return ViewMeta(rgb, depth, masks_dict,
                        extr_vecs_dict,
                        camera[DatasetFormat.KW_CAM_K],
                        camera[DatasetFormat.KW_CAM_DS],
                        bbox_3d,
                        landmarks_dict,
                        visib_fract)

class _LinemodFormat_sub1(LinemodFormat):
    class _MasksElements(DatasetFormat._Elements):
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

    def __init__(self, directory) -> None:
        super().__init__(directory)
        self.rgb_elements   = self._Elements(self, self.rgb_dir,      cv2.imread,                                    cv2.imwrite, '.jpg')