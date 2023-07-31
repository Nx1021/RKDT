# from compute_gt_poses import GtPostureComputer

# from toolfunc import *
import matplotlib.pyplot as plt
import numpy as np
from open3d import geometry, utility, io
import os
import glob
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

def as_dict(ids, objs):
    if objs is None:
        return None
    else:
        return dict(zip(ids, objs))

def savetxt_func(fmt=...):
    return lambda path, x: np.savetxt(path, x, fmt=fmt, delimiter='\t')

def loadtxt_func(shape:tuple[int]):
    return lambda path: np.loadtxt(path).reshape(shape)

class _DataCluster():
    '''
    This is a private class representing a data cluster used for managing datasets with a specific format.

    attr
    ----
    * self.format_obj: DatasetFormat
    * self.closed: bool, Control the shielding of reading and writing, 
        if it is true, the instance will not write, and the read will get None
    * register: bool, whether to register to format_obj
    * _incomplete: bool, whether the data is incomplete
    * _closed: bool, Indicates whether the cluster is closed or open.
    * _read_only: bool, Indicates whether the cluster is read-only or write-enabled.
    * _updated: bool, Indicates if any changes have been made to the cluster.
    * directory: str, Directory path for the cluster.

    property
    -----
    * cover_write: bool, Control the shielding of writing,

    method
    -----
    * _init_attr: initialize additional attributes specified by subclasses.
    * open: open the cluster for operation.
    * close: close the cluster, preventing further operations.
    * is_close: check if the cluster is closed.
    * set_read_only: set the cluster as read-only or write-enabled.
    * is_read_only: check if the cluster is read-only.
    * cluster_closed_decorator: decorator function to handle writing operations when the cluster is closed or read-only.
    * clear: clear any data in the cluster. Subclasses may implement this method.
    '''
    def __init__(self, format_obj: "DatasetFormat", sub_dir: str, register=True, *args, **kwargs) -> None:
        '''Initialize the data cluster with the provided format_obj, sub_dir, and registration flag.'''
        self.format_obj = format_obj
        self.sub_dir = sub_dir
        self.register = register
        self._incomplete = self.format_obj.incomplete
        self._closed = True  # Indicates whether the cluster is closed or open.
        self._read_only = True  # Indicates whether the cluster is read-only or write-enabled.
        self._updated = False  # Indicates if any changes have been made to the cluster.
        self.directory = os.path.join(format_obj.directory, self.sub_dir)  # Directory path for the cluster.
        self._init_attr(*args, **kwargs)  # Initializes additional attributes specified by subclasses.
        if os.path.exists(self.directory):
            self.open()  # Opens the cluster for operation.
        else:
            self.close()
        
    @property
    def cover_write(self):
        '''Property that returns whether the cluster format allows write operations.'''
        return self.format_obj.cover_write

    def _init_attr(self, *args, **kwargs):
        '''Method to initialize additional attributes specified by subclasses.'''
        pass

    def open(self):
        '''Method to open the cluster for operation.'''
        self._closed = False  # Marks the cluster as open for operation.        
        if self._incomplete:
            read_only = self._read_only  # Stores the read-only flag.
            self.set_read_only(False)  # Sets the cluster as write-enabled.
            self.clear(True)  # Clears any incomplete data if present.
            self._incomplete = False
            self.set_read_only(read_only)  # Restores the read-only flag.

    def close(self):
        '''Method to close the cluster, preventing further operations.'''
        self._updated = False  # Resets the updated flag to false.
        self._closed = True  # Marks the cluster as closed.

    def is_close(self):
        '''Method to check if the cluster is closed.'''
        return self._closed

    def set_read_only(self, read_only=True):
        '''Method to set the cluster as read-only or write-enabled.'''
        self._read_only = read_only

    def is_read_only(self):
        '''Method to check if the cluster is read-only.'''
        return self._read_only

    @staticmethod
    def cluster_closed_decorator(is_write_func):
        '''
        brief
        -----
        Decorator function to handle writing operations when the cluster is closed or read-only. \n
        if the cluster is closed, the decorated function will not be executed and return None. \n
        if the cluster is read-only and the decorated function is a writing operation, the decorated function will not be executed and return None.\n
        
        parameter
        -----
        is_write_func: bool, indicates whether the decorated function is a writing operation. 
        '''
        def write_func_wrapper(func):
            def wrapper(self: "_DataCluster", *args, **kwargs):
                if self._closed:
                    warnings.warn(f"{self.__class__.__name__}:{self.sub_dir} is closed, any I/O operation will not be executed.",
                                  Warning)
                    return None
                elif self._read_only and is_write_func:
                    warnings.warn(f"{self.__class__.__name__}:{self.sub_dir} is read-only, any write operation will not be executed.",
                                  Warning)
                    return None
                else:
                    rlt = func(self, *args, **kwargs)  # Calls the original function.
                    self._updated = is_write_func  # Marks the cluster as updated after writing operations.
                    return rlt
            return wrapper
        return write_func_wrapper

    def clear(self, ignore_warning = False):
        '''
        Method to clear any data in the cluster. Subclasses may implement this method.
        * it is dargerous
        '''
        if not ignore_warning:
            y = input("All files in {} will be deleted, please enter 'y' to confirm".format(self.directory))
        else:
            y = 'y'
        if y == 'y':
            return True
        else:
            return False

class JsonDict(dict, _DataCluster):
    '''
    dict for base json
    ----
    it is a subclass of dict, so it can be used as a dict \n
    returns None if accessing an key that does not exist

    attr
    ----
    see _DataCluster

    method
    ----
    see _DataCluster
    * clear: clear all data of the dict and clear the json file
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
            if self.directory not in self.format_obj.jsons:
                self.format_obj.jsons.update({self.directory: self})
            else:
                self.format_obj.jsons[self.directory].update(self) 

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

    @_DataCluster.cluster_closed_decorator(True)
    def clear(self, ignore_warning = False):
        '''
        brief
        -----
        clear all data and remove the directory, 
        defalut to ask before executing
        '''
        if _DataCluster.clear(self, ignore_warning):
            with open(self.directory, 'w'):
                pass
            return True
        else:
            return False

class Elements(_DataCluster):
    '''
    elements manager
    ----
    Returns None if accessing an data_id that does not exist \n
    will not write if the element is None \n 
    it can be used as an iterator, the iterator will return (data_id, element) \n

    attr
    ----
    see _DataCluster
    * readfunc:  Callable, how to read one element from disk, the parameter must be (path)
    * writefunc: Callable, how to write one element to disk, the parameter must be (path, element)
    * suffix: str, the suffix of the file
    * filllen: int, to generate the store path
    * fillchar: str, to generate the store path
    * _data_i_dir_map: dict[int, str], the map of data_id and directory name
    * _index: int, the index of the iterator
    * _max_idx: int, the max index of the iterator

    method
    ----
    * __len__: return the number of elements
    * __iter__: return the iterator
    * __next__: return the next element
    * read: read one element from disk with the logic of self.readfunc
    * write: write one element to disk with the logic of self.writefunc
    * format_path: format the path of the element
    * clear: clear all data of the dict and clear directory
    * _init_data_i_dir_map: init the _data_i_dir_map
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
            count += len(glob.glob(os.path.join(root, f'*{self.suffix}')))
        return count

    def _init_data_i_dir_map(self):
        print(f'init {self.directory} data_i_dir_map, this may take a while...')
        for root, dirs, files in os.walk(self.directory):
            paths = glob.glob(root + "/*" + self.suffix)
            for path in paths:
                appdir, file = os.path.split(os.path.relpath(path, self.directory))
                data_i = int(os.path.splitext(file)[0])
                self._data_i_dir_map[data_i] = appdir
                self._max_idx = max(self._max_idx, data_i)        

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
    def read(self, data_i, appdir = "", appname = "", **kw):
        '''
        parameter
        ----
        * data_i: int, the index of the data
        * appdir: str, the sub directory of the root directory
        * appname: str, the string to be added to the file name(before the suffix)
        '''
        path = self.format_path(data_i, appdir=appdir, appname=appname)
        if not os.path.exists(path):
            path = self.auto_path(data_i)
        if os.path.exists(path):
            return self.read_func(path, **kw)
        else:
            return None

    @_DataCluster.cluster_closed_decorator(True)
    def write(self, data_i, element, appdir = "", appname = "", **kw):
        '''
        parameter
        ----
        * data_i: int, the index of the data
        * element: the element to be written
        * appdir: str, the sub directory of the root directory
        * appname: str, the string to be added to the file name(before the suffix)
        '''
        path = self.format_path(data_i, appdir=appdir, appname=appname)
        dir_ = os.path.split(path)[0]
        os.makedirs(dir_, exist_ok=True)
        if element is not None:
            self.write_func(path, element, **kw)

    def open(self):
        if not os.path.exists(self.directory):
            print(f"Elements: {self.directory} is new, it will be created")
            os.makedirs(self.directory, exist_ok=True)
        return super().open()

    def format_base_name(self, data_i):
        return "{}".format(str(data_i).rjust(self.filllen, "0"))

    def format_path(self, data_i, appdir = "", appname = ""):
        '''
        format the path of data_i
        '''
        return os.path.join(self.directory, appdir, 
                            "{}{}{}".format(
                                self.format_base_name(data_i), 
                                appname, 
                                self.suffix))

    def auto_path(self, data_i):
        '''
        TODO: default to match the first file that matches the data_i
        '''
        if self._updated or len(self._data_i_dir_map) == 0:
            self._init_data_i_dir_map()
        appdir = self._data_i_dir_map[data_i]
        path = os.path.join(self.directory, appdir, self.format_base_name(data_i) + self.suffix)
        if os.path.exists(path):
            return path
        else:
            pattern = os.path.join(self.directory, appdir, self.format_base_name(data_i)) + '*' + self.suffix
            paths = glob.glob(pattern) #TODO: default to match the first file that matches the data_i
            if len(paths) == 0:
                return None
            else:
                return paths[0]

    @_DataCluster.cluster_closed_decorator(True)
    def clear(self, ignore_warning = False):
        '''
        brief
        -----
        clear all data and remove the directory, 
        defalut to ask before executing
        '''
        if super().clear(ignore_warning):
            shutil.rmtree(self.directory)
            os.makedirs(self.directory)
            return True
        else:
            return False

class CacheElements(Elements):
    '''
    cache viewmeta as npy
    ----
    cache viewmeta as npy to speed up the loading process by about 4 times, \n
    At the same time, the space occupied by the file will increase by about 600%
    
    '''
    def __init__(self, format_obj: "DatasetFormat", sub_dir, filllen=6, fillchar='0') -> None:
        super().__init__(format_obj, sub_dir, False, None, None, ".npy", filllen, fillchar)
        self.read_func = self._read_func
        self.write_func = self._write_func

        self.cvtMask_uint8 = False
    
    def _init_attr(self, *args, **kwargs):
        super()._init_attr(*args, **kwargs)
        self.read_func = self._read_func
        self.write_func = self._write_func

    def _get_from_meta(self, meta:dict, name):
        '''
        get value from meta dict
        '''
        value = meta[name]
        return value

    def _read_func(self, path):
        meta = np.load(path,  allow_pickle= True).item()
        ids         = self._get_from_meta(meta, 'ids')
        color         = self._get_from_meta(meta, 'color')
        depth       = self._get_from_meta(meta, 'depth')
        mask_dict   = self._decompress_mask(ids, self._get_from_meta(meta, 'cprsd_mask'), self.cvtMask_uint8)
        extr_vecs   = self._zip_dict(ids, self._get_from_meta(meta, 'extr_vecs'))
        intr        = self._get_from_meta(meta, 'intr')
        depth_scale = self._get_from_meta(meta, 'depth_scale')
        bbox_3d     = self._zip_dict(ids, self._get_from_meta(meta, 'bbox_3d'))
        landmarks   = self._zip_dict(ids, self._get_from_meta(meta, 'landmarks'))
        visib_fract = self._zip_dict(ids, self._get_from_meta(meta, 'visib_fract'))

        viewmeta = ViewMeta(color, depth, mask_dict, extr_vecs, intr, depth_scale, bbox_3d, landmarks, visib_fract)
        return viewmeta

    def _write_func(self, path, viewmeta:ViewMeta):
        color = viewmeta.color
        depth = viewmeta.depth
        masks = viewmeta.masks
        ids, cprsd_mask = self._compress_mask(masks)
        ids, extr_vecs = self._split_dict(viewmeta.extr_vecs)
        intr = viewmeta.intr
        depth_scale = viewmeta.depth_scale
        ids, bbox_3d = self._split_dict(viewmeta.bbox_3d)
        ids, landmarks = self._split_dict(viewmeta.landmarks)
        ids, visib_fract = self._split_dict(viewmeta.visib_fract)
        np.save(path, 
                 {"ids":ids, 
                 "color":color, 
                 "depth":depth, 
                 "cprsd_mask":cprsd_mask, 
                 "extr_vecs":extr_vecs, "intr":intr, 
                 "depth_scale":depth_scale, "bbox_3d":bbox_3d, 
                 "landmarks":landmarks, "visib_fract":visib_fract})
        
    @staticmethod
    def _zip_dict(ids:np.ndarray, array:np.ndarray) -> dict[int, np.ndarray]:
        '''
        generate a dict from ids and array
        '''
        if array is None:
            return None
        dict_ = dict(zip(ids, array))
        return dict_

    @staticmethod
    def _split_dict(dict_:dict[int, np.ndarray]):
        '''
        split a dict into ids and array
        '''
        if dict_ is None:
            return None, None
        return np.array(list(dict_.keys())), np.array(list(dict_.values()))

    @staticmethod
    def _compress_mask(mask_dict: dict[int, np.ndarray]):
        '''
        compress mask dict into ids and cprsd_mask \n
        a serial of mask will be compressed into a single array by bit operation
        '''
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
    def _decompress_mask(ids:np.ndarray, masks:np.ndarray, cvtMask_uint8):
        '''
        decompress ids and cprsd_mask into mask dict \n
        a single array will be decompressed into a serial of mask by bit operation
        '''
        if masks is None:
            return None
        mask_dict = {}
        for i, id in enumerate(ids):
            mask = (masks & (1 << i)).astype(np.bool8)
            if cvtMask_uint8:
                mask = mask.astype(np.uint8) * 255
            mask_dict[id] = mask
        return mask_dict

class IntArrayDictElement(Elements):
    def __init__(self, format_obj: "DatasetFormat", sub_dir:str, array_shape:tuple[int], array_fmt:str = "", register=True, filllen=6, fillchar='0') -> None:
        super().__init__(format_obj, sub_dir, register, self._read_func, self._write_func, ".txt", filllen, fillchar)
        self.array_shape:tuple[int] = array_shape
        self.array_fmt = array_fmt if array_fmt else "%.4f"
    
    def _to_dict(self, array:np.ndarray)->dict[int, np.ndarray]:
        '''
        array: np.ndarray [N, 5]
        '''
        dict_ = {}
        for i in range(array.shape[0]):
            dict_[int(array[i, 0])] = array[i, 1:].reshape(self.array_shape)
        return dict_

    def _from_dict(self, dict_:dict[int, np.ndarray]):
        '''
        dict_: dict[int, np.ndarray]
        '''
        array = []
        for i, (k, v) in enumerate(dict_.items()):
            array.append(
                np.concatenate([np.array([k]).astype(v.dtype), v.reshape(-1)])
                )
        array = np.stack(array)
        return array

    def _read_format(self, array:np.ndarray, **kw):
        return array

    def _write_format(self, array:np.ndarray, **kw):
        return array

    def _read_func(self, path, **kw):
        raw_array = np.loadtxt(path, dtype=np.float32)
        if len(raw_array.shape) == 1:
            raw_array = np.expand_dims(raw_array, 0)
        raw_array = self._read_format(raw_array, **kw)
        intarraydict = self._to_dict(raw_array)
        return intarraydict

    def _write_func(self, path, intarraydict:dict[int, np.ndarray], **kw):
        raw_array = self._from_dict(intarraydict)
        raw_array = self._write_format(raw_array, **kw)
        np.savetxt(path, raw_array, fmt=self.array_fmt, delimiter='\t')

    def read(self, data_i, appdir="", appname="", **kw) -> dict[int, np.ndarray]:
        return super().read(data_i, appdir, appname, **kw)
    
    def write(self, data_i, raw_array_dict:dict[int, np.ndarray], appdir="", appname="", **kw):
        return super().write(data_i, raw_array_dict, appdir, appname, **kw)
        
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
    * _init_clusters
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
            self.format_obj.close_all(False)
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
        os.makedirs(self.directory, exist_ok=True)
        if not init_mode == DatasetFormatMode.ONLY_CACHE:
            self.incomplete = os.path.exists(self._Writer.mark_file(self.directory))
            if self.incomplete:
                if clear_incomplete:
                    pass
                else:
                    tip = f"the last writing process of the dataset:{self.directory} has not been completed, \
                        if you want to clear all data, input 'y', else the program will exit: "
                    print("="*int(len(tip) / 2), '\n', tip, '\n', "="*int(len(tip) / 2))
                    y = input()
                    if y != 'y':
                        raise ValueError("the dataset is incomplete")
            self.writer:DatasetFormat._Writer = None       
            self.cover_write = True  
            self.stream_dumping_json = True

            self.jsons:dict[str, JsonDict] = {}
            self.elements:dict[str, Elements] = {}
            self._streams:dict[str, JsonIO.Stream] = {}

            self._init_clusters()

            self._updata_data_num()

            if self.incomplete:
                os.remove(self._Writer.mark_file(self.directory))
                self.incomplete = False
        self.cache_elements = CacheElements(self, "cache")

    def _init_clusters(self):
        self.labels_elements     = IntArrayDictElement(self, "labels", (4,), array_fmt="%8.8f")
        self.bbox_3ds_elements   = IntArrayDictElement(self, "bbox_3ds", (-1, 2), array_fmt="%8.8f") 
        self.landmarks_elements  = IntArrayDictElement(self, "landmarks", (-1, 2), array_fmt="%8.8f")
        self.extr_vecs_elements  = IntArrayDictElement(self, "trans_vecs", (2, 3), array_fmt="%8.8f")
        # self.scene_camera_info          = BaseJsonDict(self, "scene_camera.json")
        # self.scene_visib_fract_info     = BaseJsonDict(self, "scene_visib_fract.json")
        # self.scene_bbox_3d_info         = BaseJsonDict(self, 'scene_bbox_3d.json')   
        # self.scene_landmarks_info       = BaseJsonDict(self, 'scene_landmarks.json')   
        # self.scene_trans_vector_info    = BaseJsonDict(self, 'scene_trans_vector.json')

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
    def read_one(self, data_i, appdir = "", appname = "")->ViewMeta:
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
        #
        labels_dict:dict[int, np.ndarray] = self.labels_elements.read(data_i, appdir=appdir)
        extr_vecs_dict:dict[int, np.ndarray] = self.extr_vecs_elements.read(data_i, appdir=appdir)
        bbox_3d_dict:dict[int, np.ndarray] = self.bbox_3ds_elements.read(data_i, appdir=appdir)
        landmarks_dict:dict[int, np.ndarray] = self.landmarks_elements.read(data_i, appdir=appdir)
        return ViewMeta(color=None,
                        depth=None,
                        masks=None,
                        extr_vecs = extr_vecs_dict,
                        intr=None,
                        depth_scale=None,
                        bbox_3d = bbox_3d_dict, 
                        landmarks = landmarks_dict,
                        visib_fract=None,
                        labels=labels_dict)

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
            for path in self.jsons.keys():
                JsonDict(self, path)
        else:
            for file, value in self.jsons.items():
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
            self._updata_data_num()        
            data_i = self.data_num
        
        self.write_element(viewmeta, data_i)
        parse = self._parse_viewmeta_for_jsons(viewmeta, data_i)
        self._cache_source_info(parse)
        self._updata_data_num()        

    def write_element(self, viewmeta:ViewMeta, data_i:int, appdir = "", appname = ""):
        self.labels_elements.write(data_i, viewmeta.labels, appdir=appdir, appname=appname)
        self.bbox_3ds_elements.write(data_i, viewmeta.bbox_3d, appdir=appdir, appname=appname)
        self.landmarks_elements.write(data_i, viewmeta.landmarks, appdir=appdir, appname=appname)
        self.extr_vecs_elements.write(data_i, viewmeta.extr_vecs, appdir=appdir, appname=appname)

    def _parse_viewmeta_for_jsons(self, viewmeta:ViewMeta, data_i):
        # parse = {
        #     self.scene_camera_info.directory: {data_i: {LinemodFormat.KW_CAM_K: viewmeta.intr.reshape(-1).tolist(),
        #                                                       LinemodFormat.KW_CAM_DS: float(viewmeta.depth_scale),
        #                                                       LinemodFormat.KW_CAM_VL: 1}},
        #     self.scene_visib_fract_info.directory: {data_i: viewmeta.visib_fract},
        #     #
        #     self.scene_bbox_3d_info.directory: {data_i: viewmeta.bbox_3d},
        #     #
        #     self.scene_landmarks_info.directory: {data_i: viewmeta.landmarks},
        #     #
        #     self.scene_trans_vector_info.directory: {data_i: viewmeta.extr_vecs},
        # }
        parse = {}
        return parse

    def _cache_source_info(self, parsed:dict):
        for path in self.jsons.keys():
            if self.stream_dumping_json and not self.jsons[path].is_close() and not self.jsons[path].is_read_only():
                if path not in self._streams:
                    self._streams[path] = JsonIO.create_stream(path)                
                self._streams[path].write(parsed[path])

                keys = parsed[path].keys()
                self.jsons[path].update(dict(zip(keys, [None for _ in keys])))
            else:
                self.jsons[path].update(parsed[path])

    def _updata_data_num(self):
        '''
        The total number of data of all types must be equal, otherwise an exception is thrown
        '''
        datas = list(self.jsons.values()) + list(self.elements.values())
        datas = [x for x in datas if not x.is_close()]
        nums = [len(x) for x in datas]
        num = np.unique(nums)
        if len(num) > 1:
            raise ValueError("Unknown error, the numbers of different datas are not equal")
        elif len(num) == 1:
            self.data_num = int(num)
        else:
            self.data_num = 0

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
            for path in list(self.elements.keys()) + list(self.jsons.keys()):
                if os.path.exists(path):
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                        os.makedirs(path)
                    else:
                        os.remove(path)
            [x.clear() for x in self.jsons.values()]
        self._updata_data_num()

    def close_all(self, value = True):
        for obj in list(self.elements.values()) + list(self.jsons.values()):
            obj.close() if value else obj.open()

    def set_all_read_only(self, value = True):
        for obj in list(self.elements.values()) + list(self.jsons.values()):
            obj.set_read_only(value)

    def get_element_paths_of_one(self, data_i:int):
        '''
        brief
        -----
        get all paths of a data
        '''
        paths = {}
        for elem in self.elements.values():
            paths[elem.sub_dir] = elem.auto_path(data_i)
        return paths

class LinemodFormat(DatasetFormat):
    KW_CAM_K = "cam_K"
    KW_CAM_DS = "depth_scale"
    KW_CAM_VL = "view_level"
    KW_GT_R = "cam_R_m2c"
    KW_GT_t = "cam_t_m2c"
    KW_GT_ID = "obj_id"
    KW_GT_INFO_BBOX_OBJ = "bbox_obj"
    KW_GT_INFO_BBOX_VIS = "bbox_visib"
    KW_GT_INFO_PX_COUNT_ALL = "px_count_all"
    KW_GT_INFO_PX_COUNT_VLD = "px_count_valid"
    KW_GT_INFO_PX_COUNT_VIS = "px_count_visib" 
    KW_GT_INFO_VISIB_FRACT = "visib_fract"

    RGB_DIR = "rgb"
    DEPTH_DIR = "depth"
    MASK_DIR = "mask"
    GT_FILE = "scene_gt.json"
    GT_CAM_FILE = "scene_camera.json"
    GT_INFO_FILE = "scene_gt_info.json"
    
    class _MasksElements(Elements):
        def id_format(self, class_id):
            id_format = "_" + str(class_id).rjust(6, "0")
            return id_format

        def read(self, data_i):
            masks = {}
            obj:LinemodFormat = self.format_obj
            for n, scene_gt in enumerate(obj.scene_gt_dict[data_i]):
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
        self.rgb_elements   = Elements(self,      self.RGB_DIR,
                                       read_func=cv2.imread,                                    
                                       write_func=cv2.imwrite, 
                                       suffix='.png')
        self.depth_elements = Elements(self,      self.DEPTH_DIR,    
                                       read_func=lambda x:cv2.imread(x, cv2.IMREAD_ANYDEPTH),   
                                       write_func=cv2.imwrite, 
                                       suffix='.png')
        self.masks_elements = self._MasksElements(self, self.MASK_DIR,     
                                                  read_func=lambda x:cv2.imread(x, cv2.IMREAD_GRAYSCALE),  
                                                  write_func=cv2.imwrite, 
                                                  suffix='.png')      

        self.scene_gt_dict              = JsonDict(self, self.GT_FILE)        
        self.scene_camera_dict          = JsonDict(self, self.GT_CAM_FILE)
        self.scene_gt_info_dict         = JsonDict(self, self.GT_INFO_FILE)

    def write_element(self, viewmeta:ViewMeta, data_i:int):
        super().write_element(viewmeta, data_i)
        self.rgb_elements.  write(data_i, viewmeta.color)
        self.depth_elements.write(data_i, viewmeta.depth)
        self.masks_elements.write(data_i, viewmeta.masks)

    def _parse_viewmeta_for_jsons(self, viewmeta: ViewMeta, data_i):
        parsed = super()._parse_viewmeta_for_jsons(viewmeta, data_i)
        gt_one_info = []
        for obj_id, trans_vecs in viewmeta.extr_vecs.items():
            posture = Posture(rvec=trans_vecs[0], tvec=trans_vecs[1])
            gt_one_info .append(
                {   LinemodFormat.KW_GT_R: posture.rmat.reshape(-1).tolist(),
                    LinemodFormat.KW_GT_t: posture.tvec.reshape(-1).tolist(),
                    LinemodFormat.KW_GT_ID: int(obj_id)})
        parsed.update({self.scene_gt_dict.directory: {self.data_num: gt_one_info}})

        ###
        gt_cam_one_info = {self.KW_CAM_K: viewmeta.intr.reshape(-1).tolist(), self.KW_CAM_DS: viewmeta.depth_scale, self.KW_CAM_VL: 1}
        parsed.update({self.scene_camera_dict.directory: {self.data_num: gt_cam_one_info}})

        ### eg:
        # "0": 
        # [{"bbox_obj": [274, 188, 99, 106], 
        # "bbox_visib": [274, 188, 99, 106], 
        # "px_count_all": 7067, 
        # "px_count_valid": 7067, 
        # "px_count_visib": 7067, 
        # "visib_fract": 1.0}],
        gt_info_one_info = []
        bbox_2d = viewmeta.bbox_2d
        for obj_id in viewmeta.masks.keys():
            mask = viewmeta.masks[obj_id]
            bb = bbox_2d[obj_id].tolist()
            vf = viewmeta.visib_fract[obj_id]
            mask_count = int(np.sum(mask))
            mask_visib_count = int(mask_count * vf)
            gt_info_one_info.append({
                self.KW_GT_INFO_BBOX_OBJ: bb,
                self.KW_GT_INFO_BBOX_VIS: bb,
                self.KW_GT_INFO_PX_COUNT_ALL: mask_count, 
                self.KW_GT_INFO_PX_COUNT_VLD: mask_count, 
                self.KW_GT_INFO_PX_COUNT_VIS: mask_visib_count,
                self.KW_GT_INFO_VISIB_FRACT: vf
            })
        parsed.update({self.scene_gt_info_dict.directory: {self.data_num: gt_info_one_info} })

        return parsed    

    def __parse_one_scene_gt_info(self, data_i):
        one_scene_gt_info:list[dict[str, np.ndarray]] = self.scene_gt_dict[data_i]
        extr = {}
        for info in one_scene_gt_info:
            posture = Posture(rmat=info[LinemodFormat.KW_GT_R].reshape(3,3), 
                    tvec=info[LinemodFormat.KW_GT_t])
            extr[info[LinemodFormat.KW_GT_ID]] = np.array([posture.rvec, posture.tvec])
        return extr

    def read_one(self, data_i):
        super().read_one(data_i)
        color     = self.rgb_elements.read(data_i)
        depth   = self.depth_elements.read(data_i)
        masks   = self.masks_elements.read(data_i)
        bbox_3d = self.bbox_3ds_elements.read(data_i)
        landmarks = self.landmarks_elements.read(data_i)
        intr           = self.scene_camera_dict[data_i][LinemodFormat.KW_CAM_K].reshape(3, 3)
        depth_scale    = self.scene_camera_dict[data_i][LinemodFormat.KW_CAM_DS]

        ids = [x[LinemodFormat.KW_GT_ID] for x in self.scene_gt_dict[data_i]]
        postures = [Posture(rmat =x[LinemodFormat.KW_GT_R], tvec=x[LinemodFormat.KW_GT_t]) for x in self.scene_gt_dict[data_i]]
        extr_vecs = [np.array([x.rvec, x.tvec]) for x in postures]
        extr_vecs_dict = as_dict(ids, extr_vecs)
        visib_fract    = [x[LinemodFormat.KW_GT_INFO_VISIB_FRACT] for x in self.scene_gt_info_dict[data_i]]
        visib_fract_dict = as_dict(ids, visib_fract)
        return ViewMeta(color, depth, masks, 
                        extr_vecs_dict,
                        intr,
                        depth_scale,
                        bbox_3d,
                        landmarks,
                        visib_fract_dict)

class VocFormat(DatasetFormat):
    KW_TRAIN = "train"
    KW_VAL = "val"

    class _Writer(DatasetFormat._Writer):
        def __init__(self, format_obj: "VocFormat", cover=False) -> None:
            super().__init__(format_obj, cover)
            self.format_obj: VocFormat = format_obj

        def __exit__(self, exc_type, exc_value: Exception, traceback):
            exit_return = super().__exit__(exc_type, exc_value, traceback)
            np.savetxt(self.format_obj.train_txt,   self.format_obj.train_idx_array,    fmt = "%6d")
            np.savetxt(self.format_obj.val_txt,     self.format_obj.val_idx_array,      fmt = "%6d")    
            return exit_return    

    class cxcywhLabelElement(IntArrayDictElement):
        def __init__(self, format_obj: DatasetFormat, sub_dir: str, array_fmt: str = "", register=True, filllen=6, fillchar='0') -> None:
            super().__init__(format_obj, sub_dir, (4,), array_fmt, register, filllen=filllen, fillchar=fillchar)

        def _read_format(self, labels: np.ndarray, image_size):
            bbox_2d = labels[:,1:] #[cx, cy, w, h]
            bbox_2d = VocFormat._normedcxcywh_2_x1y1x2y2(bbox_2d, image_size)
            labels[:,1:] = bbox_2d   
            return labels         
        
        def _write_format(self, labels: np.ndarray, image_size):
            bbox_2d = labels[:,1:] #[cx, cy, w, h]
            bbox_2d = VocFormat._x1y1x2y2_2_normedcxcywh(bbox_2d, image_size)
            labels[:,1:] = bbox_2d
            return labels

        def read(self, data_i, appdir="", appname="", image_size = None)->dict[int, np.ndarray]:
            '''
            set_image_size() is supported be called before read() \n
            or the bbox_2d will not be converted from normed cxcywh to x1x2y1y2
            '''
            if image_size is not None:
                return super().read(data_i, appdir, appname, image_size = image_size)
            else:
                warnings.warn("image_size is None, bbox_2d will not be converted from normed cxcywh to x1x2y1y2")
                return None
        
        def write(self, data_i, labels_dict:dict[int, np.ndarray], appdir="", appname="", image_size = None):
            '''
            set_image_size() is supported be called before write() \n
            or the bbox_2d will not be converted from x1x2y1y2 to normed cxcywh
            '''
            if image_size is not None:
                return super().write(data_i, labels_dict, appdir, appname, image_size = image_size)
            else:
                warnings.warn("image_size is None, bbox_2d will not be converted from x1x2y1y2 to normed cxcywh")

    def __init__(self, directory, data_num = 0, split_rate = 0.75, clear = False) -> None:
        super().__init__(directory, clear)

        self.train_txt = os.path.join(self.directory,   VocFormat.KW_TRAIN + ".txt")
        self.val_txt = os.path.join(self.directory,     VocFormat.KW_VAL + ".txt")
        self.split_rate = split_rate
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
        self.intr_elements       = Elements(self, "intr",
                                            read_func=loadtxt_func((3,3)), 
                                            write_func=savetxt_func("%8.8f"), 
                                            suffix = ".txt")
        self.depth_scale_elements         = Elements(self, "depth_scale",
                                            read_func=lambda path: float(loadtxt_func((1,))(path)), 
                                            write_func=savetxt_func("%8.8f"), 
                                            suffix = ".txt")
        self.visib_fract_elements= IntArrayDictElement(self, "visib_fracts", (1,))
        self.labels_elements     = self.cxcywhLabelElement(self, "labels",)
                                            
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
            self.train_idx_array = np.array(data_i_list[: int(data_num*split_rate)]).astype(np.int32).reshape(-1)
            self.val_idx_array   = np.array(data_i_list[int(data_num*split_rate): ]).astype(np.int32).reshape(-1)
            np.savetxt(self.train_txt, self.train_idx_array, fmt = "%6d")
            np.savetxt(self.val_txt, self.val_idx_array, fmt = "%6d")

    def decide_set(self, data_i, create_if_not_exist = False):
        if data_i in self.train_idx_array:
            sub_set = VocFormat.KW_TRAIN
        elif data_i in self.val_idx_array:
            sub_set = VocFormat.KW_VAL
        else:
            if create_if_not_exist:
                train_num = len(self.train_idx_array)
                val_num = len(self.val_idx_array)
                if train_num/self.split_rate > val_num/(1 - self.split_rate):
                    sub_set = VocFormat.KW_VAL
                    self.val_idx_array = np.append(self.val_idx_array, data_i)
                else:
                    sub_set = VocFormat.KW_TRAIN
                    self.train_idx_array = np.append(self.train_idx_array, data_i)
            else:
                raise ValueError("can't find datas of index: {}".format(data_i))
        return sub_set

    @staticmethod
    def _x1y1x2y2_2_normedcxcywh(bbox_2d, img_size):
        '''
        bbox_2d: np.ndarray [..., (x1, x2, y1, y2)]
        img_size: (h, w)
        '''

        # Calculate center coordinates (cx, cy) and width-height (w, h) of the bounding boxes
        x1, y1, x2, y2 = np.split(bbox_2d, 4, axis=-1)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        # Normalize center coordinates and width-height by image size
        h_img, w_img = img_size
        cx_normed = cx / w_img
        cy_normed = cy / h_img
        w_normed = w / w_img
        h_normed = h / h_img

        # Return the normalized bounding boxes as a new np.ndarray with shape (..., 4)
        bbox_normed = np.concatenate([cx_normed, cy_normed, w_normed, h_normed], axis=-1)
        return bbox_normed

        # lt = bbox_2d[..., :2]
        # rb = bbox_2d[..., 2:]

        # cx, cy = (lt + rb) / 2
        # w, h = rb - lt
        # # 归一化
        # cy, h = np.array([cy, h]) / img_size[0]
        # cx, w = np.array([cx, w]) / img_size[1]
        # return np.array([cx, cy, w, h])

    @staticmethod
    def _normedcxcywh_2_x1y1x2y2(bbox_2d, img_size):
        '''
        bbox_2d: np.ndarray [..., (cx, cy, w, h)]
        img_size: (h, w)
        '''

        # Unpack the normalized bounding box coordinates
        cx, cy, w, h = np.split(bbox_2d, 4, axis=-1)

        # Denormalize the center coordinates and width-height by image size
        h_img, w_img = img_size
        x1 = (cx - w / 2) * w_img
        y1 = (cy - h / 2) * h_img
        x2 = x1 + w * w_img
        y2 = y1 + h * h_img

        # Return the bounding boxes as a new np.ndarray with shape (..., 4)
        bbox_2d = np.concatenate([x1, y1, x2, y2], axis=-1)
        return bbox_2d

    def write_element(self, viewmeta: ViewMeta, data_i: int):
        sub_set = self.decide_set(data_i, create_if_not_exist = True)
        super().write_element(viewmeta, data_i, appdir=sub_set)
        #
        self.images_elements.write(data_i, viewmeta.color, appdir=sub_set)
        #
        self.depth_elements.write(data_i, viewmeta.depth, appdir=sub_set)
        #
        self.masks_elements.write(data_i, viewmeta.masks, appdir=sub_set)
        
        ###
        self.labels_elements.write(data_i, viewmeta.bbox_2d, appdir=sub_set, image_size = viewmeta.color.shape[:2]) # necessary to set image_size
        # labels = []
        # for id_, mask in viewmeta.masks.items():
        #     img_size = mask.shape
        #     point = np.array(np.where(mask))
        #     if point.size == 0:
        #         continue
        #     bbox_2d = viewmeta_bbox2d[id_]
        #     cx, cy, w, h = self._x1y1x2y2_2_normedcxcywh(bbox_2d, img_size)
        #     labels.append([id_, cx, cy, w, h])

        # self.labels_elements.write(data_i, labels, appdir=sub_set)

        self.intr_elements.write(data_i, viewmeta.intr, appdir=sub_set)
        self.depth_scale_elements.write(data_i, np.array([viewmeta.depth_scale]), appdir=sub_set)
        self.visib_fract_elements.write(data_i, viewmeta.visib_fract, appdir=sub_set)
    
    def read_one(self, data_i) -> ViewMeta:
        # 判断data_i属于train或者val
        sub_set = self.decide_set(data_i)
        viewmeta = super().read_one(data_i, appdir=sub_set)
        
        # 读取
        color:np.ndarray = self.images_elements.read(data_i, appdir=sub_set)
        viewmeta.set_element(ViewMeta.COLOR, color)
        #
        depth = self.depth_elements.read(data_i, appdir=sub_set)
        viewmeta.set_element(ViewMeta.DEPTH, depth)
        #
        labels_dict = self.labels_elements.read(data_i, appdir=sub_set, image_size = color.shape[:2]) # {id: [cx, cy, w, h]}
        viewmeta.set_element(ViewMeta.LABELS, labels_dict)
        #
        masks_dict = self.masks_elements.read(data_i, appdir=sub_set)
        viewmeta.set_element(ViewMeta.MASKS, masks_dict)
        #
        intr = self.intr_elements.read(data_i, appdir=sub_set)
        viewmeta.set_element(ViewMeta.INTR, intr)
        #
        ds    = self.depth_scale_elements.read(data_i, appdir=sub_set)
        viewmeta.set_element(ViewMeta.DEPTH_SCALE, ds)
        #
        visib_fract_dict = self.visib_fract_elements.read(data_i, appdir=sub_set)
        viewmeta.set_element(ViewMeta.VISIB_FRACT, visib_fract_dict)

        return viewmeta

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
