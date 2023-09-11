# from compute_gt_poses import GtPostureComputer

# from toolfunc import *
from _collections_abc import dict_items, dict_keys, dict_values
from collections.abc import Iterator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from open3d import geometry, utility, io
import os
import glob
import shutil
import pickle
import cv2
import time
from tqdm import tqdm
import warnings

from abc import ABC, abstractmethod
from typing import Any, Union, Callable, TypeVar, Generic, Iterable

from . import Posture, JsonIO, JSONDecodeError, extract_doc
from .viewmeta import ViewMeta, serialize_image_container, deserialize_image_container
from .mesh_manager import MeshMeta

DCT  = TypeVar('DCT',  bound="_DataCluster") # type of the value of data cluster
DSNT = TypeVar('DSNT', bound='DatasetNode') # dataset node type
NDT  = TypeVar("NDT", bound="Node")
VDCT = TypeVar('VDCT') # type of the value of data cluster
VDST = TypeVar('VDST') # type of the value of dataset
from numpy import ndarray

def as_dict(ids, objs):
    if objs is None:
        return None
    else:
        return dict(zip(ids, objs))

def savetxt_func(fmt=...):
    return lambda path, x: np.savetxt(path, x, fmt=fmt, delimiter='\t')

def loadtxt_func(shape:tuple[int]):
    return lambda path: np.loadtxt(path).reshape(shape)

class WriteController(ABC):
    '''
    control the write operation.
    the subclass of WriteController must implement :
    * start_writing
    * stop_writing
    '''
    WRITING_MARK = '.writing'

    LOG_APPEND = 0
    LOG_REMOVE = 1
    LOG_CHANGE = 2
    class __Writer():
        def __init__(self, writecontroller:"WriteController") -> None:
            self.writecontroller = writecontroller
            self.__overwrite_allowed = False

        def allow_overwriting(self):
            self.__overwrite_allowed = True
            return self

        def __enter__(self):
            self.writecontroller.start_writing(self.__overwrite_allowed)
            return self
        
        def __exit__(self, exc_type, exc_value, traceback):
            if exc_type is not None:
                raise exc_type(exc_value).with_traceback(traceback)
            else:
                self.writecontroller.stop_writing()
                self.__overwrite_allowed = False
                return True
    
    def __init__(self) -> None:
        self.__writer = self.__Writer(self)
        self.is_writing = False

    @property
    def writer(self):
        return self.__writer

    @abstractmethod
    def get_writing_mark_file(self):
        pass

    def start_writing(self, overwrite_allowed = False):
        if self.is_writing:
            return False
        else:
            self.is_writing = True
            with open(self.get_writing_mark_file(), 'w'):
                pass
            return True

    def stop_writing(self):
        if self.is_writing:
            os.remove(self.get_writing_mark_file())
            self.is_writing = False
            return True
        else:
            return False

    def mark_exist(self):
        return os.path.exists(self.get_writing_mark_file())

    def load_from_mark_file(self):
        file_path = self.get_writing_mark_file()
        if os.path.exists(file_path):
            result_dict = {}
            with open(file_path, 'r') as file:
                for line in file:
                    # 使用strip()函数移除行末尾的换行符，并使用split()函数分割列
                    columns = line.strip().split(', ')
                    assert len(columns) == 2, f"the format of {file_path} is wrong"
                    key, value_str = columns
                    # 尝试将第二列的值转换为整数
                    value = int(value_str)
                    assert value in [self.LOG_APPEND, self.LOG_REMOVE, self.LOG_CHANGE], f"the format of {file_path} is wrong"
                    try:
                        key = str(key)
                    except ValueError:
                        pass
                    result_dict[key] = value
            return result_dict
        else:
            return None

    def log_to_mark_file(self, key, value):
        if key is None:
            return 
        assert isinstance(key, (str, int)), f"key must be str or int, not {type(key)}"
        assert isinstance(value, int), f"value must be int, not {type(value)}"
        assert value in [self.LOG_APPEND, self.LOG_REMOVE, self.LOG_CHANGE], f"value must be in {self.LOG_APPEND, self.LOG_REMOVE, self.LOG_CHANGE}"
        file_path = self.get_writing_mark_file()
        with open(file_path, 'a') as file:
            line = f"{key}, {value}\n"
            file.write(line)

    # @abstractmethod
    def _rollback_one(self, key, value):
        raise NotImplementedError

    def rollback(self):
        result_dict = self.load_from_mark_file()
        if result_dict:
            if any([v != self.LOG_APPEND for v in result_dict.values()]):
                # any value is not LOG_APPEND, can't rollback
                raise ValueError(f"can't rollback {self.get_writing_mark_file()}, because there are some irreversible changing")
            else:
                for k, v in result_dict.items():
                    self._rollback_one(k, v)

#### Warning Types ####
class ClusterDataIOError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class ClusterWarning(Warning):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class ClusterParaWarning(ClusterWarning):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class ClusterIONotExecutedWarning(ClusterWarning):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class ClusterNotRecommendWarning(ClusterWarning):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class InstanceRegistry(ABC):
    _cluster_registry = {}
    _dataset_registry = {}
    _registry:dict = None

    def __init_subclass__(cls, **kw) -> None:
        if cls.__name__ == "_DataCluster" or ("_DataCluster" in globals() and issubclass(cls, _DataCluster)):
            cls._registry = cls._cluster_registry
        elif cls.__name__ == "DatasetNode" or ("DatasetNode" in globals() and issubclass(cls, DatasetNode)):
            cls._registry = cls._dataset_registry
        else:
            raise TypeError(f"invalid subclass {cls}")
        cls._org_init__ = cls.__init__

        def decorated_init(obj:InstanceRegistry, *args, **kwargs):
            if obj in obj._registry.values():
                try:    obj._InstanceRegistry_inited
                except: return cls._org_init__(obj, *args, **kwargs)
            else:
                return cls._org_init__(obj, *args, **kwargs)
            
        cls.__init__ = decorated_init
        return super().__init_subclass__()

    def __new__(cls, *args, **kw):
        instance = super(InstanceRegistry, cls).__new__(cls)
        instance._init_identity_paramenter(*args, **kw)
        identity_string = instance.identity_string()
        if identity_string in cls._registry:
            return cls._registry[identity_string]
        # , dataset_node, sub_dir, register, name, *args, **kwargs)
        cls._registry[identity_string] = instance
        return instance
    
    def __init__(self) -> None:
        super().__init__()
        self._InstanceRegistry_inited = True

    @abstractmethod
    def _init_identity_paramenter(self):
        pass

    @abstractmethod
    def identity_string(self):
        pass

    @classmethod
    @abstractmethod
    def gen_identity_string(cls, *args, **kw):
        pass

    @staticmethod
    @abstractmethod
    def parse_identity_string(identity_string):
        pass


#### _DataCluster #####
class _DataCluster(WriteController, InstanceRegistry, ABC, Generic[DSNT, VDCT]):
    '''
    This is a private class representing a data cluster used for managing datasets with a specific format.

    # attr
    ----
    * self.dataset_node: DatasetNode
    * self.closed: bool, Control the shielding of reading and writing, 
        if it is true, the instance will not write, and the read will get None
    * register: bool, whether to register to dataset_node
    * _incomplete: bool, whether the data is incomplete
    * _closed: bool, Indicates whether the cluster is closed or open.
    * _readonly: bool, Indicates whether the cluster is read-only or write-enabled.
    * changes_unsaved: bool, Indicates if any changes have been made to the cluster.
    * directory: str, Directory path for the cluster.


    # property
    -----
    * overwrite_allowed: bool, Control the shielding of writing,
    * cluster_data_num: int, the number of data in the cluster
    * cluster_data_i_upper: int, the upper of the iterator, it is the max index of the iterator + 1
    * changed_since_opening: bool, Indicates whether the cluster has been modified since last opening.

    # method
    -----
    abstract method:
    -----
    - __len__: return the number of data in the cluster
    - keys: return the keys of the cluster
    - values: return the values of the cluster
    - items: return the items of the cluster(key and value)
    - _read: read data from the cluster
    - _write: write data to the cluster
    - _clear: clear all data of the cluster
    - _copyto: copy the cluster to dst
    - _merge: merge the cluster to self, the cluster must has the same type as self

    recommend to implement:
    -----
    - _init_attr: initialize additional attributes specified by subclasses.
    - _update_cluster_inc: update the incremental modification of the cluster after writing
    - _update_cluster_all: update the state of the cluster after writing
    - __getitem__: return the value of the key
    - __setitem__: set the value of the key    
    - _open: operation when open the cluster
    - _close: operation when close the cluster
    - _start_writing: operation when start writing
    - _stop_writing: operation when stop writing
    - check_key: check if the key is valid

    not need to implement:
    -----
    - __iter__: return the iterator of the cluster
    - open: open the cluster for operation.
    - close: close the cluster, preventing further operations.
    - is_close: check if the cluster is closed.
    - set_readonly: set the cluster as read-only or write-enabled.
    - set_writable: set the cluster as writable or read-only.
    - set_overwrite_allowed: set the cluster as writable or read-only.
    - is_readonly: check if the cluster is read-only.
    - is_writeable: check if the cluster is writable.
    - is_overwrite_allowed: check if the cluster is writable.
    - _read_decorator: decorator function to handle reading operations when the cluster is closed.
    - _write_decorator: decorator function to handle writing operations when the cluster is closed or read-only.
    - clear: clear any data in the cluster. Subclasses may implement _clear.
    - read: read data from the cluster. Subclasses must implement _read.
    - write: write data to the cluster. Subclasses must implement _write.
    - copyto: copy the cluster to dst
    - merge: merge the cluster to self, the cluster must has the same type as self
    - start_writing: start writing
    - stop_writing: stop writing
    - __repr__: return the representation of the cluster
    - register_to_format: register the cluster to dataset_node
    '''
    def __init__(self, dataset_node:DSNT, sub_dir: str, register=True, name = "", *args, **kwargs) -> None:
        '''
        Initialize the data cluster with the provided dataset_node, sub_dir, and registration flag.
        '''
        WriteController.__init__(self)    
        InstanceRegistry.__init__(self)
        self._init_identity_paramenter(dataset_node, sub_dir, register, name, *args, **kwargs)
        self.dataset_node:DSNT = dataset_node
        self.sub_dir = os.path.normpath(sub_dir) 
        self.name = name       
        self.directory = os.path.normpath(os.path.join(self.dataset_node.directory, self.sub_dir))  # Directory path for the cluster.        
        self.register = register
        self._incomplete = self.mark_exist()
        self._incomplete_operation = self.dataset_node._incomplete_operation
        self._error_to_load = False
        self._changed = False  # Indicates if any changes have been made to the cluster. 
        self._data_i_upper = 0      

        self.__closed = True  # Indicates whether the cluster is closed or open.
        self.__readonly = True  # Indicates whether the cluster is read-only or write-enabled.
        self.__overwrite_allowed = False

        self.__cluster_read_func:Callable =  self._read_decorator(self.__class__._read)
        self.__cluster_write_func:Callable = self._write_decorator(self.__class__._write)
        self.__cluster_clear_func:Callable = self._write_decorator(self.__class__._clear) 

        self._init_attr(*args, **kwargs)  # Initializes additional attributes specified by subclasses.
     
        if os.path.exists(self.directory) and not self._error_to_load:
            self.open()  # Opens the cluster for operation.
        else:
            self.close()
  
        self.register_to_dataset()
    
    ### implement InstanceRegistry ###
    def _init_identity_paramenter(self, dataset_node:DSNT, sub_dir: str, register=True, name = "", *args, **kwargs):
        self.dataset_node = dataset_node
        self.sub_dir = sub_dir
        self.name = name
        self.directory = os.path.normpath(os.path.join(self.dataset_node.directory, self.sub_dir))

    @classmethod
    def gen_identity_string(cls, dataset_node:"DatasetNode", sub_dir, name, *arg, **kw):
        return f"{cls.__name__}({dataset_node.__class__.__name__}, {dataset_node.directory}, {sub_dir}, {name})"

    @staticmethod
    def parse_identity_string(identity_string):
        import re
        pattern = re.compile(r"(\w*)\((\w*), (.*), (.*), (.*)\)")
        match = pattern.match(identity_string)
        if match:
            cls_name, dataset_node_cls_name, dataset_node_dir, sub_dir, name = match.groups()
            return cls_name, dataset_node_cls_name, dataset_node_dir, sub_dir, name
        else:
            raise ValueError(f"invalid identity string {identity_string}")

    def identity_string(self):
        return self.gen_identity_string(self.dataset_node, self.sub_dir, self.name)
    ###

    ### implement WriteController ###
    def get_writing_mark_file(self):
        path = os.path.join(self.directory, self.name)
        if '.' in path:
            return path + self.WRITING_MARK
        else:
            return os.path.join(path, self.WRITING_MARK)
    ### 

    @property
    def overwrite_allowed(self):
        '''Property that returns whether the cluster format allows write operations.'''
        return self.__overwrite_allowed

    @property
    def cluster_data_num(self):
        return len(self)

    @property
    def cluster_data_i_upper(self):
        return max(self.keys()) + 1 if len(self) > 0 else 0

    @property
    def changed_since_opening(self):
        '''Indicates whether the cluster has been modified since last opening.'''
        return self._changed
    
    @changed_since_opening.setter
    def changed_since_opening(self, value:bool):
        self._changed = bool(value)
        self.dataset_node.updated = True

    @abstractmethod
    def __len__(self):
        '''Returns the number of data in the cluster.'''
        pass     

    @abstractmethod
    def keys(self) -> Iterable[Any]:
        pass

    @abstractmethod
    def values(self) -> Iterable[VDCT]:
        pass

    @abstractmethod
    def items(self) -> Iterable[tuple[Any, VDCT]]:
        pass

    @abstractmethod
    def _read(self, data_i, *arg, **kwargs) -> VDCT:
        pass

    @abstractmethod
    def _write(self, data_i, value:VDCT, *arg, **kwargs):
        pass

    @abstractmethod
    def _clear(self, *arg, **kwargs):
        pass

    @abstractmethod
    def _copyto(self, dst: str, *args, **kwargs):
        pass

    @abstractmethod
    def _merge_one(src, k, *args, **kwargs):
        pass

    def _init_attr(self, *args, **kwargs):
        '''Method to initialize additional attributes specified by subclasses.'''
        pass

    def _update_cluster_inc(self, data_i, *args, **kwargs):
        '''
        update the state of the cluster after writing
        '''
        pass

    def _update_cluster_all(self, *args, **kwargs):
        pass
   
    def __getitem__(self, data_i) -> VDCT:
        return self.read(data_i)
    
    def __setitem__(self, data_i, value:VDCT):
        return self.write(data_i, value)

    def __iter__(self) -> Iterable[VDCT]:
        return self.values()

    def choose_incomplete_operation(obj):
        if isinstance(obj, _DataCluster):
            tip_0 = "skip"
        elif isinstance(obj, DatasetFormat):
            tip_0 = "decide one by one"
        choice = int(input(f"please choose an operation to continue:\n\
                    0. {tip_0}\n\
                    1. clear the incomplete data\n\
                    2. try to rollback the incomplete data\n\
                    3. exit\n"))
        if choice not in [0, 1, 2, 3]:
            raise ValueError(f"invalid choice {choice}")
        return choice

    def _open(self):
        '''Method to open the cluster for operation.'''    
        return True

    def _close(self):
        return True

    def _start_writing(self):
        pass

    def _stop_writing(self):
        self._changed = False  # Resets the updated flag to false.
        self.__closed = True  # Marks the cluster as closed.

    def check_key(self, key) -> bool:
        return True

    def process_incomplete(self):
        if self._incomplete:
            if self._incomplete_operation == 0:
                self._incomplete_operation = self.choose_incomplete_operation()
            if self._incomplete_operation == 0:
                return False
            if self._incomplete_operation == 1:
                self.set_readonly(False)  # Sets the cluster as write-enabled.
                self.clear(ignore_warning=True)  # Clears any incomplete data if present.
                self._incomplete = False
                self.set_readonly(True)  # Restores the read-only flag.
                return True
            elif self._incomplete_operation == 2:
                raise NotImplementedError
            elif self._incomplete_operation == 3:
                raise NotImplementedError
            else:
                raise ValueError(f"invalid operation {self._incomplete_operation}")

    def open(self):
        if self.__closed == True:
            self.__closed = not self._open()

    def close(self):
        '''Method to close the cluster, preventing further operations.'''
        if self.__closed == False:
            self.set_readonly()
            self.__closed = self._close()

    def is_close(self, with_warning = False):
        '''Method to check if the cluster is closed.'''
        if with_warning and self.__closed:
            warnings.warn(f"{self.__class__.__name__}:{self.sub_dir} is closed, any I/O operation will not be executed.",
                            ClusterIONotExecutedWarning)
        return self.__closed

    def is_open(self):
        return not self.is_close()

    def set_readonly(self, readonly=True):
        '''Method to set the cluster as read-only or write-enabled.'''
        if self.is_writable() and readonly == True:
            self.stop_writing()
        self.__readonly = readonly

    def set_writable(self, writable=True):
        '''Method to set the cluster as writable or read-only.'''
        self.set_readonly(not writable)

    def set_overwrite_allowed(self, overwrite_allowed=True):
        '''Method to set the cluster as writable or read-only.'''
        self.__overwrite_allowed = overwrite_allowed

    def is_readonly(self, with_warning = False):
        '''Method to check if the cluster is read-only.'''
        if with_warning and self.__readonly:
            warnings.warn(f"{self.__class__.__name__}:{self.sub_dir} is read-only, any write operation will not be executed.",
                ClusterIONotExecutedWarning)
        return self.__readonly
    
    def is_writable(self):
        return not self.is_readonly()

    def is_overwrite_allowed(self):
        return self.__overwrite_allowed

    @staticmethod
    def _read_decorator(func):
        '''
        brief
        -----
        Decorator function to handle reading operations when the cluster is closed. \n
        if the cluster is closed, the decorated function will not be executed and return None. \n
        
        parameter
        -----
        func: Callable, the decorated function
        '''
        def wrapper(self: "_DataCluster", data_i, *args, force = False, **kwargs):
            if not force:
                if self.is_close(with_warning=True):
                    return None
            try:
                rlt = func(self, data_i, *args, **kwargs)  # Calls the original function.
            except ClusterDataIOError:
                rlt = None
            return rlt
        return wrapper

    @staticmethod
    def _write_decorator(func):
        '''
        brief
        -----
        Decorator function to handle writing operations when the cluster is closed or read-only. \n
        if the cluster is closed, the decorated function will not be executed and return None. \n
        if the cluster is read-only and the decorated function is a writing operation, the decorated function will not be executed and return None.\n
        
        parameter
        -----
        func: Callable, the decorated function
        '''
        def wrapper(self: "_DataCluster", data_i = None, value = None, *args, force = False, **kwargs):
            if force:
                orig_closed = self.__closed
                orig_readonly = self.__readonly
                self.open()
                self.set_writable()

            overwrited = False

            if self.is_close(with_warning=True) or self.is_readonly(with_warning=True):
                return None
            if data_i in self.keys():
                if not self.overwrite_allowed and not force:
                    warnings.warn(f"{self.__class__.__name__}:{self.sub_dir} \
                                is not allowed to overwitre, any write operation will not be executed.",
                                    ClusterIONotExecutedWarning)
                    return None
                overwrited = True                    
            
            try:
                if not self.is_writing:
                    self.start_writing()
                rlt = func(self, data_i, value, *args, **kwargs)  # Calls the original function.
            except ClusterDataIOError:
                rlt = None
            else:
                self.changed_since_opening = True  # Marks the cluster as updated after writing operations.
                if data_i is None:
                    self._update_cluster_all(*args, **kwargs)
                else:
                    self._update_cluster_inc(data_i, *args, **kwargs)
                
                if overwrited:
                    self.log_to_mark_file(data_i, self.LOG_CHANGE)
                else:
                    self.log_to_mark_file(data_i, self.LOG_APPEND)

            if force:
                if orig_closed:
                    self.close()
                if orig_readonly:
                    self.set_readonly()
            return rlt
        return wrapper

    def clear(self, *, force = False, ignore_warning = False):
        '''
        Method to clear any data in the cluster. Subclasses may implement this method.
        * it is dargerous
        '''
        if not ignore_warning:
            y = input("All files in {} will be deleted, please enter 'y' to confirm".format(self.directory))
        else:
            y = 'y'
        if y == 'y':
            self.__cluster_clear_func(self, force = force)
            return True
        else:
            return False

    def read(self, data_i, *args, force = False, **kwargs)->VDCT:
        '''
        Method to read data from the cluster. Subclasses must implement this method.
        '''
        return self.__cluster_read_func(self, data_i, *args, force=force, **kwargs)

    def write(self, data_i, value:VDCT, *args, force = False, **kwargs):
        '''
        Method to write data to the cluster. Subclasses must implement this method.

        parameter
        ----
        * data_i: Any, the key of the data
        * value: DCT, the value of the data
        * force: bool, whether to force write, it's not recommended to use force=True if you want to write a lot of data
        '''
        assert self.check_key(data_i)
        return self.__cluster_write_func(self, data_i, value, *args, force = force, **kwargs)

    def copyto(self, dst: str, cover:bool, *args, **kwargs):
        '''
        copy the cluster to dst
        '''
        if os.path.exists(dst):
            if cover:
                warnings.warn(f"{dst} exists, it will be deleted", ClusterParaWarning)
                shutil.rmtree(dst)
            else:
                raise ValueError(f"{dst} exists, it will not be deleted, please set cover=True")
        self._copyto(dst, *args, **kwargs)

    def merge(self, src:"_DataCluster", *args, **kwargs):
        assert self.is_open() and self.is_writable(), f"{self.__class__.__name__}:{self.sub_dir} is not writable"
        assert type(src) == type(self), f"can't merge {type(src)} to {type(self)}"
        for k in tqdm(src.keys(), desc=f"merge {src.sub_dir} to {self.sub_dir}", total=len(src)):
            self._merge_one(src, k, *args, **kwargs)

    def start_writing(self, overwrite_allowed = False):
        '''
        rewrite the method of WriteController
        '''
        if super().start_writing(overwrite_allowed):
            self.open()
            self.set_writable()
            self._start_writing()
            if not self.overwrite_allowed:
                self.set_overwrite_allowed(overwrite_allowed)
            return True
        else:
            return False

    def stop_writing(self):
        '''
        rewrite the method of WriteController
        '''
        if not self.is_writing:
            return False
        else:
            self.set_overwrite_allowed(False)
            self._stop_writing()
            self._update_cluster_all()
            return super().stop_writing()

    def __repr__(self):
        return f"{self.identity_string()} at {hex(id(self))}"

    def register_to_dataset(self):
        if self.register:
            self.dataset_node.cluster_map[self.identity_string()] = self

class JsonDict(_DataCluster[DSNT, VDCT], dict):
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
    SAVE_IMMIDIATELY = 0
    SAVE_AFTER_CLOSE = 1
    SAVE_STREAMLY = 2

    class _Placeholder():
        def __init__(self) -> None:
            pass

    def __init__(self, dataset_node:DSNT, sub_dir, register = True, name = "", *args, **kwargs):       
        dict.__init__(self)
        super().__init__(dataset_node, sub_dir, register, name, *args, **kwargs)

    @property
    def save_mode(self):
        return self.__save_mode
    
    @save_mode.setter
    def save_mode(self, mode):
        assert mode in [self.SAVE_IMMIDIATELY, self.SAVE_AFTER_CLOSE, self.SAVE_STREAMLY]
        if self.is_writing and mode != self.SAVE_STREAMLY:
            warnings.warn("can't change save_mode from SAVE_STREAMLY to the others while writing streamly", 
                          ClusterParaWarning)
        if self.__save_mode == self.SAVE_AFTER_CLOSE and mode != self.SAVE_AFTER_CLOSE and self.is_open():
            self.save()
        self.__save_mode = mode

    @property
    def write_streamly(self):
        return self.save_mode == self.SAVE_STREAMLY

    def __len__(self) -> int:
        super().__len__()
        return dict.__len__(self)

    def keys(self):
        super().keys()
        return dict.keys(self)

    def values(self):
        super().values()
        return dict.values(self)
    
    def items(self):
        super().items()
        return dict.items(self)
    
    def _read(self, data_i, *arg, **kwargs):
        super()._read(data_i, *arg, **kwargs)
        try:
            return dict.__getitem__(self, data_i)
        except KeyError:
            raise ClusterDataIOError(f"key {data_i} does not exist")

    def _write(self, data_i, value, *arg, **kwargs):
        super()._write(data_i, value, *arg, **kwargs)

        if self.save_mode == self.SAVE_IMMIDIATELY:
            rlt = dict.__setitem__(self, data_i, value)
            self.save()
        elif self.save_mode == self.SAVE_AFTER_CLOSE:
            rlt = dict.__setitem__(self, data_i, value)
        elif self.save_mode == self.SAVE_STREAMLY:
            set_value = self._Placeholder()
            if not self.is_writing:
                self.start_writing() # auto start writing if save_mode is SAVE_STREAMLY
            self.stream.write({data_i: value})
            rlt = dict.__setitem__(self, data_i, set_value)
        return rlt

    def _clear(self, *arg, **kwargs):
        with open(self.directory, 'w'):
            pass
        dict.clear(self)

    def _copyto(self, dst: str, *args, **kwargs):
        super()._copyto(dst, *args, **kwargs)
        os.makedirs(os.path.split(dst)[0], exist_ok=True)
        shutil.copy(self.directory, dst)

    def _merge_one(self, src: "JsonDict", k, *args, **kwargs):
        v = src[k]
        if isinstance(k, int):
            self[self.cluster_data_i_upper] = v
        elif isinstance(k, str):
            new_k = k
            _count = 1
            while new_k in self.keys():
                warnings.warn(f"key {new_k} exists, it will be renamed to {new_k}.{str(_count)}", ClusterNotRecommendWarning)
                new_k = k + f".{str(_count)}"
                _count += 1
            self[new_k] = v
        else:
            raise TypeError(f"unsupported type: {type(k)} in src")
        self.log_to_mark_file(self.cluster_data_i_upper - 1, self.LOG_APPEND)

    def _init_attr(self, *args, **kwargs):
        _DataCluster._init_attr(self, *args, **kwargs)
        self.reload()
        self.__save_mode = self.SAVE_AFTER_CLOSE
        self.stream = JsonIO.Stream(self.directory)

    def __iter__(self) -> Iterator:
        return _DataCluster.__iter__(self)

    def _stop_writing(self):
        '''
        rewrite the method of WriteController
        '''
        if self.is_writing:
            self.stop_writing()
        if self.save_mode == self.SAVE_AFTER_CLOSE:
            self.sort()
            self.save()
        super()._stop_writing()

    def update(self, _m:Union[dict, Iterable[tuple]] = None, **kw):
        if _m is None:
            warnings.warn(f"It's not recommended to use update() for {self.__class__.__name__} \
                for it will call {self._update_cluster_all.__name__} and spend more time. \
                nothine is done. \
                use {self.__setitem__.__name__} or {self.write.__name__} or {self.__setitem__.__name__} instead", ClusterNotRecommendWarning)
        elif isinstance(_m, dict):
            for k, v in _m.items():
                assert self.checkkey(k), f"key {k} is not allowed"
                self[k] = v
        elif isinstance(_m, Iterable):
            assert all([len(x) == 2 for x in _m])
            for k, v in _m:
                assert self.checkkey(k), f"key {k} is not allowed"
                self[k] = v
        else:
            raise TypeError(f"unsupported type: {type(_m)}")

    def reload(self, value = {}):
        dict.clear(self)
        if isinstance(value, dict) and len(value) > 0:
            pass
        elif os.path.exists(self.directory):
            try:
                value = JsonIO.load_json(self.directory)
            except JSONDecodeError:
                self._error_to_load = True
                value = {}
        else:
            value = {}
        dict.update(self, value)

    @_DataCluster._write_decorator
    def sort(self, *arg, **kwargs):
        '''
        sort by keys
        '''
        if self.is_writing:
            warnings.warn(f"It's no effect to call {self.sort.__name__} while writing", ClusterNotRecommendWarning)
            return 
        new_dict = dict(sorted(self.items(), key=lambda x: x[0]))
        self.reload(new_dict)
        if self.save_mode == self.SAVE_IMMIDIATELY:
            self.save()

    def start_writing(self, overwrite_allowed = False):
        '''
        rewrite the method of WriteController
        '''
        if _DataCluster.start_writing(self, overwrite_allowed):
            self.save_mode = self.SAVE_STREAMLY
            self.stream.open()
            return True
        else:
            return False

    def stop_writing(self):
        '''
        rewrite the method of WriteController
        '''
        if not self.is_writing:
            return False
        else:
            self.stream.close()
            self.reload()
            _DataCluster.stop_writing(self)
            return 

    def save(self):
        JsonIO.dump_json(self.directory, self)

class Elements(_DataCluster[DSNT, VDCT]):
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
    * data_i_upper: int, the max index of the iterator

    method
    ----
    * __len__: return the number of elements
    * __iter__: return the iterator
    * __next__: return the next element
    * read: read one element from disk with the logic of self.readfunc
    * write: write one element to disk with the logic of self.writefunc
    * format_path: format the path of the element
    * clear: clear all data of the dict and clear directory
    '''

    WRITING_MARK = ".elmsw" # the mark file of writing process, Elements streamly writing

    def __init__(self, 
                dataset_node:DSNT,
                sub_dir,
                register = True,
                name = "",
                read_func:Callable = lambda x: None, 
                write_func:Callable = lambda x,y: None, 
                suffix:str = '.txt', 
                filllen = 6, 
                fillchar = '0', *,
                alternative_suffix:list[str] = []) -> None:
        super().__init__(dataset_node, sub_dir, register, name, 
                         read_func, write_func, suffix, filllen, fillchar, alternative_suffix = alternative_suffix)

    @property
    def data_i_dir_map(self):
        if len(self._data_i_dir_map) == 0:
            self._update_cluster_all()
        return self._data_i_dir_map
    
    @property
    def data_i_appendnames(self):
        if len(self._data_i_appendnames) == 0:
            self._update_cluster_all()
        return self._data_i_appendnames

    @property
    def cache_used(self):
        return self.__cache_used
    
    @cache_used.setter
    def cache_used(self, value:bool):
        if not self.has_cache and value:
            warnings.warn(f"no cache file, can not set. call {self.save_cache.__name__} first", ClusterParaWarning)
        elif not self.has_source and not value:
            warnings.warn(f"no source file, can not set. try {self.unzip_cache.__name__} first", ClusterParaWarning)
        else:
            if value == True:
                self.load_cache()
            self.__cache_used = bool(value)

    @property
    def has_cache(self):
        cache_path = os.path.join(self.directory, self.__cache_name)
        return os.path.exists(cache_path)

    @property
    def has_source(self):
        try:
            file_path = self.auto_path(0)
            return True
        except ClusterDataIOError:
            return False

    def __len__(self):
        '''
        Count the total number of files in the directory
        '''
        super().__len__()
        # count = 0
        # for root, dirs, files in os.walk(self.directory):
        #     count += len(glob.glob(os.path.join(root, f'*{self.suffix}')))
        return len(self.data_i_dir_map)

    def keys(self):
        '''
        brief
        -----
        return a generator of data_i
        * Elements is not a dict, so it can't be used as a dict.
        '''
        super().keys()
        def data_i_generator():
            for i in self.data_i_dir_map.keys():
                yield i
        return data_i_generator()
    
    def values(self):
        super().values()
        def value_generator():
            for i in self.data_i_dir_map.keys():
                yield self.read(i)
        return value_generator()
    
    def items(self):
        super().items()
        def items_generator():
            for i in self.data_i_dir_map.keys():
                yield i, self.read(i)
        return items_generator()

    def _read(self, data_i, appdir = "", appname = "", *arg, **kwargs)->VDCT:
        super()._read(data_i, *arg, **kwargs)
        if not self.cache_used:
            path = self.format_path(data_i, appdir=appdir, appname=appname)
            if not os.path.exists(path):
                path = self.auto_path(data_i)
            if os.path.exists(path):
                return self.read_func(path, *arg, **kwargs)
            else:
                raise ClusterDataIOError(f"can't find {path}")
        else:
            if data_i in self.__cache:
                return self.__cache[data_i]
            else:
                raise ClusterDataIOError(f"can't find {path}")
        
    def _write(self, data_i, value:VDCT, appdir = "", appname = "", *arg, **kwargs):
        super()._write(data_i, value, *arg, **kwargs)
        if not self.cache_used:
            path = self.format_path(data_i, appdir=appdir, appname=appname)
            dir_ = os.path.split(path)[0]
            os.makedirs(dir_, exist_ok=True)
            if value is not None:
                self.write_func(path, value, *arg, **kwargs)
        else:
            warnings.warn("can't write when 'cache_used' is True", ClusterIONotExecutedWarning)

    def _clear(self, *arg, **kwargs):
        super()._clear(*arg, **kwargs)
        shutil.rmtree(self.directory)
        os.makedirs(self.directory)

    def _copyto(self, dst: str, *args, **kwargs):
        super()._copyto(dst, *args, **kwargs)
        if not self.cache_used:
            shutil.copytree(self.directory, dst)
        else:
            os.makedirs(dst, exist_ok=True)
            shutil.copy(os.path.join(self.directory, self.__data_i_dir_map_name), 
                        os.path.join(dst, self.__data_i_dir_map_name))
            shutil.copy(os.path.join(self.directory, self.__data_i_appendnames_name),
                        os.path.join(dst, self.__data_i_appendnames_name))
            shutil.copy(os.path.join(self.directory, self.__cache_name),
                        os.path.join(dst, self.__cache_name))

    def _merge_one(self, src: "Elements", data_i:int, *args, **kwargs):
        appdir, append_names = src.auto_path(data_i, return_app=True, allow_mutil_appendname=True)
        for name in append_names:
            src_path = src.format_path(data_i, appdir=appdir, appname=name)
            new_data_i = self.cluster_data_i_upper
            new_path = self.format_path(new_data_i, appdir=appdir, appname=name)
            shutil.copy(src_path, new_path)
            self._update_cluster_inc(new_data_i, appdir=appdir, appname=name)
            self.log_to_mark_file(new_data_i, self.LOG_APPEND)
        self.save_maps()

    def _init_attr(self, read_func, write_func, suffix, filllen, fillchar, alternative_suffix = [], *args, **kwargs):
        def _check_dot(suffix:str):
            if not suffix.startswith('.'):
                suffix = '.' + suffix
            return suffix

        super()._init_attr(*args, **kwargs)
        assert isinstance(suffix, str), f"suffix must be str, not {type(suffix)}"
        assert isinstance(filllen, int), f"filllen must be int, not {type(filllen)}"
        assert isinstance(fillchar, str), f"fillchar must be str, not {type(fillchar)}"
        assert isinstance(alternative_suffix, (list, tuple)), f"alternative_suffix must be list or tuple, not {type(alternative_suffix)}"
        self.__data_i_dir_map_name = "data_i_dir_map.elmm"
        self.__data_i_appendnames_name = "data_i_appendnames.elmm"
        self.__cache_name = "cache.elmcache"

        self.filllen    = filllen
        self.fillchar   = fillchar
        self.suffix     = suffix
        self.__alternative_suffix = alternative_suffix
        self.suffix = _check_dot(self.suffix)
        self.__alternative_suffix = [_check_dot(x) for x in self.__alternative_suffix]

        self.read_func  = read_func
        self.write_func = write_func
        
        self._data_i_dir_map = {}
        self._data_i_appendnames = {}

        self.__cache_used = False
        self.__cache = {}

        self.load_maps()
        if not self.has_source:
            if self.load_cache():
                self.cache_used = True

    def _update_cluster_inc(self, data_i, appdir = "", appname = "", *arg, **kwargs):
        self._data_i_dir_map[data_i] = appdir
        self._data_i_appendnames.setdefault(data_i, []).append(appname)

    def _update_cluster_all(self, *args, **kwargs):
        # if len(self) > 2000:
        #     print(f'init {self.directory} data_i_dir_map, this may take a while...')        
        self._data_i_dir_map.clear()
        self._data_i_appendnames.clear()
        paths = glob.glob(os.path.join(self.directory, "**/*" + self.suffix), recursive=True)
        for path in paths:
            data_i, appdir, appname = self.parse_path(path)
            self._update_cluster_inc(data_i, appdir=appdir, appname=appname)
        # sort by data_i
        self._data_i_dir_map = dict(sorted(self._data_i_dir_map.items(), key=lambda x:x[0]))
        self._data_i_appendnames = dict(sorted(self._data_i_appendnames.items(), key=lambda x:x[0]))

        self.save_maps()

    def __getitem__(self, idx):
        return self.read(data_i=idx)
        
    def __setitem__(self, idx, value):
        if idx in self.data_i_dir_map:
            appdir, appname = self.auto_path(idx, return_app=True)
            self.write(idx, value, appdir=appdir, appname = appname)
        else:
            raise KeyError(f'idx {idx} not in {self.directory}, if you want to add new data, \
                           please use method:write to specify the appdir and appname')

    def __iter__(self):
        self.data_i_dir_map
        return self.values()
    
    def _open(self):
        open_allowed = super()._open()
        if open_allowed and not os.path.exists(self.directory):
            print(f"Elements: {self.directory} is new, it will be created")
            os.makedirs(self.directory, exist_ok=True)
        return open_allowed

    def _stop_writing(self):
        if self.changed_since_opening:
            # if not self.check_storage():
            self._update_cluster_all()     
            self.save_maps()
        return super()._stop_writing()

    def read(self, data_i, appdir = "", appname = "", *arg, force = False, **kwarg):
        '''
        parameter
        ----
        * data_i: int, the index of the data
        * appdir: str, the sub directory of the root directory
        * appname: str, the string to be added to the file name(before the suffix)
        '''
        return super().read(data_i, appdir = appdir, appname = appname, *arg, force = force, **kwarg)

    def write(self, data_i, element:VDCT, appdir = "", appname = "", *arg, force = False, **kwarg):
        '''
        parameter
        ----
        * data_i: int, the index of the data
        * element: the element to be written
        * appdir: str, the sub directory of the root directory
        * appname: str, the string to be added to the file name(before the suffix)
        '''
        return super().write(data_i, element, appdir = appdir, appname = appname, *arg, force=force, **kwarg)

    def format_base_name(self, data_i):
        return "{}".format(str(data_i).rjust(self.filllen, "0"))

    def format_path(self, data_i, appdir = "", appname = "", **kw):
        '''
        format the path of data_i
        '''
        if appname and appname[0] != '_':
            appname = '_' + appname # add '_' before appname
        return os.path.join(self.directory, appdir, 
                            "{}{}{}".format(
                                self.format_base_name(data_i), 
                                appname, 
                                self.suffix))

    def parse_path(self, path:str):
        '''
        parse the path to get data_i, appdir, appname, it is the reverse operation of format_path
        '''
        appdir, file = os.path.split(os.path.relpath(path, self.directory))
        filename = os.path.splitext(file)[0]
        split_filename = filename.split('_')
        mainname = split_filename[0]
        try:
            appname  = "_".join(split_filename[1:])
        except IndexError:
            appname  = ""
        data_i = int(mainname)
        return data_i, appdir, appname

    def auto_path(self, data_i, return_app = False, allow_mutil_appendname = False):
        '''
        auto find the path of data_i. \n
        * if data_i has multiple appendnames, raise IndexError

        if return_app is True, return appdir, appname, else return path

        if allow_mutil_appendname is True, the type of appname will be list[str], else str; 
        and the type of path will be list[str], else str
        '''
        def format_one(data_i, appdir, appname):
            path = self.format_path(data_i, appdir=appdir, appname=appname)
            if os.path.exists(path):
                return path
            else:
                raise ClusterDataIOError(f'idx {data_i} has no file in {self.directory}')     
        
        if data_i in self.data_i_dir_map and data_i in self.data_i_appendnames:
            appdir = self.data_i_dir_map[data_i]
            appendnames = self.data_i_appendnames[data_i]
            if allow_mutil_appendname:
                appname:list[str] = appendnames
            else:
                if len(appendnames) == 1:
                    appname:str = appendnames[0]
                else:
                    raise ClusterDataIOError(f'idx {data_i} has more than one appendname: {appendnames}, its path is ambiguous. \
                                    You must specify the appname by using method:read(data_i, appname=...)')
        else:
            appdir = ""
            appname = ""
        if not return_app:
            if isinstance(appname, list):
                path_list = []
                for n in appname:
                    path = format_one(data_i, appdir, n)
                    path_list.append(path)
                return path_list
            else:
                return format_one(data_i, appdir, appname)
        else:
            return appdir, appname

    def check_storage(self):
        paths = glob.glob(os.path.join(self.directory, "**/*" + self.suffix), recursive=True)
        files = [int(os.path.split(path)[1][:self.filllen]) for path in paths]
        files_set = set(files)
        cur_keys_set = set(self.keys())        
        if len(files_set) != len(files):
            raise ValueError(f"there are duplicate files in {self.directory}")
        if len(files_set) != len(cur_keys_set):
            return False
        elif files_set == cur_keys_set:
            return True
        else:
            return False

    def save_maps(self):
        if not len(self._data_i_dir_map) > 0:
            warnings.warn("len(self._data_i_dir_map) == 0, nothing to save", ClusterNotRecommendWarning)
            return
        for name, map in zip((self.__data_i_dir_map_name, self.__data_i_appendnames_name), 
                             (self._data_i_dir_map, self._data_i_appendnames)):
            path = os.path.join(self.directory, os.path.splitext(name)[0] + ".npy")
            path_elmm = os.path.join(self.directory, name)
            np.save(path, map)
            try:
                os.remove(path_elmm)
            except FileNotFoundError:
                pass
            os.rename(path, path_elmm)

    def load_maps(self):
        data_i_dir_map_path         = os.path.join(self.directory, self.__data_i_dir_map_name)
        data_i_appendnames_path     = os.path.join(self.directory, self.__data_i_appendnames_name)
        if os.path.exists(data_i_dir_map_path) and os.path.exists(data_i_appendnames_path):
            self._data_i_dir_map        = np.load(data_i_dir_map_path, allow_pickle=True).item()
            self._data_i_appendnames    = np.load(data_i_appendnames_path, allow_pickle=True).item()
            return True
        else:
            self._update_cluster_all()
    
    def __parse_cache_kw(self, **kw):
        assert all([isinstance(v, list) for v in kw.values()])
        assert all([len(v) == len(self) for v in kw.values()])
        kw_keys = list(kw.keys())
        kw_values = list(kw.values())
        return kw_keys, kw_values

    def save_cache(self, **kw):
        self.__cache.clear()
        data_i_list = list(self.keys())
        kw_keys, kw_values = self.__parse_cache_kw(**kw)
        for data_i in tqdm(data_i_list, desc="save cache for {}".format(self.directory)):
            read_kw = {k:v[data_i] for k, v in zip(kw_keys, kw_values)}
            elem = self.read(data_i, **read_kw)
            self.__cache[data_i] = elem
        serialize_object(os.path.join(self.directory, self.__cache_name), self.__cache)

    def load_cache(self):
        cache_path = os.path.join(self.directory, self.__cache_name)
        if os.path.exists(cache_path):
            self.__cache = deserialize_object(cache_path)
            return True
        else:
            return False
        
    def unzip_cache(self, force = False, **kw):
        '''
        unzip the cache to a dict
        '''
        assert os.path.exists(os.path.join(self.directory, self.__data_i_dir_map_name))
        assert os.path.exists(os.path.join(self.directory, self.__data_i_appendnames_name))
        assert os.path.exists(os.path.join(self.directory, self.__cache_name))
        kw_keys, kw_values = self.__parse_cache_kw(**kw)
        if len(self.__cache) == 0:
            self.load_cache()
        progress = tqdm(zip(self._data_i_dir_map.keys(), self._data_i_appendnames.keys(), self.__cache.keys()), 
                        desc="unzip cache for {}".format(self.directory),
                        total=len(self.data_i_dir_map))
        for kd, ka, kc in progress:
            assert kd == ka == kc
            data_i = kd
            value = self.__cache[kd]
            appdir = self.data_i_dir_map[data_i]
            appname = self.data_i_appendnames[data_i][0]
            write_kw = {k:v[data_i] for k, v in zip(kw_keys, kw_values)}
            self.write(kd, value, appdir, appname, force=force, **write_kw)

class SingleFile(Generic[VDCT]):
    def __init__(self, sub_path:str, read_func:Callable, write_func:Callable) -> None:
        self.sub_path = sub_path
        self.read_func:Callable = read_func
        self.write_func:Callable = write_func
        self.cluster:FileCluster = None

    @property
    def path(self):
        return os.path.join(self.cluster.directory, self.sub_path)

    @property
    def exist(self):
        return os.path.exists(self.path)

    def set_cluster(self, cluster:"FileCluster"):
        self.cluster = cluster

    def read(self) -> VDCT:
        return self.read_func(self.path)
    
    def write(self, data):
        self.write_func(self.path, data)

class FileCluster(_DataCluster[DSNT, VDCT]):
    '''
    a cluster of multiple files, they may have different suffixes and i/o operations
    but they must be read/write together
    '''
    SingleFile = SingleFile

    def __init__(self, dataset_node: DSNT, sub_dir, register = True, name = "", singlefile_list:list[SingleFile] = []) -> None:
        super().__init__(dataset_node, sub_dir, register, name, singlefile_list)
        os.makedirs(self.directory, exist_ok=True)

    @property
    def all_exist(self):
        for f in self.fileobjs_dict.values():
            if not f.exist:
                return False
        return True

    def __len__(self):
        super().__len__()
        return len(self.fileobjs_dict)

    def keys(self):
        super().keys()
        return list(self.fileobjs_dict.keys())
    
    def values(self) -> list[SingleFile[VDCT]]:
        super().values()
        return list(self.fileobjs_dict.values())
    
    def items(self):
        super().items()
        return self.keys(), self.values()

    def _read(self, data_i, *arg, **kwargs):
        super()._read(data_i, *arg, **kwargs)
        file_path = self.filter_data_i(data_i)
        return self.fileobjs_dict[file_path].read()
    
    def _write(self, data_i, value, *arg, **kwargs):
        super()._write(data_i, value, *arg, **kwargs)
        file_path = self.filter_data_i(data_i)
        return self.fileobjs_dict[file_path].write(value)

    def _clear(self, *arg, **kwargs):
        super()._clear(*arg, **kwargs)
        for fp in self.fileobjs_dict.keys():
            if os.path.exists(fp):
                os.remove(fp)

    def _copyto(self, dst: str, cover = False, *args, **kwargs):
        super()._copyto(dst, *args, **kwargs)
        os.makedirs(dst, exist_ok=True)
        for f in self.fileobjs_dict.values():
            dst_path = os.path.join(dst, f.sub_path)
            if os.path.exists(dst_path):
                if cover:
                    os.remove(dst_path)
                else:
                    raise FileExistsError(f"{dst_path} already exists, please set cover=True")
            shutil.copy(f.path, dst_path)

    def _merge_one(self, src: "FileCluster", key:str, merge_funcs=[], *args, **kwargs):
        if key in self.fileobjs_dict:
            this_i = list(self.fileobjs_dict.keys()).index(key)
            func = merge_funcs[this_i]
            src_data = src.read(key)
            this_data = self.read(this_i)
            new_data = func(this_data, src_data)
            self.write(this_i, new_data)
        else:
            new_dir = os.path.join(self.directory, self.sub_dir)
            new_path = os.path.join(new_dir, src.fileobjs_dict[key].sub_path)
            os.makedirs(new_dir, exist_ok=True)
            shutil.copy(src.fileobjs_dict[key].path, new_path)
            self.update_file(
                SingleFile(
                    src.fileobjs_dict[key].sub_path, 
                    src.fileobjs_dict[key].read_func, 
                    src.fileobjs_dict[key].write_func))
            
            self.log_to_mark_file(key, self.LOG_APPEND)

    def merge(self, src: "FileCluster", merge_funcs:Union[list, Callable]=[], *args, **kwargs):
        # assert all([selfkey == srckey for selfkey, srckey in zip(self.keys(), src.keys())]), "the keys of two FileCluster must be the same"
        if isinstance(merge_funcs, Callable):
            merge_funcs = [merge_funcs for _ in range(len(self.fileobjs_dict))]
        elif isinstance(merge_funcs, (list, tuple)):
            assert len(merge_funcs) == len(self.fileobjs_dict), f"the length of merge_funcs must be {len(self.fileobjs_dict)}"
            assert all([isinstance(f, Callable) for f in merge_funcs]), "all merge_funcs must be Callable"
        super().merge(src, merge_funcs = merge_funcs, *args, **kwargs)

    def _init_attr(self, singlefile:list[SingleFile], *args, **kwargs):
        super()._init_attr(singlefile, **kwargs)
        self.fileobjs_dict:dict[str, SingleFile] = {}

        for f in singlefile:
            self.update_file(f)

    def filter_data_i(self, data_i, return_index = False):
        def process_func(key, string):
            return key == get_mainname(string)
        target_int = search_in_dict(self.fileobjs_dict, data_i, return_index = True, process_func = process_func)
        if not return_index:
            return list(self.fileobjs_dict.keys())[target_int]
        else:
            return target_int

    def copyto(self, dst: str, cover = False, *args, **kwargs):
        '''
        copy the whole directory to dst
        '''
        self._copyto(dst, cover, *args, **kwargs)

    def read_all(self):
        return [f.read() for f in self.values()]

    def write_all(self, values):
        assert len(values) == len(self), f"the length of value must be {len(self)}"
        for f, d in zip(self.values(), values):
            f.write(d)

    def remove_file(self, idx:Union[int, str]):
        path = self.filter_data_i(idx)
        self.fileobjs_dict.pop(path)

    def update_file(self, singlefile:SingleFile):
        singlefile.set_cluster(self)
        self.fileobjs_dict[singlefile.path] = singlefile

    def get_file(self, idx:Union[int, str]):
        return self.fileobjs_dict[self.filter_data_i(idx)]

    def paths(self):
        return list(self.keys())

class IntArrayDictElement(Elements[DSNT, dict[int, np.ndarray]]):
    def __init__(self, dataset_node: DSNT, sub_dir:str, array_shape:tuple[int], array_fmt:str = "", register=True, name = "", filllen=6, fillchar='0') -> None:
        super().__init__(dataset_node, sub_dir, register, name, self._read_func, self._write_func, ".txt", filllen, fillchar)
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

### Data Node ###
class Node():
    def __init__(self:NDT, parent:"Node" = None):
        assert isinstance(parent, Node) or parent is None
        self.parent:NDT = None
        self.children:list[NDT] = []
        self.move_node(parent)

    def add_child(self, child_node:"Node"):
        assert isinstance(child_node, Node), f"child_node must be Node, not {type(child_node)}"
        child_node.parent = self
        self.children.append(child_node)

    def remove_child(self, child_node:"Node"):
        assert isinstance(child_node, Node), f"child_node must be Node, not {type(child_node)}"
        if child_node in self.children:
            child_node.parent = None
            self.children.remove(child_node)

    def move_node(self, new_parent:"Node"):
        assert isinstance(new_parent, Node) or new_parent is None, f"new_parent must be Node, not {type(new_parent)}"
        if self.parent is not None:
            self.parent.remove_child(self)
        if new_parent is not None:
            new_parent.add_child(self)

class _ClusterMap(dict[str, DCT]):
    def __init__(self, dataset_node:"DatasetNode", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dataset_node = dataset_node

    def __set_update(self):
        self.dataset_node.updated = True
        if self.dataset_node.inited:
            self.dataset_node.update_dataset()

    # def values(self) -> dict_values[str, DCT]:
    #     return super().values()
    
    # def items(self) -> dict_items[str, DCT]:
    #     return super().items()

    def __setitem__(self, __key: Any, __value: Any) -> None:
        self.__set_update()
        return super().__setitem__(__key, __value)
    
    def update(self, __m, **kwargs: Any) -> None:
        self.__set_update()
        return super().update(__m, **kwargs)

    def setdefault(self, __key: Any, __default: Any = ...) -> Any:
        self.__set_update()
        return super().setdefault(__key, __default)
    
    # def __getitem__(self, __key: Any) -> Any:
    #     return search_in_dict(self, __key, process_func=self._search_func)
    #     # return super().__getitem__(__key)

    def search(self, __key: Any, return_index = False):
        return search_in_dict(self, __key, return_index, process_func=self._search_func) ### TODO
    
    def add_cluster(self, cluster:DCT):
        cluster.dataset_node = self.dataset_node
        cluster.register_to_dataset()

    def get_keywords(self):
        keywords = []
        for indetity_string in self.keys():
            _,_,_,sub_dir, name = _DataCluster.parse_identity_string(indetity_string)
            if name != "":
                keywords.append(name)
            else:
                keywords.append(sub_dir)
        return keywords

    @staticmethod
    def _search_func(key, indetity_string:str):
        _,_,_,sub_dir, name = _DataCluster.parse_identity_string(indetity_string)
        if name != "":
            return key == name
        else:
            return key == sub_dir

class DatasetNode(Node, InstanceRegistry, Generic[DCT]):
    '''
    DatasetNode, only gather the clusters. 
    have no i/o operations
    '''
    def __init__(self, directory, parent = None, _init_clusters = True) -> None:
        def is_directory_inside(base_dir, target_dir):
            base_dir:str = os.path.abspath(base_dir)
            target_dir:str = os.path.abspath(target_dir)
            return target_dir.startswith(base_dir)
        Node.__init__(self, parent)
        InstanceRegistry.__init__(self)
        self.__inited = False # if the dataset has been inited
        self._updated = False
        self._incomplete_operation = 0
        self._init_identity_paramenter(directory, parent, _init_clusters)
        if self.parent is not None:
            assert is_directory_inside(self.parent.directory, self.directory), f"{self.directory} is not inside {self.parent.directory}"
            self.sub_dir:str = os.path.relpath(self.directory, self.parent.directory)
        else:
            self.sub_dir:str = self.directory
        self.cluster_map = _ClusterMap[DCT](self)
        if _init_clusters:
            self._init_clusters()

        self.__inited = True # if the dataset has been inited

    ### implement InstanceRegistry
    def _init_identity_paramenter(self, directory, parent = None, _init_clusters = True):
        self.directory:str = os.path.normpath(directory)

    def identity_string(self):
        return self.gen_identity_string(self.directory)

    @staticmethod
    def parse_identity_string(identity_string:str):
        return identity_string.split(':')

    @classmethod
    def gen_identity_string(cls, directory, *arg, **kw):
        return f"{cls.__name__}:{directory}"
    ###

    def _init_clusters(self):
        pass    

    @property
    def parent_directory(self):
        if self.parent is None:
            return ""
        else:
            return self.parent.directory

    @property
    def inited(self):
        return self.__inited

    @property
    def updated(self):
        return self._updated
    
    @updated.setter
    def updated(self, value:bool):
        self._updated = bool(value)

    @property
    def clusters(self) -> list[DCT]:
        clusters = list(self.cluster_map.values())
        return clusters

    @property
    def data_clusters(self) -> list[DCT]:
        clusters = [x for x in self.clusters if not isinstance(x, FileCluster)]
        return clusters

    @property
    def opened_clusters(self):
        clusters = [x for x in self.clusters if not x.is_close()]
        return clusters

    @property
    def opened_data_clusters(self):
        clusters = [x for x in self.data_clusters if not x.is_close()]
        return clusters

    @property
    def jsondict_map(self):
        '''
        select the key-value pair whose value is JsonDict
        '''
        return {k:v for k, v in self.cluster_map.items() if isinstance(v, JsonDict)}

    @property
    def elements_map(self):
        '''
        select the key-value pair whose value is Elements
        '''
        return {k:v for k, v in self.cluster_map.items() if isinstance(v, Elements)}
    
    @property
    def filecluster_map(self):
        '''
        select the key-value pair whose value is FileCluster
        '''
        return {k:v for k, v in self.cluster_map.items() if isinstance(v, FileCluster)}

    def update_dataset(self):
        pass

    def save_elements_cache(self):
        for elem in self.elements_map.values():
            elem.save_cache()

    def set_elements_cachemode(self, mode:bool):
        for elem in self.elements_map.values():
            elem.cache_used = bool(mode)
        
    def close_all(self, value = True):
        for obj in self.clusters:
            obj.close() if value else obj.open()

    def open_all(self, value = True):
        self.close_all(not value)

    def set_all_readonly(self, value = True):
        for obj in self.clusters:
            obj.set_readonly(value)

    def set_all_writable(self, value = True):
        self.set_all_readonly(not value)

    def set_all_overwrite_allowed(self, value = True):
        for obj in self.clusters:
            obj.set_overwrite_allowed(value)

    def get_element_paths_of_one(self, data_i:int):
        '''
        brief
        -----
        get all paths of a data
        '''
        paths = {}
        for elem in self.elements_map.values():
            paths[elem.sub_dir] = elem.auto_path(data_i)
        return paths
    
    def get_all_clusters(self, _type:Union[type, tuple[type]] = None):
        cluster_map = _ClusterMap(self)
        cluster_map.update(self.cluster_map)
        for child in self.children:
            cluster_map.update(child.get_all_clusters())

        return cluster_map

    def copyto(self, dst:str, cover = False):
        progress = tqdm(self.opened_clusters)
        for c in progress:
            progress.set_postfix({'copying': "{}:{}".format(c.__class__.__name__, c.directory)})            
            c.copyto(os.path.join(dst, c.sub_dir), cover=cover)

class DatasetFormat(DatasetNode[DCT], WriteController, ABC, Generic[DCT, VDST]):
    """
    # Dataset Format
    -----
    A dataset manager, support mutiple data types.
    It is useful if you have a series of different data. 
    For example, a sample contains an image and a set of bounding boxes. 
    There is a one-to-one correspondence between them, and there are multiple sets of such data.

    properties
    ----
    * inited : bool, if the dataset has been inited
    * updated : bool, if the dataset has been updated, it can bes set.
    * directory : str, the root directory of the dataset
    * incomplete : bool, the last writing process of the dataset has not been completed
    * clusters : list[_DataCluster], all clusters of the dataset
    * opened_clusters : list[_DataCluster], all opened clusters of the dataset
    * jsondict_map : dict[str, JsonDict], the map of jsondict
    * elements_map : dict[str, Elements], the map of elements
    * files_map : dict[str, FileCluster], the map of fileclusters
    * data_num : int, the number of data in the dataset
    * data_i_upper : int, the max index of the iterator

    virtual function
    ----
    * read_one: Read one piece of data

    recommended to rewrite
    ----
    * _init_clusters: init the clusters
    * _write_jsondict: write one piece of data to jsondict
    * _write_elementss: write one piece of data to elements
    * _write_files: write one piece of data to files
    * _update_dataset: update the dataset, it should be called when the dataset is updated

    not necessary to rewrite
    ----
    * update_dataset: update the dataset, it should be called when the dataset is updated
    * read_from_disk: read all data from disk as a generator
    * write_to_disk: write one piece of data to disk
    * start_writing: start writing
    * stop_writing: stop writing    
    * clear: clear all data of the dataset
    * close_all: close all clusters
    * open_all : open all clusters
    * set_all_readonly: set all file streams to read only
    * set_all_writable: set all file streams to writable
    * get_element_paths_of_one: get the paths of one piece of data
    * __getitem__: get one piece of data
    * __setitem__: set one piece of data
    * __iter__: return the iterator of the dataset

    example
    -----
    * read

    df1 = DatasetFormat(directory1) 

    df2 = DatasetFormat(directory2) 

    for data in self.read_from_disk(): 
        ...

    * write_to_disk : 1

    df2.write_to_disk(data) × this is wrong

    with df2.writer:  # use context manager
    
        df2.write_to_disk(data)

    df2.clear()

    * write_to_disk : 2

    df2.start_writing()

    df2.write_to_disk(data)

    df2.stop_writing()
    
    * write_one

    df2.write_one(data_i, data)
    '''

    """
    DATAFRAME = "data_frame.csv"
    DEFAULT_SPLIT_TYPE = ["default"]
    SPLIT_DIR = "ImageSets"

    KW_TRAIN = "train"
    KW_VAL = "val" 

    WRITING_MARK = ".dfsw" # the mark file of writing process, dataset format streamly writing

    def __init__(self, directory, split_rate = 0.75, clear_incomplete = False, parent = None) -> None:
        DatasetNode.__init__(self, directory, parent, _init_clusters = False)        
        WriteController.__init__(self)
        ABC.__init__(self)

        self.split_default_rate = split_rate
        self._data_num = 0
        self._data_i_upper = 0     

        print(f"initializing dataset: {self.__class__.__name__} at {self.directory} ...")
        os.makedirs(self.directory, exist_ok=True)

        self.spliter_group = SpliterGroup(os.path.join(self.directory, self.SPLIT_DIR), 
                                          self.DEFAULT_SPLIT_TYPE, 
                                          self)

        incomplete = self.mark_exist()
        if incomplete and not clear_incomplete:
            y:int = _DataCluster.choose_incomplete_operation(self)
            self._incomplete_operation = y        

        self._init_clusters()
        self.update_dataset()

        self.load_data_frame()        
        if incomplete:
            os.remove(self.get_writing_mark_file())
    
    @property
    def data_num(self):
        return self._data_num
    
    @property
    def data_i_upper(self):
        return self._data_i_upper
        # return max([x.cluster_data_i_upper for x in self.opened_clusters])

    @property
    def train_idx_array(self):
        return self.spliter_group.cur_training_spliter[self.KW_TRAIN]
    
    @property
    def val_idx_array(self):
        return self.spliter_group.cur_training_spliter[self.KW_VAL]

    @property
    def default_train_idx_array(self):
        return self.spliter_group.get_split_array_of(self.DEFAULT_SPLIT_TYPE[0], self.KW_TRAIN)
    
    @property
    def default_val_idx_array(self):
        return self.spliter_group.get_split_array_of(self.DEFAULT_SPLIT_TYPE[0], self.KW_VAL)

    @abstractmethod
    def read_one(self, data_i, *arg, **kwargs) -> VDST:
        pass

    def _write_jsondict(self, data_i, data):
        pass

    def _write_elements(self, data_i, data):
        pass

    def _write_files(self, data_i, data):
        pass

    def _update_dataset(self, data_i = None):
        nums = [len(x) for x in self.opened_data_clusters]
        num = np.unique(nums)
        if len(num) > 1:
            raise ValueError("Unknown error, the numbers of different datas are not equal")
        elif len(num) == 1:
            self._data_num = int(num)
        else:
            self._data_num = 0
        try:
            self._data_i_upper = max([x.cluster_data_i_upper for x in self.opened_data_clusters])
        except ValueError:
            self._data_i_upper = 0
        if self._data_i_upper != self.data_num:
            warnings.warn(f"the data_i_upper of dataset:{self.directory} is not equal to the data_num, \
                          it means the the data_i is not continuous, this may cause some errors", ClusterParaWarning)
        # if self.data_i_upper - 1 >= 0 and hasattr(self, "data_frame"):
        #     self.add_to_data_frame(self.data_i_upper - 1)

    def write_one(self, data_i, data:VDST, *arg, **kwargs):
        assert self.is_writing or all([not x.write_streamly for x in self.jsondict_map.values()]), \
            "write_one cannot be used when any jsondict's stream_dumping_json is True. \
                considering use write_to_disk instead"
        self._write_jsondict(data_i, data)
        self._write_elements(data_i, data)
        self._write_files(data_i, data)

        self.update_dataset(data_i)

        self.log_to_mark_file(data_i, self.LOG_APPEND) ### TODO

    def update_dataset(self, data_i = None, f = False):
        if self.updated or f:
            self._update_dataset(data_i)
            self.updated = False

    def read_from_disk(self, with_data_i = False):
        '''
        brief
        ----
        *generator
        Since the amount of data may be large, return one by one
        '''
        if not with_data_i:
            for i in self.data_frame.index:
                yield self.read_one(i)
        else:
            for i in self.data_frame.index:
                yield i, self.read_one(i)

    def write_to_disk(self, data:VDST, data_i = -1):
        '''
        brief
        -----
        write elements immediately, write basejsoninfo to cache, 
        they will be dumped when exiting the context of self.writer
        
        NOTE
        -----
        For DatasetFormat, the write mode has only 'append'. 
        If you need to modify, please call 'DatasetFormat.clear' to clear all data, and then write again.
        '''
        if not self.is_writing:
            print("please call 'self.start_writing' first, here is the usage example:")
            print(extract_doc(DatasetFormat.__doc__, "example"))
            raise ValueError("please call 'self.start_writing' first")
        if data_i == -1:
            # self._updata_data_num()        
            data_i = self.data_i_upper
        
        self.write_one(data_i, data)

    def start_writing(self, overwrite_allowed = False):
        if super().start_writing(overwrite_allowed):
            print(f"start to write to {self.directory}")
            self.close_all(False)
            self.set_all_readonly(False)
            self.set_all_overwrite_allowed(overwrite_allowed)
        else:
            return False

    def stop_writing(self):
        if not self.is_writing:
            return False
        else:
            for jd in self.jsondict_map.values():
                jd.stop_writing()              
            self.set_all_overwrite_allowed(False)
            self.close_all()
            self.save_data_frame()
            self.spliter_group.save()
            super().stop_writing()
            return True

    def clear(self, ignore_warning = False, force = False):
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
            if force:
                self.close_all(False)
                self.set_all_readonly(False)
                cluster_to_clear = self.clusters
            else:
                cluster_to_clear = self.opened_clusters

            for cluster in cluster_to_clear:
                cluster.clear(ignore_warning=True)

        self.set_all_readonly(True)

    def values(self):
        return self.read_from_disk()
    
    def keys(self):
        for i in self.data_frame.index:
            yield i

    def items(self):
        return self.read_from_disk(True)

    def __getitem__(self, data_i:Union[int, slice]):
        if isinstance(data_i, slice):
            # 处理切片操作
            start, stop, step = data_i.start, data_i.stop, data_i.step
            if start is None:
                start = 0
            if step is None:
                step = 1
            def g():
                for i in range(start, stop, step):
                    yield self.read_one(i)
            return g()
        elif isinstance(data_i, int):
            # 处理单个索引操作
            return self.read_one(data_i)
        else:
            raise TypeError("Unsupported data_i type")

    def __setitem__(self, data_i:int, value):
        self.write_one(data_i, value)

    def __len__(self):
        return self.data_num

    def __iter__(self):
        return self.read_from_disk()
    
    def load_data_frame(self):
        data_frame_path = os.path.join(self.directory, self.__class__.__name__ + "-" + DatasetFormat.DATAFRAME)
        if os.path.exists(data_frame_path):
            self.data_frame = pd.read_csv(data_frame_path)
            # index_names = self.data_frame.iloc[:,0]
            # self.data_frame = self.data_frame.drop(self.data_frame.columns[0], axis = 1)
            # self.data_frame.index = index_names
        else:
            # self.init_data_frame()
            # self.data_frame.loc[-1, -1] = 'Left-Top'
            self.save_data_frame()
        return self.data_frame

    def init_data_frame(self):
        data = np.zeros((self.data_i_upper, len(self.data_clusters)), np.bool_)
        cols = [x.sub_dir for x in self.data_clusters]
        self.data_frame = pd.DataFrame(data, columns=cols)

        if self.data_num != self.data_i_upper:
            for data_i in tqdm(self.data_frame.index, desc="initializing data frame"):
                self.calc_data_frame(data_i)
        else:
            self.data_frame.loc[:] = True

    def save_data_frame(self):
        self.init_data_frame()
        if not self.data_frame.empty:
            data_frame_path = os.path.join(self.directory, self.__class__.__name__ + "-" + DatasetFormat.DATAFRAME)
            self.data_frame.to_csv(data_frame_path, index=False)

    def calc_data_frame(self, data_i):
        for cluster in self.data_clusters:
            self.data_frame.loc[data_i, cluster.sub_dir] = data_i in cluster.keys()

    def clear_invalid_data_i(self):
        raise NotImplementedError

    def add_to_data_frame(self, data_i:int):
        if data_i in self.data_frame.index:
            self.calc_data_frame(data_i)
        else:
            self.data_frame.loc[data_i] = [False for _ in range(len(self.data_frame.columns))]
            self.calc_data_frame(data_i)

    def get_writing_mark_file(self):
        return os.path.join(self.directory, self.WRITING_MARK)

#### Posture ####
class PostureDatasetFormat(DatasetFormat[_DataCluster, ViewMeta]):
    def _init_clusters(self):
        self.labels_elements     = IntArrayDictElement(self, "labels", (4,), array_fmt="%8.8f")
        self.bbox_3ds_elements   = IntArrayDictElement(self, "bbox_3ds", (-1, 2), array_fmt="%8.8f") 
        self.landmarks_elements  = IntArrayDictElement(self, "landmarks", (-1, 2), array_fmt="%8.8f")
        self.extr_vecs_elements  = IntArrayDictElement(self, "trans_vecs", (2, 3), array_fmt="%8.8f")

    def read_one(self, data_i, appdir="", appname="", *arg, **kwargs):
        super().read_one(data_i, appdir, appname, *arg, **kwargs)
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

    def _write_elements(self, data_i: int, viewmeta: ViewMeta, appdir="", appname=""):
        self.labels_elements.write(data_i, viewmeta.labels, appdir=appdir, appname=appname)
        self.bbox_3ds_elements.write(data_i, viewmeta.bbox_3d, appdir=appdir, appname=appname)
        self.landmarks_elements.write(data_i, viewmeta.landmarks, appdir=appdir, appname=appname)
        self.extr_vecs_elements.write(data_i, viewmeta.extr_vecs, appdir=appdir, appname=appname)

    def calc_by_base(self, mesh_dict:dict[int, MeshMeta], overwitre = False):
        '''
        brief
        -----
        calculate data by base data, see ViewMeta.calc_by_base
        '''
        with self.writer:
            self.set_overwrite_allowed(True)
            for i in range(self.data_num):
                viewmeta = self.read_one(i)
                viewmeta.calc_by_base(mesh_dict, overwitre=overwitre)
                self.write_to_disk(viewmeta, i)
            self.set_overwrite_allowed(False)

class LinemodFormat(PostureDatasetFormat):
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
    
    class _MasksElements(Elements["LinemodFormat", dict[int, np.ndarray]]):
        def id_format(self, class_id):
            id_format = str(class_id).rjust(6, "0")
            return id_format

        def _read(self, data_i, *arg, **kwargs) -> dict[int, np.ndarray]:
            masks = {}
            appdir = ""
            for n, scene_gt in enumerate(self.dataset_node.scene_gt_dict[data_i]):
                id_ = scene_gt[LinemodFormat.KW_GT_ID]
                mask:np.ndarray = super()._read(data_i, appdir, appname=self.id_format(n))
                if mask is None:
                    continue
                masks[id_] = mask
            return masks

        def _write(self, data_i, value: dict[int, ndarray], appdir="", appname="", *arg, **kwargs):
            for n, scene_gt in enumerate(self.dataset_node.scene_gt_dict[data_i]):
                id_ = scene_gt[LinemodFormat.KW_GT_ID]
                mask = value[id_]
                super().write(data_i, mask, appname=self.id_format(n))
            return super()._write(data_i, value, appdir, appname, *arg, **kwargs)

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

    def _write_elements(self, data_i:int, viewmeta:ViewMeta):
        super()._write_elements(data_i, viewmeta)
        self.rgb_elements.  write(data_i, viewmeta.color)
        self.depth_elements.write(data_i, viewmeta.depth)
        self.masks_elements.write(data_i, viewmeta.masks)

    def _write_jsondict(self, data_i:int, viewmeta:ViewMeta):
        super()._write_jsondict(data_i, viewmeta)
        gt_one_info = []
        for obj_id, trans_vecs in viewmeta.extr_vecs.items():
            posture = Posture(rvec=trans_vecs[0], tvec=trans_vecs[1])
            gt_one_info .append(
                {   LinemodFormat.KW_GT_R: posture.rmat.reshape(-1),
                    LinemodFormat.KW_GT_t: posture.tvec.reshape(-1),
                    LinemodFormat.KW_GT_ID: int(obj_id)})
        self.scene_gt_dict.write(self.data_num, gt_one_info)

        ###
        gt_cam_one_info = {self.KW_CAM_K: viewmeta.intr.reshape(-1), self.KW_CAM_DS: viewmeta.depth_scale, self.KW_CAM_VL: 1}
        self.scene_camera_dict.write(self.data_num, gt_cam_one_info)

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
            bb = bbox_2d[obj_id]
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
        self.scene_gt_info_dict.write(self.data_num, gt_info_one_info)         

    def read_one(self, data_i, *arg, **kwargs):
        super().read_one(data_i, *arg, **kwargs)
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
        # visib_fract    = [x[LinemodFormat.KW_GT_INFO_VISIB_FRACT] for x in self.scene_gt_info_dict[data_i]]
        visib_fract_dict = zip_dict(ids, self.scene_gt_info_dict[data_i], 
                                         lambda obj: [x[LinemodFormat.KW_GT_INFO_VISIB_FRACT] for x in obj])
        return ViewMeta(color, depth, masks, 
                        extr_vecs_dict,
                        intr,
                        depth_scale,
                        bbox_3d,
                        landmarks,
                        visib_fract_dict)

class VocFormat(PostureDatasetFormat):
    IMGAE_DIR = "images"

    class cxcywhLabelElement(IntArrayDictElement):
        def __init__(self, dataset_node: Any, sub_dir: str, array_fmt: str = "", register=True, name="", filllen=6, fillchar='0') -> None:
            super().__init__(dataset_node, sub_dir, (4,), array_fmt, register, name, filllen, fillchar)
            self.image_size_required = True
            self.__trigger = False

            self.default_image_size = None

        def ignore_warning_once(self):
            self.image_size_required = False
            self.__trigger = True

        def __reset_trigger(self):
            if self.__trigger:
                self.__trigger = False
                self.image_size_required = True

        def _read_format(self, labels: np.ndarray, image_size):
            if image_size is not None:
                bbox_2d = labels[:,1:].astype(np.float32) #[cx, cy, w, h]
                bbox_2d = VocFormat._normedcxcywh_2_x1y1x2y2(bbox_2d, image_size)
                labels[:,1:] = bbox_2d   
            return labels         
        
        def _write_format(self, labels: np.ndarray, image_size):
            if image_size is not None:
                bbox_2d = labels[:,1:].astype(np.float32) #[cx, cy, w, h]
                bbox_2d = VocFormat._x1y1x2y2_2_normedcxcywh(bbox_2d, image_size)
                labels[:,1:] = bbox_2d
            return labels

        def read(self, data_i, appdir="", appname="", *arg, 
                 force = False, image_size = None, **kw)->dict[int, np.ndarray]:
            '''
            set_image_size() is supported be called before read() \n
            or the bbox_2d will not be converted from normed cxcywh to x1x2y1y2
            image_size: (w, h)
            '''
            if image_size is None:
                image_size = self.default_image_size
            if image_size is not None:
                return super().read(data_i, appdir, appname, *arg, force = force, image_size = image_size, **kw)
            else:
                if self.image_size_required:
                    warnings.warn("image_size is None, bbox_2d will not be converted from normed cxcywh to x1x2y1y2",
                                  ClusterParaWarning)
                self.__reset_trigger()
                return super().read(data_i, appdir, appname, *arg, image_size = image_size, **kw)
        
        def write(self, data_i, labels_dict:dict[int, np.ndarray], appdir="", appname="", *arg, 
                  force = False, image_size = None, **kw):
            '''
            set_image_size() is supported be called before write() \n
            or the bbox_2d will not be converted from x1x2y1y2 to normed cxcywh
            image_size: (w, h)
            '''
            if image_size is None:
                image_size = self.default_image_size
            if image_size is not None:
                return super().write(data_i, labels_dict, appdir, appname, *arg, force=force, image_size = image_size, **kw)
            else:
                if self.image_size_required:
                    warnings.warn("image_size is None, bbox_2d will not be converted from x1x2y1y2 to normed cxcywh",
                                    ClusterParaWarning)
                self.__reset_trigger()                    
                return super().write(data_i, labels_dict, appdir, appname, *arg, force=force, image_size = image_size, **kw)    

    def __init__(self, directory, split_rate = 0.75, clear = False, parent = None) -> None:
        super().__init__(directory, split_rate, clear, parent)

    def _init_clusters(self):
        super()._init_clusters()
        self.images_elements:Elements[VocFormat, np.ndarray]     = Elements(self, self.IMGAE_DIR,    
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
        self.visib_fract_elements= IntArrayDictElement(self, "visib_fracts", ())
        self.labels_elements     = self.cxcywhLabelElement(self, "labels", )

    def get_default_set(self, data_i):
        if data_i in self.default_train_idx_array:
            sub_set = VocFormat.KW_TRAIN
        elif data_i in self.default_val_idx_array:
            sub_set = VocFormat.KW_VAL
        else:
            raise ValueError("can't find datas of index: {}".format(data_i))
        return sub_set
    
    def decide_default_set(self, data_i):
        try:
            return self.get_default_set(data_i)
        except ValueError:
            spliter = self.spliter_group[self.DEFAULT_SPLIT_TYPE[0]]
            return spliter.set_one_by_rate(data_i, self.split_default_rate)

    @staticmethod
    def _x1y1x2y2_2_normedcxcywh(bbox_2d, img_size):
        '''
        bbox_2d: np.ndarray [..., (x1, x2, y1, y2)]
        img_size: (w, h)
        '''

        # Calculate center coordinates (cx, cy) and width-height (w, h) of the bounding boxes
        x1, y1, x2, y2 = np.split(bbox_2d, 4, axis=-1)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        # Normalize center coordinates and width-height by image size
        w_img, h_img = img_size
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
        img_size: (w, h)
        '''

        # Unpack the normalized bounding box coordinates
        cx, cy, w, h = np.split(bbox_2d, 4, axis=-1)

        # Denormalize the center coordinates and width-height by image size
        w_img, h_img = img_size
        x1 = (cx - w / 2) * w_img
        y1 = (cy - h / 2) * h_img
        x2 = x1 + w * w_img
        y2 = y1 + h * h_img

        # Return the bounding boxes as a new np.ndarray with shape (..., 4)
        bbox_2d = np.concatenate([x1, y1, x2, y2], axis=-1)
        return bbox_2d

    def _write_elements(self, data_i: int, viewmeta: ViewMeta):
        sub_set = self.decide_default_set(data_i)
        self.labels_elements.ignore_warning_once()
        super()._write_elements(data_i, viewmeta, appdir=sub_set)
        #
        self.images_elements.write(data_i, viewmeta.color, appdir=sub_set)
        #
        self.depth_elements.write(data_i, viewmeta.depth, appdir=sub_set)
        #
        self.masks_elements.write(data_i, viewmeta.masks, appdir=sub_set)
        
        ###
        self.labels_elements.write(data_i, viewmeta.bbox_2d, appdir=sub_set, image_size = viewmeta.color.shape[:2][::-1]) # necessary to set image_size
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
    
    def read_one(self, data_i, *arg, **kwargs) -> ViewMeta:
        # 判断data_i属于train或者val
        sub_set = self.decide_default_set(data_i)

        self.labels_elements.ignore_warning_once()
        viewmeta = super().read_one(data_i, appdir=sub_set, *arg, **kwargs)
        # 读取
        color:np.ndarray = self.images_elements.read(data_i, appdir=sub_set)
        viewmeta.set_element(ViewMeta.COLOR, color)
        #
        depth = self.depth_elements.read(data_i, appdir=sub_set)
        viewmeta.set_element(ViewMeta.DEPTH, depth)
        #
        labels_dict = self.labels_elements.read(data_i, appdir=sub_set, image_size = color.shape[:2][::-1]) # {id: [cx, cy, w, h]}
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

    def save_elements_cache(self):
        self.labels_elements.save_cache(image_size=[img.shape[:2][::-1] for img in self.images_elements])
        self.bbox_3ds_elements.save_cache()
        self.depth_scale_elements.save_cache()
        self.intr_elements.save_cache()
        self.landmarks_elements.save_cache()
        self.extr_vecs_elements.save_cache()
        self.visib_fract_elements.save_cache()

class _LinemodFormat_sub1(LinemodFormat):
    class _MasksElements(LinemodFormat._MasksElements):

        def _read(self, data_i, *arg, **kwargs) -> dict[int, ndarray]:
            masks = {}
            for n in range(100):
                mask = super().read(data_i, appname=self.id_format(n))
                if mask is None:
                    continue
                masks[n] = mask
            return masks
        
        def _write(self, data_i, value: dict[int, ndarray], appdir="", appname="", *arg, **kwargs):
            for id_, mask in value.items():
                super().write(data_i, mask, appname=self.id_format(id_))
            return super()._write(data_i, value, appdir, appname, *arg, **kwargs)

    def __init__(self, directory, clear = False) -> None:
        super().__init__(directory, clear)
        self.rgb_elements   = Elements(self, "rgb", 
                                       read_func=cv2.imread,  
                                       write_func=cv2.imwrite, 
                                       suffix='.jpg')

    def read_one(self, data_i, *arg, **kwargs):
        viewmeta = super().read_one(data_i, *arg, **kwargs)

        for k in viewmeta.bbox_3d:
            viewmeta.bbox_3d[k] = viewmeta.bbox_3d[k][:, ::-1]
        for k in viewmeta.landmarks:
            viewmeta.landmarks[k] = viewmeta.landmarks[k][:, ::-1]
        viewmeta.depth_scale *= 1000

        return viewmeta
  
class Mix_VocFormat(VocFormat):
    DEFAULT_SPLIT_TYPE = ["default", "posture", "reality", "basis"]
    MODE_DETECTION = 0
    MODE_POSTURE = 1

    IMGAE_DIR = "images"

    def __init__(self, directory, split_rate=0.75, clear=False, parent = None) -> None:
        super().__init__(directory, split_rate, clear, parent)

        self.split_mode = self.DEFAULT_SPLIT_TYPE[1]
        self.spliter_group[self.DEFAULT_SPLIT_TYPE[0]].split_for = Spliter.SPLIT_FOR_TRAINING
        self.spliter_group[self.DEFAULT_SPLIT_TYPE[1]].split_for = Spliter.SPLIT_FOR_TRAINING
        self.spliter_group[self.DEFAULT_SPLIT_TYPE[2]].split_for = Spliter.SPLIT_FOR_DATATYPE
        self.spliter_group[self.DEFAULT_SPLIT_TYPE[3]].split_for = Spliter.SPLIT_FOR_DATATYPE

        self.spliter_group[self.DEFAULT_SPLIT_TYPE[2]]._default_subtypes = ["real", "sim"]
        self.spliter_group[self.DEFAULT_SPLIT_TYPE[3]]._default_subtypes = ["basic", "augment"]

    @property
    def posture_train_idx_list(self):
        return self.spliter_group[self.DEFAULT_SPLIT_TYPE[1]][self.KW_TRAIN]
    
    @property
    def posture_val_idx_list(self):
        return self.spliter_group[self.DEFAULT_SPLIT_TYPE[1]][self.KW_VAL]

    @property
    def default_spliter(self):
        return self.spliter_group[self.DEFAULT_SPLIT_TYPE[0]]
    
    @property
    def posture_spliter(self):
        return self.spliter_group[self.DEFAULT_SPLIT_TYPE[1]]
    
    @property
    def reality_spliter(self):
        return self.spliter_group[self.DEFAULT_SPLIT_TYPE[2]]
    
    @property
    def basis_spliter(self):
        return self.spliter_group[self.DEFAULT_SPLIT_TYPE[3]]

    def _init_clusters(self):
        super()._init_clusters()

    def gen_posture_log(self, ratio = 0.85):
        """
        Only take ratio of the real data as the verification set
        """
        assert self.reality_spliter.total_num == self.data_num, "reality_spliter.total_num != data_num"
        assert self.basis_spliter.total_num == self.data_num, "basis_spliter.total_num != data_num"
        
        real_idx = self.reality_spliter.get_subtype_idx_list(0).copy()
        np.random.shuffle(real_idx)
        self.posture_val_idx_list.clear()
        self.posture_val_idx_list.extend(real_idx[:int(len(real_idx)*ratio)])

        posture_train_idx_list = np.setdiff1d(
            np.union1d(self.reality_spliter.get_subtype_idx_list(0), self.reality_spliter.get_subtype_idx_list(1)),
            self.posture_val_idx_list
            ).astype(np.int32).tolist()
        self.posture_train_idx_list.clear()
        self.posture_train_idx_list.extend(posture_train_idx_list)

        with self.posture_spliter.writer:
            self.posture_spliter.save()
            

    def record_data_type(self, data_i, is_real, is_basic):
        self.reality_spliter.set_one(data_i, self.reality_spliter.subtypes[int(not is_real)])
        self.basis_spliter.set_one(data_i, self.basis_spliter.subtypes[int(not is_basic)])

    def get_data_type_as_bool(self, data_i):
        types = self.get_data_type(data_i)
        for k, v in types.items():
            types[k] = (v == self.spliter_group[k].subtypes[0])
        return types

class Spliter(FileCluster["SpliterGroup", list[int]]):
    SPLIT_FOR_TRAINING = 0
    SPLIT_FOR_DATATYPE = 1

    SPLIT_INFO_FILE = "__split_for.txt"

    def __init__(self, dataset_node:"SpliterGroup", sub_dir:str, split_for=None, default_subtypes:list = [], register = True, name = "") -> None:
        self.split_info_file = SingleFile[tuple[int, list[str]]](
            self.SPLIT_INFO_FILE, 
            self.load_split_info_func, 
            self.save_split_info_func)
        super().__init__(dataset_node, sub_dir, register, name, singlefile_list=[self.split_info_file])
        ### init split_for
        if split_for is None:
            if self.split_info_file.exist:
                self.split_for, subs = self.split_info_file.read()
            else:
                self.split_for = self.SPLIT_FOR_TRAINING
                subs = []

        ### init default_subtypes
        if self.split_for == self.SPLIT_FOR_TRAINING:
            default_subtypes = [VocFormat.KW_TRAIN, VocFormat.KW_VAL]

        subtypes = [os.path.splitext(sub)[0] for sub in subs if sub != self.SPLIT_INFO_FILE]
        self.__default_subtypes:list[str] = remove_duplicates(default_subtypes + subtypes)

        ### init subtypes_files
        for file_mainname in self.__default_subtypes:
            sf = SingleFile(file_mainname + ".txt", self.loadsplittxt_func, self.savesplittxt_func)
            self.update_file(sf)

        self.__exclusive = True
        self.load()

    @property 
    def split_mode(self):
        return os.path.split(self.sub_dir)[-1]

    @property
    def exclusive(self):
        return self.__exclusive
    
    @exclusive.setter
    def exclusive(self, value):
        if self.split_for == self.SPLIT_FOR_TRAINING:
            self.__exclusive = True
        else:
            self.__exclusive = bool(value)

    @property
    def subtypes(self):
        return tuple(self.split.keys())
    
    @property
    def subtype_fileobjs_dict(self) -> dict[str, SingleFile[list[int]]]:
        objs_dict = {}
        for k in self.fileobjs_dict.keys():
            if self.SPLIT_INFO_FILE in k:
                continue
            k = get_mainname(k)
            objs_dict.update({k: self.fileobjs_dict[self.filter_data_i(k)]})
        return objs_dict

    @property
    def _default_subtypes(self):
        return self.__default_subtypes
    
    @_default_subtypes.setter
    def _default_subtypes(self, value:Union[str, Iterable[str]]):
        if isinstance(value, str):
            value = [value]
        if isinstance(value, Iterable):
            self.__default_subtypes = tuple(remove_duplicates(value))
        for t in value:
            if t not in self.split:
                # 
                self.split[t] = []
                # file
                sf = SingleFile(t + ".txt", self.loadsplittxt_func, self.savesplittxt_func)
                self.update_file(sf)
        for t in list(self.split.keys()):
            if t not in value:
                self.split.pop(t)
                self.remove_file(t)
        self.load()

    @property
    def total_num(self):
        return sum([len(v) for v in self.split.values()])

    def get_subtype_idx_list(self, subtype:Union[str, int]):
        if isinstance(subtype, str):
            return self.split[subtype]
        elif isinstance(subtype, int):
            return self.split[self.subtypes[subtype]]
        else:
            raise ValueError("subtype must be str or int")

    def get_file_path(self, subtype):
        return os.path.join(self.directory, subtype + ".txt")
    
    def load_split_info_func(self, path:str):
        with open(path, 'r') as file:
            lines = file.readlines()

        # parse split_for
        try:
            split_for = int(lines[0].strip())
        except ValueError:
            return None, []

        # extract strings
        strings = [line.strip() for line in lines[1:]]

        return split_for, strings

    def save_split_info_func(self, path:str, value:list):
        split_for, strings = value
        with open(path, 'w') as file:
            file.write(str(split_for) + '\n')  # 写入整数值并换行
            for string in strings:
                file.write(string + '\n')  # 逐行写入字符串


    def loadsplittxt_func(self, path:str):
        if os.path.exists(path):
            with warnings.catch_warnings():
                rlt = np.loadtxt(path).astype(np.int32).reshape(-1).tolist()
            return rlt
        else:
            return []
        
    def savesplittxt_func(self, path:str, value:Iterable):
        assert isinstance(value, Iterable), "split_dict value must be a Iterable"
        np.savetxt(path, np.array(value).astype(np.int32).reshape(-1, 1), fmt="%6d")

    def load(self):
        self.split:dict[str, list[int]] = {}
        for st, sf in self.subtype_fileobjs_dict.items():
            self.split[st] = self.read(st)
    
    def save(self):
        with self.writer.allow_overwriting():
            os.makedirs(self.directory, exist_ok=True)
            for st, sf in self.subtype_fileobjs_dict.items():
                self.write(st, self.split[st])
            self.split_info_file.write((self.split_for, self.subtypes))
        # for subtype in self.subtypes:
        #     file_path = self.get_file_path(subtype)
        #     self.savesplittxt_func(file_path, self.split[subtype])
        # self.savesplittxt_func(os.path.join(self.directory, self.SPLIT_FOR_FILE), [self.split_for])

    def _check_split_rate(self, split_rate:Union[float, Iterable[float], dict[str, float]]):
        assert len(self.subtypes) > 1, "len(subtypes) must > 1"
        if len(self.subtypes) == 2 and isinstance(split_rate, float):
            split_rate = (split_rate, 1 - split_rate)
        if isinstance(split_rate, dict):
            split_rate = tuple([split_rate[subtype] for subtype in self.subtypes])
        elif isinstance(split_rate, Iterable):
            split_rate = tuple(split_rate)
        else:
            raise ValueError("split_rate must be Iterable or dict[str, float], (or float if len(subtypes) == 2)")
        assert len(split_rate) == len(self.subtypes), "splite_rate must have {} elements".format(len(self.subtypes))
        return split_rate

    def gen(self, data_num:int, split_rate:Union[float, Iterable[float], dict[str, float]]):
        assert isinstance(data_num, int), "data_num must be int"
        split_rate = self._check_split_rate(split_rate)
        _s = np.arange(data_num, dtype=np.int32)
        np.random.shuffle(_s)
        for subtype, rate in zip(self.subtypes, split_rate):
            num = int(data_num * rate)
            self.split[subtype] = _s[:num]
            _s = _s[num:]
            
    def set_one(self, data_i, subtype, sort = False):
        if data_i not in self.split[subtype]:
            self.split[subtype].append(data_i)
        if self.exclusive:
            # remove from other subtypes
            for _subtype in self.subtypes:
                if _subtype != subtype and data_i in self.split[_subtype]:
                    self.split[_subtype].remove(data_i)
        if sort:
            self.split[subtype].sort()

    def set_one_by_rate(self, data_i, split_rate):
        split_rate = self._check_split_rate(split_rate)
        total_nums = [len(st) for st in self.split.values()]
        if sum(total_nums) == 0:
            # all empty, choose the first
            subtype_idx = 0
        else:
            rates = np.array(total_nums) / sum(total_nums)
            subtype_idx = 0
            for idx, r in enumerate(rates):
                if r <= split_rate[idx]:
                    subtype_idx = idx
                    break
        self.set_one(data_i, self.subtypes[subtype_idx])
        return self.subtypes[subtype_idx]

    def sort_all(self):
        for subtype in self.subtypes:
            self.split[subtype].sort()
    
    def get_one_subtype(self, data_i):
        subtypes = []
        for subtype in self.subtypes:
            if data_i in self.split[subtype]:
                subtypes.append(subtype)
        return subtypes

    def one_is_subtype(self, data_i, subtype):
        assert subtype in self.subtypes, "subtype must be one of {}".format(self.subtypes)
        return data_i in self.split[subtype]
    
    def __getitem__(self, key):
        return self.get_subtype_idx_list(key)

class SpliterGroup(DatasetNode[Spliter]):
    DEFAULT_SPLIT_TYPE = ["default"]

    def __init__(self, directory, split_mode_list:list, parent = None) -> None:
        self.__split_mode_list = split_mode_list
        super().__init__(directory, parent)
        self.__split_mode:str = self.DEFAULT_SPLIT_TYPE[0]

    def _init_clusters(self):
        self.__split_mode_list = remove_duplicates(self.__split_mode_list + os.listdir(self.split_subdir))
        _spliter_list:list = [Spliter(self, m, register=True) for m in self.__split_mode_list]
        return super()._init_clusters()

    @property
    def split_subdir(self):
        return self.sub_dir
    
    @property
    def split_mode_list(self):
        # mode_list = []
        # for key, spliter in self.cluster_map.items():
        #     _, _, _, sub_dir, name = spliter.parse_identity_string(key)
        #     mode_list.append(get_mainname(path))
        # return tuple(mode_list)
        return tuple(self.cluster_map.get_keywords())

    @property
    def split_mode(self):
        return self.__split_mode
    
    @split_mode.setter
    def split_mode(self, split_type:str):
        self.set_split_mode(split_type)
    
    @property
    def cur_training_spliter(self):
        spliter = self[self.split_mode]
        assert spliter.split_for == Spliter.SPLIT_FOR_TRAINING, "cur_training_spliter must be Spliter.SPLIT_FOR_TRAINING"
        return spliter

    def set_split_mode(self, split_type:str):
        assert split_type in self.split_mode_list, "split_type must be one of {}".format(self.split_mode_list)
        self.__split_mode = split_type

    def get_split_array_of(self, split_type, sub_set:str = ""):
        if sub_set == "":
            dict_:dict[str, list] = self[split_type]
            # convert as dict[str, tuple]
            dict_ = {k:tuple(v) for k, v in dict_.items()}
            return dict_
        else:
            return tuple(self[split_type][sub_set])

    def _query(self, split_for:int, data_i:int) -> dict[str, list[str]]:
        set_ = {}
        for s in self.cluster_map.values():
            if s.split_for == split_for:
                set_.update({s.split_mode: s.get_one_subtype(data_i)})
        return set_

    def query_datatype_set(self, data_i:int) -> dict[str, list[str]]:
        return self._query(Spliter.SPLIT_FOR_DATATYPE, data_i)

    def query_training_set(self, data_i) -> dict[str, list[str]]:
        return self._query(Spliter.SPLIT_FOR_TRAINING, data_i)

    def save(self):
        for spliter in self.cluster_map.values():
            spliter.save()
    
    def __getitem__(self, key):
        return self.cluster_map.search(key)



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

def remove_duplicates(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def zip_dict(ids:list[int], item:Union[Iterable, None, Any], func = lambda x: x):
    if item:
        processed = func(item)
        return as_dict(ids, processed)
    else:
        return None
    
def get_mainname(path):
    return os.path.splitext(os.path.split(path)[1])[0]

KT = TypeVar("KT")
VT = TypeVar("VT")
def search_in_dict(_dict:dict[KT, VT], key:Union[int, str], return_index=False, process_func:Callable = None):
    def default_func(key, string):
        return key in string 

    if process_func is None:
        process_func:Callable = default_func

    if isinstance(key, int):
        value = list(_dict.values())[key]
        return key if return_index else value
    elif isinstance(key, str):
        if key in _dict:
            return list(_dict.keys()).index(key) if return_index else _dict[key]
        else:
            matching_keys = [k for k in _dict if process_func(key, k)]
            if len(matching_keys) == 1:
                matching_key = matching_keys[0]
                return list(_dict.keys()).index(matching_key) if return_index else _dict[matching_key]
            else:
                raise ValueError("Key not found or ambiguous")
    else:
        raise TypeError("Key must be an int or str")
    
