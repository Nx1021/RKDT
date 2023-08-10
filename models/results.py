from torch import Tensor
import torch
from .utils import denormalize_bbox, normalize_bbox, tensor_to_numpy

from posture_6d.posture import Posture
from posture_6d.intr import CameraIntr

import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Union, TypeVar, Generator, Iterable, Generic, Type

PLE = TypeVar('PLE')
PL = TypeVar('PL', bound='PairedLists')
NT = TypeVar('NT', bound='NestedTensor')

TensorLike = Union[Tensor, list[Tensor], np.ndarray, None]
 

class PairedLists(Generic[PLE, PL]):
    def __init__(self, *objs, names:Iterable[str]=None) -> None:
        if names is not None:
            names = list(names)
            assert len(objs) == len(names), f"len(objs) must be equal to len(names), but got {len(objs)} and {len(names)}"
        else:
            names = [None for i in range(len(objs))]
        self.__names:list[str] = []
        self.__objs :list[PLE] = []
        self.__inited = True        
        for name, obj in zip(names, objs):
            self.__setattr__(name, obj)

    @property
    def inited(self):
        try:
            self.__inited
            return True
        except:
            return False

    @property
    def empty(self):
        return len(self.__objs) == 0

    def traverse_names(self)->list[str]:
        def g():
            for name in self.__names:
                yield name
        return g()
    
    def traverse_objs(self)->list[PLE]:
        def g():
            for name in self.__objs:
                yield name
        return g()

    def query_name(self, obj:PLE)->Any:
        try: 
            idx = self.query_name_idx(obj)
            return self.__names[idx]
        except:
            return None
        
    def query_obj(self, name:str)->Any:
        try: 
            idx = self.query_name_idx(name)
            return self.__objs[idx]
        except:
            return None
        
    def query_obj_idx(self, obj:PLE)->Any:
        assert obj is not None, "obj cannot be None"
        values_id = list([id(x) for x in self.__objs])
        return values_id.index(id(obj))
    
    def query_name_idx(self, name:str)->Any:
        assert name is not None, "name cannot be None"
        return self.__names.index(name)
        
    def has_name(self, name:str)->bool:
        return hasattr(self, name)
    
    def has_obj(self, obj:PLE)->bool:
        try:
            self.query_obj_idx(obj)
            return True
        except ValueError:
            return False

    def get_name(self, idx):
        return self.get_items(idx)[0]
    
    def get_obj(self, idx):
        return self.get_items(idx)[1]

    def __iter__(self:PL):
        def g():
            for i in range(len(self)):
                yield self.get_items(i)
        return g()

    def get_items(self, index:Union[int, str])->tuple[str, PLE]:
        if isinstance(index, str):
            assert self.has_name(index), f"Cannot find {index} in {self.__names}"
            name = index
            index = self.query_name_idx(name)
        elif isinstance(index, int):
            name = self.__names[index]
        else:
            raise TypeError(f"index must be str or int, but got {type(index)}")
        return name, self.__objs[index]
    
    def set_item(self, index:Union[int, str], value:tuple[str, PLE]):
        if isinstance(index, str):
            assert self.has_name(index), f"Cannot find {index} in {self.__names}"
            name = index
        elif isinstance(index, int):
            name = self.__names[index]
        else:
            raise TypeError(f"index must be str or int, but got {type(index)}")
        name = self.__names[index]
        self.__setattr__(name, value)

    def __setattr__(self, _name: str, _value: PLE) -> None:
        # assert isinstance(__name, str), f"__name must be str, but got {type(__name)}"
        if self.inited:
            if isinstance(_name, str):
                if self.has_name(_name):
                    # modify
                    idx = self.__objs.index(self.query_obj(_name))
                    self.__objs[idx] = _value
                # elif self.query_name(__value) is not None: 
                #     # no allow to set if tensor_like is already in NestedTensor and has a name
                #     raise ValueError("Cannot set if tensor_like is already in NestedTensor")
                else:
                    self.__names.append(_name)                
                    self.__objs.append(_value)
                super().__setattr__(_name, _value)
            elif _name is None:
                assert not self.has_obj(_value), "Cannot set if tensor_like is already in __objs"
                self.__names.append(_name)                
                self.__objs.append(_value)
        else:
            super().__setattr__(_name, _value)

    def __delattr__(self, __name: Union[str, int]) -> None:
        if isinstance(__name, int):
            index = __name
            assert index < len(self.__names), f"index {index} out of range {len(self.__names)}"
        elif isinstance(__name, str):
            assert self.has_name(__name), f"Cannot find {__name} in {self.__names}"
            index = self.query_name_idx(__name)
        else:
            raise TypeError(f"__name must be str or int, but got {type(__name)}")
        name = self.__names.pop(index)
        obj = self.__objs.pop(index)
        if isinstance(name, str):
            super().__delattr__(name)
        return name, obj

    def append(self, value:PLE, name:str=None):
        self.__setattr__(name, value)

    def pop(self, index:int = -1):
        return self.__delattr__(index)

    def clear(self):
        while not self.empty:
            self.pop()

    @classmethod
    def combine(cls:Type[PL], PL_list:list[PL])->PL:
        assert len(PL_list) > 0, "Empty NestedTensor"
        # all of the names in nested_tensor_list must be unique
        all_names = [name for nt in PL_list for name in nt.traverse_names()]
        all_names = [name for name in all_names if name is not None]
        assert len(set(all_names)) == len(all_names), \
            "all of the str names in nested_tensor_list must be unique"
        new_list:list[TensorLike]   = []
        new_names:list[str]         = []
        for nt in PL_list:
            new_list  += nt.__objs
            new_names += nt.__names
        new_nested_tensor = cls(*new_list, names=new_names)

        return new_nested_tensor

class NestedTensor(PairedLists[TensorLike, "NestedTensor"]):
    class NoneTensor():
        def __init__(self):
            pass

        def to(self, device):
            return NestedTensor.NoneTensor()

    def __init__(self, *tensor_like_list:TensorLike, names:list[str]=None):
        super().__init__(*tensor_like_list, names=names)

    @property
    def valid(self):
        return not self.empty and \
            not isinstance(self.get_obj(0), self.NoneTensor) and \
            not isinstance(self.get_obj(0), np.ndarray)

    @property
    def tensor_num(self):
        return len(list(self.traverse_objs()))

    @property
    def device(self):
        if self.tensor_num == 0:
            return None
        obj_0:TensorLike = self.get_obj(0)
        if isinstance(obj_0, Tensor):
            return obj_0.device
        elif isinstance(obj_0, list):
            return obj_0[0].device

    @property
    def min_depth(self)->int:
        min_depth = []
        for tensorlike in self.traverse_objs():
            if isinstance(tensorlike, (Tensor, np.ndarray)):
                min_depth.append(len(tensorlike.shape))
            elif isinstance(tensorlike, list):
                min_depth.append(min([len(t.shape) for t in tensorlike]) + 1)
        return min(min_depth)

    def to(self:NT, device)->NT:
        # type: (torch.Device) -> NestedTensor # noqa
        casted_tensors = []
        for t in self.traverse_objs():
            if isinstance(t, (Tensor, self.NoneTensor)):
                casted = t.to(device)
            elif isinstance(t, list):
                casted = [x.to(device) for x in t]
            casted_tensors.append(casted)
        return self.__class__(*casted_tensors, names=self.traverse_names())

    def boardcast(self, tensor_like:Iterable, dtype = torch.float32):
        assert self.valid, "NestedTensor is not valid"
        assert isinstance(tensor_like, Iterable), "tensor_like must be Iterable"
        tensor_like = torch.Tensor(tensor_like).to(dtype).to(self.device)
        if len(tensor_like) != len(self):
            tensor_like = tensor_like.unsqueeze(0)
            tensor_like = tensor_like.expand(len(self), -1)
        return tensor_like

    def _check_tensor_like(self, tensor_like):
        assert isinstance(tensor_like, (Tensor, list, np.ndarray, self.NoneTensor)), "tensor_like must be Tensor, list or None"
        if isinstance(tensor_like, (Tensor, list, np.ndarray)) and self.valid:
            assert len(tensor_like) == len(self), "tensor_like must have the same batch size as nestedtensor"
            assert not self.has_obj(tensor_like), "tensor_like already in nestedtensor"
        if isinstance(tensor_like, list):
            assert all([isinstance(t, Tensor) for t in tensor_like]), "all elements in tensor_like must be Tensor if tensor_like is list"

    def __cvt_to_NoneTensor(self, input):
        if input is None:
            input = self.NoneTensor()
        return input
    
    def __cvt_to_None(self, input):
        if isinstance(input, self.NoneTensor):
            input = None
        return input

    def __iter__(self:NT):
        '''
        it is totally different from it in its super class
        '''
        def g() -> Generator[NT, Any, None]:
            for i in range(len(self)):
                yield self[i]
        return g()

    def __len__(self):
        if self.valid:
            return len(self.get_obj(0))
        else:
            return 0

    def __getitem__(self:NT, index) -> NT:
        tensors = []
        for t in self.traverse_objs():
            if isinstance(t, Tensor):
                tensors.append(t[index].unsqueeze(0))
            elif isinstance(t, list):
                x:Tensor = t[index]
                tensors.append(x.unsqueeze(0))
            elif isinstance(t, np.ndarray):
                tensors.append(np.expand_dims(t[index], 0))
            else:
                tensors.append(None)
        return self.__class__(*tensors, names=self.traverse_names())

    def __repr__(self):
        return self.__class__.__name__ + ':' + str([type(t).__name__ for t in self.traverse_objs()])
    
    def __iadd__(self:NT, nested:NT):
        self.add(nested)
        return self

    def __setattr__(self:NT, name:str, tensor_like:TensorLike):
        if self.inited:
            tensor_like = self.__cvt_to_NoneTensor(tensor_like)
            self._check_tensor_like(tensor_like)
        super().__setattr__(name, tensor_like)


    def add(self:NT, nested:NT):
        assert isinstance(nested, NestedTensor), "nested must be NestedTensor"
        new_nested = NestedTensor.cat([self, nested])
        self.copy_from(new_nested)

    def copy_from(self, nested:"NestedTensor"):
        self.clear()
        for n, t in zip(nested.traverse_names(), nested.traverse_objs()):
            self.__setattr__(n, t)

    def squeeze(self:NT)->NT:
        assert self.valid and len(self) == 1 and self.min_depth > 1, "NestedTensor must have only one tensor"
        squeezed = []
        for t in self.traverse_objs():
            if isinstance(t, Tensor):
                squeezed.append(t.squeeze(0))
            elif isinstance(t, list):
                squeezed.append(t[0])
            elif isinstance(t, np.ndarray):
                squeezed.append(t.squeeze(0))
            else:
                squeezed.append(None)
        return self.__class__(*squeezed, names=self.traverse_names())

    @staticmethod
    def cat(nested_tensor_list:list[NT])->NT:
        assert len(nested_tensor_list) > 0, "Empty NestedTensor"
        nested_tensor_list = [x for x in nested_tensor_list if x.valid] # remove invalid NestedTensor
        assert all([nt.tensor_num == nested_tensor_list[0].tensor_num for nt in nested_tensor_list]), \
            "all NestedTensor must have the same tensor_num"
        tensors_list:list[list[TensorLike]] = []
        for i in range(nested_tensor_list[0].tensor_num):
            tensors_list.append([])
            for nt in nested_tensor_list:
                tensors_list[i].append(nt.get_obj(i))
            # the type of the elements in tensors_list[i] must be the same
            if any([type(t) != type(tensors_list[i][0]) for t in tensors_list[i]]):
                raise TypeError("the type of the elements in tensors_list[i] must be the same")
        cated:list[TensorLike] = []
        for tensors in tensors_list:
            if isinstance(tensors[0], Tensor):
                cated.append(torch.cat(tensors))
            elif isinstance(tensors[0], list):
                cated.append([item for sublist in tensors for item in sublist]) # list[list[Tensor]] to list[Tensor]
            elif isinstance(tensors[0], np.ndarray):
                cated.append(np.concatenate(tensors))
            else:
                cated.append(None)

        return nested_tensor_list[0].__class__(*cated, names=nested_tensor_list[0].traverse_names())

    @classmethod
    def combine(cls:Type[NT], nested_tensor_list:list[NT])->NT:
        nested_tensor_list = [x for x in nested_tensor_list if x.valid] # remove invalid NestedTensor
        assert all([len(nt) == len(nested_tensor_list[0]) for nt in nested_tensor_list]), \
            "all NestedTensor must have the same length"
        return super(NestedTensor, cls).combine(nested_tensor_list)

class GtResult(NestedTensor):
    def __init__(self, *tensors,
                 gt_class_ids:TensorLike = None,                  
                 gt_landmarks:TensorLike = None, 
                 gt_bboxes_n:TensorLike = None, 
                 gt_trans_vecs:TensorLike = None,
                 batch_idx:TensorLike = None,
                 intr_M:np.ndarray = None,
                 names:list[str] = None):
        # create attributes
        if len(tensors) > 0:
            super().__init__(*tensors, names=names)
        else:
            super().__init__()
            self.gt_class_ids  = gt_class_ids        
            self.gt_landmarks  = gt_landmarks 
            self.gt_bboxes_n   = gt_bboxes_n
            self.gt_trans_vecs = gt_trans_vecs
            self.gt_batch_idx  = batch_idx  
            self.intr_M        = intr_M
                  
class PredResult(NestedTensor):
    def __init__(self, *tensors,
                pred_landmarks_coord = None,
                pred_landmarks_probs = None,
                pred_class = None,                
                pred_bboxes_n = None,
                pred_batch_num = None,
                input_size = None, names:list[str] = None):
        # create attributes
        if len(tensors) > 0:
            super().__init__(*tensors, names=names)
        else:
            super().__init__()
            self.pred_landmarks_coord:TensorLike  = pred_landmarks_coord
            self.pred_landmarks_probs:TensorLike  = pred_landmarks_probs
            self.pred_bboxes_n:TensorLike         = pred_bboxes_n
            self.pred_batch_idx:TensorLike        = pred_batch_num
            self.pred_class:TensorLike            = pred_class
            self.input_size:TensorLike            = input_size

class MatchedRoi(GtResult, PredResult):
    def __init__(self, *tensorlikes, names:list[str] = None):
        NestedTensor.__init__(self, *tensorlikes, names = names)

    @staticmethod
    def create(gtresult: GtResult, predsult:PredResult)->"MatchedRoi":
        return MatchedRoi.combine([gtresult, predsult])

class RoiFeatureMapWithMask(NestedTensor):
    def __init__(self, *tensor_like_list: TensorLike, 
                 feature_maps:torch.Tensor = None, 
                 masks:torch.Tensor = None,
                 names: list[str] = None):
        if len(tensor_like_list) > 0:
            super().__init__(*tensor_like_list, names=names)
        else:
            super().__init__()
            self.rois_feature_map = feature_maps
            self.masks = masks

class LandmarkDetectionResult():
    def __init__(self, bbox_n:Tensor, class_id:int, landmarks_n:Tensor, landmarks_probs:Tensor, image_size:Tensor) -> None:
        '''
        parameter
        -----
        bbox_n          : normalized (x1, y1, x2, y2)
        class_id        : int
        landmarks_n     : normalized (ldmk_num, (x, y))
        image_size        : tuple|Tensor  (w, h)
        '''
        self.bbox_n         = bbox_n      # normalized (x1, y1, x2, y2)
        self.class_id       = class_id
        self.landmarks_n    = landmarks_n # normalized (ldmk_num, (x, y))
        self.landmarks_probs = landmarks_probs
        if not isinstance(image_size, Tensor):
            image_size = Tensor(image_size).to(bbox_n.device).to(torch.int32)

        self.image_size       = image_size    # pixel (w, h)

    @property
    def bbox(self):
        # with torch.no_grad():
        bbox = denormalize_bbox(self.bbox_n.unsqueeze(0), self.image_size)
        bbox = bbox.squeeze(0)
        return bbox

    @property
    def landmarks(self):
        # with torch.no_grad():
        bbox = self.bbox
        wh = torch.Tensor((bbox[2] - bbox[0], bbox[3] - bbox[1])).to(bbox.device)
        lt = torch.Tensor((bbox[0], bbox[1])).to(bbox.device)
        landmarks_box = self.landmarks_n * wh
        landmarks = landmarks_box + lt
        return landmarks
    
class ObjPosture():
    def __init__(self, landmarks, bbox_n, class_id, image_size, trans_vecs=None):
        """
        初始化ObjPosture对象。

        参数:
            landmarks: array，表示物体的关键点坐标，形状为 [ldmk_num, 2]，其中 ldmk_num 为关键点的数量，每个关键点有2个坐标值。
            bbox_n: 相对于图像大小归一化的边界框坐标，以列表形式表示 [(x1, y1, x2, y2)]。
            class_id: int，表示物体的类别标识。
            trans_vecs: 可选参数，表示平移向量。如果提供，应为包含 rvec 和 tvec 值的元组。如果未提供，默认值为 None。
        """
        self.landmarks = landmarks
        self.bbox_n:np.ndarray = bbox_n
        self.class_id: int = class_id
        if trans_vecs is not None:
            rvec, tvec = trans_vecs
            self.posture = Posture(rvec=rvec, tvec=tvec)
        else:
            self.posture = None
        self.image_posture:ImagePosture = None

        self.image_size = image_size

    @property
    def bbox(self):
        # with torch.no_grad():
        bbox = denormalize_bbox(np.expand_dims(self.bbox_n, 0), self.image_size)
        bbox = bbox.squeeze(0)
        return bbox

    @property
    def landmarks_n(self):
        # with torch.no_grad():
        bbox = self.bbox
        wh = np.array([bbox[2] - bbox[0], bbox[3] - bbox[1]])
        lt = np.array([bbox[0], bbox[1]])
        return (self.landmarks - lt) / wh

    @property
    def intr_M(self):
        return self.image_posture.intr_M

    @property
    def rvec(self):
        """
        获取物体姿态的旋转向量。

        返回:
            如果姿态有效，则返回旋转向量 rvec，否则返回 None。
        """
        if self.valid:
            return self.posture.rvec
        else:
            return None

    @property
    def tvec(self):
        """
        获取物体姿态的平移向量。

        返回:
            如果姿态有效，则返回平移向量 tvec，否则返回 None。
        """
        if self.valid:
            return self.posture.tvec
        else:
            return None

    @property
    def valid(self):
        """
        检查姿态是否有效。

        返回:
            如果姿态有效，则返回 True，否则返回 False。
        """
        return bool(self.posture)

    def plot(self, color = "black"):
        """
        在一个新的图形窗口中绘制物体的关键点和边界框，并以文字形式注释物体的类别标识。

        使用matplotlib库的plt.scatter和plt.text函数绘制关键点和文字注释。

        注意：在调用该方法之前，请确保已创建一个matplotlib的Figure和Axes对象，或者调用plt.subplots()创建。

        示例用法：
        fig, ax = plt.subplots()
        obj_posture.plot(ax)
        plt.show()
        """
        if self.landmarks is not None:
            # 绘制关键点
            for i, (x, y) in enumerate(self.landmarks):
                if x == 0 and y == 0:
                    continue
                plt.scatter(x, y, color=color, s = 4)
                plt.text(x + 1, y + 1, str(i), color=color, fontsize=4, va='top')

        if self.bbox_n is not None:
            # 绘制边界框
            bbox = denormalize_bbox(self.bbox_n, self.image_size)
            if isinstance(bbox, torch.Tensor):
                bbox = tensor_to_numpy(bbox)
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            rect = plt.Rectangle((x1, y1), width, height, fill=False, edgecolor=color)
            plt.gca().add_patch(rect)

        if self.class_id is not None:
            # 注释物体类别标识
            x = x1
            y = y1 - 8  # 文字注释位置在边界框上方
            plt.text(x, y, str(self.class_id), color=color, fontsize=8, fontweight='bold')

        plt.axis('scaled')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Object Posture')

class ImagePosture():
    def __init__(self, image: np.ndarray, *objs: ObjPosture, intr_M:np.ndarray = None) -> None:
        """
        初始化ImagePosture对象。

        参数:
            image: np.ndarray，表示图像的数组。
            *objs: 可变参数，表示ObjPosture对象的列表，用于初始化ImagePosture对象中的obj_list属性。
        """
        self.obj_list: list[ObjPosture] = list(objs)
        self.image = image
        self.intr_M = intr_M

    @property
    def image_size(self):
        """
        获取图像的尺寸。

        返回:
            图像的尺寸，形式为 (width, height)。
        """
        return self.image.shape[:2][::-1]

    def append(self, obj: ObjPosture):
        """
        将ObjPosture对象添加到ImagePosture对象中。

        参数:
            obj: ObjPosture，要添加的ObjPosture对象。
        """
        self.obj_list.append(obj)
        obj.image_posture = self

    def pop(self, index: int):
        """
        从ImagePosture对象中删除指定索引的ObjPosture对象。

        参数:
            index: int，要删除的ObjPosture对象的索引。
        """
        obj = self.obj_list.pop(index)
        obj.image_posture = None

    def insert(self, index: int, obj: ObjPosture):
        """
        将ObjPosture对象插入到ImagePosture对象的指定索引位置。

        参数:
            index: int，要插入的位置的索引。
            obj: ObjPosture，要插入的ObjPosture对象。
        """
        self.obj_list.insert(index, obj)
        obj.image_posture = self

    def split(self):
        """
        将ImagePosture对象拆分为图像、关键点、类别标识、边界框和姿态向量的列表。

        返回:
            image: np.ndarray，原始图像数组。
            landmarks: list[np.ndarray]，关键点数组的列表，每个数组形状为 (ldmknum, 2)。
            class_ids: list[int]，类别标识的列表。
            bboxes: list[np.ndarray]，边界框数组的列表，每个数组形状为 [x1, y1, x2, y2]。
            trans_vecs: list[np.ndarray]，姿态向量数组的列表，每个数组形状为 [rvec, tvec]。
        """
        landmarks = []
        class_ids = []
        bboxes = []
        trans_vecs = []
        for obj_i in range(len(self.obj_list)):
            landmarks.append(self.obj_list[obj_i].landmarks)
            class_ids.append(self.obj_list[obj_i].class_id)
            bboxes.append(self.obj_list[obj_i].bbox_n)
            trans_vecs.append((self.obj_list[obj_i].rvec, self.obj_list[obj_i].tvec))
        return self.image, landmarks, class_ids, bboxes, trans_vecs

    def plot(self, color = "black", show_image = True):
        """
        绘制ImagePosture对象的图像和其中的ObjPosture对象。

        首先绘制图像，然后遍历obj_list中的每个ObjPosture对象，调用其plot方法进行绘制。

        * 示例用法：
        
        image_posture.plot()

        plt.show()
        """
        # 绘制图像
        if show_image:
            plt.imshow(self.image)
            plt.axis('off')

        # 绘制每个ObjPosture对象
        for obj in self.obj_list:
            obj.plot(color)

def compare_image_posture(gt:ImagePosture, pred:ImagePosture):
    gt_color = "lawngreen"
    pred_color = "lightslategray"
    gt.plot(gt_color)
    pred.plot(pred_color, False)


def NestedTensor_test():
    nt1 = NestedTensor(torch.rand(3, 4), torch.rand(3, 4, 5), [torch.rand(4, 6), torch.rand(4, 5), torch.rand(4, 7)], None)
    nt2 = NestedTensor()
    nt3 = NestedTensor(torch.rand(3, 4), torch.rand(3, 4, 5), [torch.rand(4, 6), torch.rand(4, 5), torch.rand(4, 7)], None)
    try:
        nt3 = NestedTensor(torch.rand(3, 4), torch.rand(3, 4, 5), [torch.rand(4, 6), torch.rand(4, 5), torch.rand(4, 7)], torch.rand(4, 4))
    except:
        print("NestedTensor init error!")

    nt4 = NestedTensor(torch.rand(3, 4), torch.rand(3, 4, 5), [torch.rand(4, 6), torch.rand(4, 5), torch.rand(4, 7)])
    print(nt4.min_depth)
    nt4[0].squeeze()

    nt1.add(nt2)
    nt1.add(nt3)

    nt1.a = torch.rand(6, 4)
    nt2.a = torch.rand(6, 4)
    nt1.get_items(0)
    nt1.get_items("a")
    nt1.pop("a")
    nt1.pop(0)
    nt1.clear()
    NestedTensor.combine([nt1, nt2])
    nt4 = NestedTensor.combine([nt1, nt3])
    nt4.valid
    nt3.tensor_num
    nt3.device
    nt3.min_depth
    nt3.to("cuda")
    nt3.boardcast([0])
    x = nt3[0]
    x = nt3.squeeze()
    x = x.squeeze()
    NestedTensor.cat([nt3, nt4])


    g = GtResult()
    g.valid