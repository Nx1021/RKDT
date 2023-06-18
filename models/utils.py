import torch
import h5py
import re
import numpy as np
import os
import scipy
from scipy import optimize
import matplotlib.pyplot as plt
import Levenshtein
import time
from typing import OrderedDict, Union
import torch

class WeightLoader():
    STRICT     = 0
    NOT_STRICT = 1      
    CORRISPONDING = 2   # 忽略key，对应位置赋值
    HANGARIAN  = 3      # 根据名称相近程度匹配
    def __init__(self, model:torch.nn.Module) -> None:
        self.model = model

    def weights_filter(self, exclude:list[str]=[], include:list[str]=[]):
        keys = list(self.model.state_dict()) # 模型权重名
        for include_pattern in include:
            # 包含需要加载的权重名
            keys    = list(filter(lambda name: bool(re.fullmatch(include_pattern, name)),
                                            keys))
        for exclude_pattern in exclude:
            # 排除不需要加载的权重名
            keys    = list(filter(lambda name: not bool(re.fullmatch(exclude_pattern, name)),
                                            keys))
        return keys

    def load_weights_to_layar(self, pretrained_state_dict:OrderedDict[str, torch.Tensor], load_mode:int, exclude:list[str]=[], include:list[str]=[]):
        if load_mode == WeightLoader.STRICT:
            self.model.load_state_dict(pretrained_state_dict)
        else:
            # 过滤
            # to_load_keys = self.weights_filter(exclude, include)
            # to_load_pretrained_state_dict = {key: value for key, value in pretrained_state_dict.items() if key in to_load_keys}
            to_load_pretrained_state_dict = pretrained_state_dict
            # 不严格匹配名称
            if load_mode == WeightLoader.NOT_STRICT:
                self.model.load_state_dict(to_load_pretrained_state_dict, strict = False)
            # 对应位置
            elif load_mode == WeightLoader.CORRISPONDING:
                replacement_dict = {k1:k2 for k1, k2 in zip(to_load_pretrained_state_dict.keys(), self.model.state_dict().keys())}
                new_ordered_dict = OrderedDict((replacement_dict.get(key, key), value) for key, value in to_load_pretrained_state_dict.items())
                self.model.load_state_dict(new_ordered_dict)
            elif load_mode == WeightLoader.HANGARIAN:
                pass

    # def __search_match(self, layername, lw:tf.Variable, g, weight_names):
    #     try:
    #         candi_match = np.array([layername in x for x in weight_names])
    #         indeices = np.where(candi_match)[0]
    #         for index in indeices:
    #             # 逐个测试大小是否匹配
    #             weight_toload = np.asarray(g[weight_names[index]])
    #             if lw.shape == weight_toload.shape:
    #                 weight_names[index] = ""
    #                 return weight_toload
    #         raise ValueError
    #     except ValueError:
    #         ### 逐层寻找大小一致的权重
    #         layername = os.path.split(layername)[0]
    #         if layername == "":
    #             return np.random.random(lw.shape).astype(np.dtype(lw.dtype.name))
    #         else:
    #             return self.__search_match(layername, lw, g, weight_names)

    # @staticmethod
    # def match_by_Hungarian(layer:tf.keras.layers.Layer, g):
    #     '''
    #     定义cost matrix 用匈牙利算法匹配
    #     '''
    #     cost_matrix_default = 500
    #     # 查找并修改cost matrix，递归
    #     def search_match(ln, lw:tf.Variable):
    #         nonlocal iter_num, cost_matrix, li
    #         try:
    #             candi_match = np.array([ln in x for x in weight_names])
    #             indeices = np.where(candi_match)[0]
    #             match = []
    #             for index in indeices:
    #                 # 逐个测试大小是否匹配
    #                 weight_toload = np.asarray(g[weight_names[index]])
    #                 if lw.shape == weight_toload.shape:
    #                     # weight_names[index] = ""
    #                     match.append(index)
    #             if len(match) == 0:
    #                 raise ValueError
    #             else:
    #                 cost_matrix[li, match] = iter_num
    #                 return 
    #         except ValueError:
    #             ### 逐层寻找大小一致的权重
    #             ln = os.path.split(ln)[0]
    #             if ln == "":
    #                 return 
    #             else:
    #                 iter_num += 1
    #                 return search_match(ln, lw)    

    #     def calc_score(li, lw:tf.Variable):
    #         nonlocal cost_matrix
    #         ### 先查找有没有完全相同的
    #         candi = []
    #         for col, wn in enumerate(weight_names):
    #             weight_toload = np.asarray(g[weight_names[col]])
    #             if lw.shape == weight_toload.shape:
    #                 if wn == lw.name:
    #                     cost_matrix[li, col] = 0
    #                     return
    #                 else:
    #                     candi.append((col, wn))
    #         ### 没有再逐一计算编辑距离
    #         for col, wn in candi:
    #             weight_toload = np.asarray(g[weight_names[col]])
    #             # 计算字符串的编辑距离
    #             dist = Levenshtein.distance(wn, lw.name)
    #             cost_matrix[li, col] = dist

    #     ### 循环计算
    #     layer_names = [x.name for x in layer.weights]
    #     weight_names =  hdf5_format.load_attributes_from_hdf5_group(g, 'weight_names')
    #     cost_matrix = np.full((len(layer_names), len(weight_names)), cost_matrix_default) # 创建cost matrix
    #     start = time.time()
    #     for li, (ln, lw) in enumerate(zip(layer_names, layer.weights)):
    #         iter_num = 0
    #         # search_match(ln, lw)
    #         calc_score(li, lw)
    #     print("用时:{:>3.4f}s".format(time.time() - start))
    #     row_indices, col_indices = optimize.linear_sum_assignment(cost_matrix)
        
    #     # 赋值
    #     weight_values = [None for _ in range(len(layer_names))]
    #     for row, col in zip(row_indices, col_indices):
    #         score = cost_matrix[row, col]
    #         if score == cost_matrix_default:
    #             print("None", " -> ", layer_names[row])
    #         else:
    #             lw:tf.Variable = layer.weights[row] #层的权重
    #             print(weight_names[col], " -> ", layer_names[row])
    #             weight_values[row] = np.asarray(g[weight_names[col]]) # 保存的权重
    #     # 未能匹配的层不作修改
    #     for li, v in enumerate(weight_values):
    #         lw = layer.weights[li]
    #         if v is None:
    #             weight_values[li] = np.asarray(lw)
    #     return weight_values

    # @staticmethod
    # def match_by_name(layer:tf.keras.layers.Layer, g):
    #     pass

    # def __del__(self):
    #     if hasattr(self.f, 'close'):
    #         self.f.close()

def denormalize_bbox(bbox_n:torch.Tensor, img_size:Union[tuple, list, torch.Tensor]):
    """
    将归一化的边界框坐标还原为像素单位。

    parameter
    -----
        - bbox_n: 归一化的边界框坐标，形状为 (N, 4)，其中 N 是边界框的数量。每个边界框由四个值表示 (xmin, ymin, xmax, ymax)。值应在 [0, 1] 范围内。
        - img_size: 图像的大小，可以是元组、列表或形状为 (2,) 的张量，表示图像的宽度和高度。

    return
    -----
        - bbox: 还原为像素单位的边界框坐标，形状为 (N, 4)。
    """
    if isinstance(bbox_n, torch.Tensor):
        img_size = torch.Tensor(img_size)
        img_size = torch.cat([img_size, img_size]).to(bbox_n.device)
    elif isinstance(bbox_n, np.ndarray):
        img_size = np.array(img_size)
        img_size = np.concatenate((img_size, img_size))

    bbox = bbox_n * img_size

    return bbox

def normalize_bbox(bbox: torch.Tensor, img_size: Union[tuple, list, torch.Tensor]):
    """
    将边界框坐标转换为归一化的形式。

    parameter
    -----
        - bbox_n: 归一化的边界框坐标，形状为 (N, 4)，其中 N 是边界框的数量。每个边界框由四个值表示 (xmin, ymin, xmax, ymax)。值应在 [0, 1] 范围内。
        - img_size: 图像的大小，可以是元组、列表或形状为 (2,) 的张量，表示图像的宽度和高度。

    return
    -----
        - bbox: 还原为像素单位的边界框坐标，形状为 (N, 4)。
    """
    img_size = torch.Tensor(img_size)
    img_size = torch.cat([img_size, img_size])
    bbox_n = bbox / img_size

    return bbox_n

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    if tensor.is_cuda:
        tensor = tensor.detach().cpu()
    else:
        tensor = tensor.detach()
    return tensor.numpy()