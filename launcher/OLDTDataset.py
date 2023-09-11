import os
import torch
from torch.utils.data import Dataset
from models.results import ObjPosture, ImagePosture
from models.utils import normalize_bbox
from posture_6d.data.dataset_format import VocFormat
from posture_6d.data.viewmeta import  ViewMeta
from posture_6d.core.posture import Posture

import numpy as np
import os
import cv2

from typing import Union

def transpose_data(batch_data) -> tuple[list]:
    """
    自定义 batch 内各个数据条目的组织方式
    :param data: 元组，第一个元素：句子序列数据，第二个元素：长度 第2维：句子标签
    :return: 填充后的句子列表、实际长度的列表、以及label列表
    """
    item_num = len(batch_data[0])
    outputs = [[bd[i] for bd in batch_data] for i in range(item_num)]
    return tuple(outputs)

def collate_fn(batch_data):
    return batch_data

class OLDTDataset(Dataset):
    def __init__(self, data_folder:str, set_:str, formattype = VocFormat):
        '''
        set must be "trian" or "val"
        '''
        self.data_folder = data_folder
        self.vocformat = formattype(data_folder)
        self.vocformat.depth_elements.close()
        self.vocformat.bbox_3ds_elements.close()
        self.vocformat.visib_fract_elements.close()
        self.vocformat.depth_scale_elements.close()
        # self.vocformat.masks_elements.close()
        # self.vocformat.intr_elements.close()

        self.set = set_
        self.image_folder       = os.path.join(data_folder, 'images', set_)
        self.landmarks_folder   = os.path.join(data_folder, 'landmarks', set_)
        self.labels_folder      = os.path.join(data_folder, 'labels', set_)
        self.trans_vecs_folder  = os.path.join(data_folder, 'trans_vecs', set_)
        
        # self.data_files = len(self.idx_array)
        self.data_files = self.vocformat.data_num

        self.set_augment_para(1,0)

    @property
    def idx_array(self):
        if self.set == self.vocformat.KW_TRAIN:
            return self.vocformat.train_idx_array
        elif self.set == self.vocformat.KW_VAL:
            return self.vocformat.val_idx_array
        else:
            raise ValueError("parameter set must be {} or {}".format(self.vocformat.KW_TRAIN, self.vocformat.KW_VAL))

    def set_augment_para(self, data_inflation: int, max_rotate_angle:float):
        self.data_inflation = int(max(data_inflation, 1))
        self.max_rotate_angle = max_rotate_angle

    def augment(self, viewmeta:ViewMeta)->ViewMeta:
        '''
        brief
        -----
        augment
        
        return
        -----
        ViewMeta
        '''
        angle = 2*np.random.rand()*self.max_rotate_angle - self.max_rotate_angle
        viewmeta = viewmeta.rotate(angle)
        bv = 200 * np.random.rand() - 100
        sv = 200 * np.random.rand() - 100
        viewmeta = viewmeta.change_brightness(bv)
        viewmeta = viewmeta.change_saturation(sv)
        return viewmeta

    def __len__(self):
        return len(self.idx_array) * self.data_inflation
    
    @staticmethod
    def _box_cxcywh2xyxy(bbox):
        x1 = bbox[:, 0] - bbox[:, 2] / 2
        y1 = bbox[:, 1] - bbox[:, 3] / 2
        x2 = bbox[:, 0] + bbox[:, 2] / 2
        y2 = bbox[:, 1] + bbox[:, 3] / 2

        # 将 x1, y1, x2, y2 组合成新的包围框表示
        new_bbox = np.stack([x1, y1, x2, y2], axis=1)
        return new_bbox

    def read_cache(self, idx):
        data_i = self.idx_array[idx]
        path = os.path.join(self.data_folder, self.set, str(data_i).rjust(6, '0')+'.pkl')

    def __getitem__(self, idx) -> ImagePosture:
        if_aug = idx % self.data_inflation > 0
        orig_idx = int(idx / self.data_inflation)
        data_i = self.idx_array[orig_idx]
        viewmeta = self.vocformat.read_one(data_i)   
        viewmeta.calc_bbox2d_from_mask(viewmeta.masks)
        if if_aug:
            viewmeta = self.augment(viewmeta)
        # viewmeta = viewmeta.rotate(0.2)

        image = viewmeta.color
        
        class_id = list(viewmeta.labels.keys()) 
        landmarks = [viewmeta.landmarks[x] for x in class_id]
        bbox_2d = viewmeta.bbox_2d
        bbox = np.array([bbox_2d[x] for x in class_id])
        bbox_n = normalize_bbox(bbox, image.shape[:2][::-1])
        trans_vecs = [viewmeta.extr_vecs[x] for x in class_id]
        postures = [Posture(rvec=x[0], tvec=x[1]) for x in trans_vecs]
        intr_M = viewmeta.intr

        image_posture = ImagePosture(image, intr_M=intr_M)
        for obj_i in range(len(class_id)):
            image_posture.obj_list.append(
                ObjPosture(landmarks[obj_i], 
                       bbox_n[obj_i],
                       class_id[obj_i],
                       image_posture.image_size,
                       postures[obj_i])
            )

        return image_posture

    # def __getitem__(self, idx) -> ImagePosture:
    #     file_name = self.data_files[idx]
    #     image_path = os.path.join(self.image_folder, file_name)
    #     landmarks_path = os.path.join(self.landmarks_folder, file_name.replace('.jpg', '.txt'))
    #     labels_path = os.path.join(self.labels_folder, file_name.replace('.jpg', '.txt'))
    #     trans_vecs_path = os.path.join(self.trans_vecs_folder, file_name.replace('.jpg', '.txt'))
    #     # 读取图像数据
    #     # image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    #     # image = torch.Tensor(image.transpose((2,0,1)))
    #     image = cv2.imread(image_path)
        
    #     # 读取关键点数据
    #     landmarks_data = np.loadtxt(landmarks_path)
    #     landmarks = landmarks_data.reshape(-1, 24, 2)
        
    #     # 读取标签数据
    #     labels_data = np.loadtxt(labels_path)
    #     labels_data = labels_data.reshape(-1, 5)
    #     class_id = labels_data[:, 0].astype(np.int64)  # class_id
    #     bbox_n = labels_data[:, 1:]  # bbox (cx, cy, w, h)
    #     bbox_n = self._box_cxcywh2xyxy(bbox_n) # bbox (x1, y1, x2, y2)

    #     # 读取变换向量
    #     trans_vecs_data = np.loadtxt(trans_vecs_path)
    #     trans_vecs = trans_vecs_data.reshape(-1, 2, 3)
        
    #     image_posture = ImagePosture(image)
    #     for obj_i in range(landmarks.shape[0]):
    #         image_posture.obj_list.append(
    #             ObjPosture(landmarks[obj_i], 
    #                    bbox_n[obj_i],
    #                    class_id[obj_i],
    #                    image_posture.image_size,
    #                    trans_vecs[obj_i])
    #         )

    #     return image_posture
    
