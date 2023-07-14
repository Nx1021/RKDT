from torch import Tensor
import torch
from .utils import denormalize_bbox, normalize_bbox, tensor_to_numpy

from posture_6d.posture import Posture

import matplotlib.pyplot as plt
import numpy as np
from typing import Union

class LandmarkDetectionResult():
    def __init__(self, bbox_n:Tensor, class_id:int, landmarks_n:Tensor, landmarks_probs:Tensor, img_size:Tensor) -> None:
        '''
        parameter
        -----
        bbox_n          : normalized (x1, y1, x2, y2)
        class_id        : int
        landmarks_n     : normalized (ldmk_num, (x, y))
        img_size        : tuple|Tensor  (w, h)
        '''
        self.bbox_n         = bbox_n      # normalized (x1, y1, x2, y2)
        self.class_id       = class_id
        self.landmarks_n    = landmarks_n # normalized (ldmk_num, (x, y))
        self.landmarks_probs = landmarks_probs
        if not isinstance(img_size, Tensor):
            img_size = Tensor(img_size).to(bbox_n.device).to(torch.int32)

        self.img_size       = img_size    # pixel (w, h)

    @property
    def bbox(self):
        # with torch.no_grad():
        bbox = denormalize_bbox(self.bbox_n.unsqueeze(0), self.img_size)
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
        self.bbox_n = bbox_n
        self.class_id: int = class_id
        if trans_vecs is not None:
            rvec, tvec = trans_vecs
            self.posture = Posture(rvec=rvec, tvec=tvec)
        else:
            self.posture = None

        self.image_size = image_size

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
    def __init__(self, image: np.ndarray, *objs: ObjPosture) -> None:
        """
        初始化ImagePosture对象。

        参数:
            image: np.ndarray，表示图像的数组。
            *objs: 可变参数，表示ObjPosture对象的列表，用于初始化ImagePosture对象中的obj_list属性。
        """
        self.obj_list: list[ObjPosture] = list(objs)
        self.image = image

    @property
    def image_size(self):
        """
        获取图像的尺寸。

        返回:
            图像的尺寸，形式为 (width, height)。
        """
        return self.image.shape[:2][::-1]

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

        示例用法：
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