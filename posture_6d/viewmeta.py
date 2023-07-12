# from compute_gt_poses import GtPostureComputer

# from toolfunc import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import cv2
import copy
from typing import Union, Any, Callable
from .posture import Posture
from .intr import CameraIntr
from .mesh_manager import MeshMeta
from .derive import calc_bbox_3d_proj, calc_landmarks_proj, calc_masks
from .utils import get_bbox_connections
import inspect

def copy_by_rect(crop_rect, org:np.ndarray):
    if len(crop_rect.shape) == 1:
        crop_rect = np.expand_dims(crop_rect, 0)
    assert len(crop_rect.shape) == 2 and crop_rect.shape[1] == 4, "the shape of crop_rect must be [N, 4]"
    if not np.issubdtype(crop_rect.dtype, np.integer):
        crop_rect = np.round(crop_rect).astype(np.int32)
    new = np.zeros(org.shape, org.dtype)
    for r in crop_rect:
        new[r[1]: r[3], r[0]: r[2]] = \
            org[r[1]: r[3], r[0]: r[2]]
    return new

def rotate_image(M, image:np.ndarray):
    # 执行旋转
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), 
                                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    return rotated

def rot_xy_list_2dpoints(M, points_2d:np.ndarray) -> list[float]:
    '''
    points_2d [N, (x,y)]
    '''
    points_2d = np.pad(points_2d, ((0,0),(0,1)), constant_values=1)
    new_points_2d:np.ndarray = M.dot(points_2d.T).T[:, :2]
    return new_points_2d

class ViewMeta():
    '''
    一个视角下的所有数据的元
    '''
    class AugmentPipeline():
        def get_ap_of_meta(self, class_)-> Union["ViewMeta.AugmentPipeline", None]:
            for value in self.meta.agmts.values():
                if isinstance(value, class_):
                    return value
            return None

        def __init__(self, meta:"ViewMeta") -> None:
            self.meta = meta
            self.new_obj = None

        @property
        def obj(self):
            return None

        def __inner_func(self, func, *args):
            if self.obj is None:
                return None
            else:
                self.new_obj = func(*args)
                return self.new_obj

        def _crop(self, crop_rect: ndarray):
            return self.obj
        
        def crop(self, crop_rect: ndarray)-> Union[cv2.Mat, None]:
            '''
            crop_rect: [N, [y1, x1, y2, x2]] int
            '''
            return self.__inner_func(self._crop, crop_rect)

        def _rotate(self, M: ndarray):
            return self.obj

        def rotate(self, M: ndarray):
            return self.__inner_func(self._rotate, M)

        def _change_brightness(self, delta_value, direction):
            return self.obj

        def change_brightness(self, delta_value:float, direction:tuple[float]=(0, 0)):
            return self.__inner_func(self._change_brightness, delta_value, direction)

        def _change_saturation(self, delta_value):
            return self.obj

        def change_saturation(self, delta_value:float):
            return self.__inner_func(self._change_saturation, delta_value)

    class RgbAP(AugmentPipeline):
        '''
        Rgb image augment pipline
        '''
        def __init__(self, meta:"ViewMeta") -> None:
            super().__init__(meta)

        @property
        def obj(self):
            return self.meta.rgb

        def _crop(self, crop_rect: np.ndarray):
            return copy_by_rect(crop_rect, self.obj)

        def _rotate(self, M: ndarray):
            return rotate_image(M, self.obj)
        
        def _change_brightness(self, delta_value, direction):
            direction = np.array(direction, np.float32)
            # 创建一个640x480的网格
            x, y = np.meshgrid(np.arange(self.obj.shape[0]), np.arange(self.obj.shape[1]))
            # 将网格转换为 (640*480, 2) 的坐标数组
            coords = np.stack((x.ravel(), y.ravel()), axis=1)
            value = np.sum(coords * direction, axis=1)
            if value.max() == 0:
                value[:] = 1 # 亮度整体变化
            else:
                value -= value.min()
                value = (value / value.max() - 0.5) * delta_value #沿着梯度方向变换

            hsv = cv2.cvtColor(self.obj, cv2.COLOR_BGR2HSV)
            v = hsv[:,:,2].astype(np.float32)
            v[tuple(coords.T)] += value
            v = np.clip(v, 0, 255)
            hsv[:,:,2] = v.astype(np.uint8)
            new_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            return new_rgb
        
        def _change_saturation(self, delta_value):
            hsv = cv2.cvtColor(self.obj, cv2.COLOR_BGR2HSV)
            s = hsv[:,:,1].astype(np.float32)
            s += delta_value
            s = np.clip(s, 0, 255)
            hsv[:,:,1] = s.astype(np.uint8)
            new_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            return new_rgb

    class DepthAP(AugmentPipeline):
        def __init__(self, meta: "ViewMeta") -> None:
            super().__init__(meta)

        @property
        def obj(self):
            return self.meta.depth

        def _crop(self, crop_rect: np.ndarray):
            new_depth = copy_by_rect(crop_rect, self.obj)
            return new_depth
        
        def _rotate(self, M: ndarray):
            new_depth = rotate_image(M, self.obj)
            return new_depth

    class MasksAP(AugmentPipeline):
        def __init__(self, meta: "ViewMeta") -> None:
            super().__init__(meta)

        @property
        def obj(self):
            return self.meta.masks

        def _crop(self, crop_rect: np.ndarray):
            new_masks:dict[str, cv2.Mat] = {}
            for _id, mask in self.obj.items():
                new_mask = copy_by_rect(crop_rect, mask)
                new_masks.update({_id: new_mask})
            return new_masks
        
        def _rotate(self, M:ndarray):
            new_masks = {}
            # 裁剪所有mask
            for _id, mask in self.obj.items():
                new_mask = rotate_image(M, mask)
                new_mask = cv2.threshold(new_mask, 127, 255, cv2.THRESH_BINARY)[-1]
                new_masks.update({_id: new_mask})
            return new_masks
        
    class ExtrVecAP(AugmentPipeline):
        def __init__(self, meta: "ViewMeta") -> None:
            super().__init__(meta)

        @property
        def obj(self):
            return self.meta.extr_vecs

        def _rotate(self, M:ndarray):
            new_trans_vector_info:dict[int, ndarray] = {}
            angle = np.arccos(M[0,0]) ####
            rot_Z_posture = Posture(rvec=[0, 0, -angle]) # 绕相机坐标系Z轴逆向旋转
            for _id, vecs in self.obj.items():
                org_posture = Posture(rvec=vecs[0], tvec=vecs[1])
                new_posture = rot_Z_posture * org_posture
                new_trans_vector_info.update({_id: np.stack([new_posture.rvec, new_posture.tvec], axis=0)})
            return new_trans_vector_info

    class IntrAP(AugmentPipeline):
        def __init__(self, meta: "ViewMeta") -> None:
            super().__init__(meta)

        @property
        def obj(self):
            return self.meta.intr

    class DepthScaleAP(AugmentPipeline):
        def __init__(self, meta: "ViewMeta") -> None:
            super().__init__(meta)

        @property
        def obj(self):
            return self.meta.depth_scale

    class Bbox3dAP(AugmentPipeline):
        def __init__(self, meta: "ViewMeta") -> None:
            super().__init__(meta)
        
        @property
        def obj(self):
            return self.meta.bbox_3d
        
        def _rotate(self, M:cv2.Mat):
            new_bbox_info = {}
            for _id, bbox in self.obj.items():
                new_bbox_info.update({_id: rot_xy_list_2dpoints(M, bbox)})
            return new_bbox_info

    class LandmarksAP(AugmentPipeline):
        def __init__(self, meta: "ViewMeta") -> None:
            super().__init__(meta)

        @property
        def obj(self):
            return self.meta.landmarks

        def _rotate(self, M:cv2.Mat):
            new_landmarks_info = {}
            for _id, landmark in self.obj.items():
                new_landmarks_info.update({_id: rot_xy_list_2dpoints(M, landmark)})
            return new_landmarks_info

    class VisibFractAP(AugmentPipeline):
        def __init__(self, meta: "ViewMeta") -> None:
            super().__init__(meta)
            self.old_masks_callback = lambda :self.get_ap_of_meta(self.meta.MasksAP).obj
            self.new_masks_callback = lambda :self.get_ap_of_meta(self.meta.MasksAP).new_obj

        @property
        def obj(self):
            return self.meta.visib_fract

        def calc_new_visib_fract(self, old_masks:dict[str, np.ndarray], new_masks:dict[str, np.ndarray]):
            # 修改visib_fract
            new_visib_fract = {}
            for _id, visib in self.obj.items():
                if visib == 0:
                    new_visib_fract.update({_id: 0})
                else:
                    mask = old_masks[_id]
                    proj_pixel_num = np.sum(mask.astype(np.bool_)) / visib # 不考虑任何遮挡和相机视野边界的像素数量
                    new_pixel_num = np.sum(new_masks[_id].astype(np.bool_))
                    vf = min(new_pixel_num / proj_pixel_num, 1)
                    new_visib_fract.update({_id: vf})      
            return new_visib_fract 

        def _crop(self, crop_rect: ndarray):
            old_masks = self.old_masks_callback()
            new_masks = self.new_masks_callback()
            return self.calc_new_visib_fract(old_masks, new_masks)
        
        def _rotate(self, crop_rect: ndarray):
            old_masks = self.old_masks_callback()
            new_masks = self.new_masks_callback()
            return self.calc_new_visib_fract(old_masks, new_masks)

    def __init__(self, 
                rgb: np.ndarray, 
                depth: np.ndarray, 
                masks: dict[int, np.ndarray], 
                extr_vecs:dict[int, ndarray],
                intr: ndarray,
                depth_scale:float,
                bbox_3d: dict[int, ndarray],        
                landmarks: dict[int, ndarray],
                visib_fract: dict[int, float],
                # keypoints_visib: dict[int, list],
                ) -> None:
        '''
        rgb: np.ndarray, 
        depth: np.ndarray, 
        masks: dict[int, np.ndarray], 
        extr_gt: list[dict[int, list]], 外参
        camera: dict, 相机参数
        visib_fract: dict[int, float], 物体可见性
        bbox: dict[int, ndarray:[B, (x,y)]],        
        keypoints: dict[int, ndarray:[N, (x,y)]],
        trans_vector:dict[int, list] 外参
        '''
        self.__init_parameters_keys = [x for x in locals().keys() if x in inspect.signature(self.__init__).parameters.keys()]
        self.__init_parameters_values = [id(x) for x in list(locals().values())]
        self.agmts:dict[str, ViewMeta.AugmentPipeline] = {}
        self.rgb:np.ndarray                 = self.__set_element(ViewMeta.RgbAP,         rgb)
        self.depth:np.ndarray               = self.__set_element(ViewMeta.DepthAP,       depth)            
        self.masks:dict[int, ndarray]       = self.__set_element(ViewMeta.MasksAP,       masks) #[N, H, W]
        self.extr_vecs:dict[int, ndarray]   = self.__set_element(ViewMeta.ExtrVecAP,     extr_vecs,  lambda x:self.__reshape_array_in_dict(x, (2, 3)))
        self.intr:ndarray                   = self.__set_element(ViewMeta.IntrAP,        intr)
        self.depth_scale: float             = self.__set_element(ViewMeta.DepthScaleAP,  depth_scale)
        self.bbox_3d:dict[int, ndarray]     = self.__set_element(ViewMeta.Bbox3dAP,      bbox_3d,    lambda x:self.__reshape_array_in_dict(x, (-1,2)))
        self.landmarks:dict[int, ndarray]   = self.__set_element(ViewMeta.LandmarksAP,   landmarks,  lambda x:self.__reshape_array_in_dict(x, (-1,2)))
        self.visib_fract: dict[int, float]  = self.__set_element(ViewMeta.VisibFractAP,  visib_fract)
        # self.keypoints_visib                = copy.deepcopy(keypoints_visib)
 
    def calc_by_base(self, mesh_dict:dict[int, MeshMeta], cover = False):
        assert self.rgb is not None
        assert self.extr_vecs is not None
        assert self.intr is not None
        assert self.depth_scale is not None

        camera_intr = CameraIntr(self.intr, self.rgb.shape[1], self.rgb.shape[0], self.depth_scale)
        postures = []
        mesh_list = []
        keys = []
        bbox_3d = {}
        landmarks = {}
        for key in self.extr_vecs:
            posture = Posture(rvec=self.extr_vecs[key][0], tvec=self.extr_vecs[key][1])
            meshmeta = mesh_dict[key]
            postures.append(posture)
            mesh_list.append(meshmeta)
            keys.append(key)
            bbox_3d[key] = calc_bbox_3d_proj(meshmeta, posture, camera_intr)
            landmarks[key] = calc_landmarks_proj(meshmeta, posture, camera_intr)
        
        if cover or self.masks is None or self.visib_fract is None:
            masks, visib_fracts = calc_masks(mesh_list, postures, camera_intr, ignore_depth=True)

            masks_dict = {}
            visib_fract_dict = {}
            for key, mask, visib_fract in zip(keys, masks, visib_fracts):
                masks_dict[key] = mask
                visib_fract_dict[key] = visib_fract

            if cover or self.masks is None:
                self.masks = masks_dict
            if cover or self.visib_fract is None:
                self.visib_fract = visib_fract_dict

        if cover or self.bbox_3d is None:
            self.bbox_3d = bbox_3d
        if cover or self.landmarks is None:
            self.landmarks = landmarks

    @staticmethod
    def from_base_data( rgb: np.ndarray, 
                        depth: np.ndarray, 
                        extr_vecs:dict[int, ndarray],
                        intr: np.ndarray,
                        depth_scale: float,
                        mesh_dict:dict[int, MeshMeta]):
        viewmeta = ViewMeta(rgb, depth, None, extr_vecs, intr, depth_scale, None, None, None)
        viewmeta.calc_by_base(mesh_dict)
        return viewmeta

    def __set_element(self, ap_type, value, func = lambda x:x):
        matched_idx = self.__init_parameters_values.index(id(value))
        key = self.__init_parameters_keys[matched_idx]
        self.__init_parameters_keys.pop(matched_idx)
        self.__init_parameters_values.pop(matched_idx)
        value = func(copy.deepcopy(value))
        ### 初始化增强
        ap = ap_type(self)
        self.agmts.update({key: ap})
        return value

    @staticmethod
    def __reshape_array_in_dict(dictionary: dict[Any, ndarray], shape):
        if isinstance(dictionary, dict):
            for key in dictionary.keys():
                dictionary[key] = dictionary[key].reshape(shape)
        return dictionary

    def __augment(self, funcname:str, *arg):
        aug_results = dict(zip(self.agmts.keys(), [agmt.__getattribute__(funcname)(*arg) for agmt in self.agmts.values()]))
        return ViewMeta(**aug_results) 

    ### augment
    def crop(self, crop_rect:np.ndarray):
        '''
        裁剪，去除部分变为全黑，不改变实际图像大小（考虑到裁剪+缩放会导致内参不一致，影响关键点位置预测）
        crop_rect: [N, [y1, x1, y2, x2]]
        '''
        return self.__augment("crop", crop_rect)

    def rotate(self, angle:float):
        '''
        brief
        -----
        旋转

        parameters
        -----
        angle: 旋转角度 弧度制
        
        return
        -----
        Description of the return
        '''
        cam_K = self.intr
        center:tuple[float] = (cam_K[0,2], cam_K[1,2])
        # 获取旋转矩阵
        M = cv2.getRotationMatrix2D(center, angle * 180/ np.pi, 1.0) # 角度制，旋转方向向外，与相机坐标系Z轴想法

        return self.__augment("rotate", M)

    def change_brightness(self, delta_value, direction = (0,0)):
        '''
        brief
        -----
        修改亮度
        
        parameters
        -----
        delta_value: float ,差值
        direction: Iterable,梯度方向，(0,0)则均匀变化 注意是(y,x)
        
        return
        -----
        FieldData
        '''
        direction = np.array(direction, np.float32)
        # 生成新的对象并返回
        return self.__augment("change_brightness", delta_value, direction)

    def change_saturation(self, delta_value):
        '''
        修改饱和度
        '''
        # 生成新的对象并返回
        return self.__augment("change_saturation", delta_value)
    
    def gaussian_noise(self):
        '''
        高斯噪声
        '''
        pass

    def show(self):
        masks = np.stack(list(self.masks.values()))
        mask = np.sum(masks.astype(np.float32) * 0.2, axis=0).astype(np.uint8)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        rgb = np.clip((self.rgb.astype(np.float32) + mask), 0, 255).astype(np.uint8)

        plt.subplot(1,2,1)
        plt.imshow(rgb)
        # landmarks
        for ldmks in self.landmarks.values():
            plt.scatter(ldmks[:,0], ldmks[:,1], c = 'g')
        # bbox_3d
        for bbox_3d in self.bbox_3d.values():
            plt.scatter(bbox_3d[:,0], bbox_3d[:,1], c = 'r')
            lines = get_bbox_connections(bbox_3d)
            for line in lines:
                plt.plot(line[0], line[1], c = 'r')
        # 标注类别与可见性
        for class_id, vb in self.visib_fract.items():
            label = "{} {:>.3}".format(class_id, vb)
            lt = np.min(self.bbox_3d[class_id], axis = 0)
            plt.text(lt[0], lt[1], label, verticalalignment='top')
        plt.subplot(1,2,2)
        plt.imshow(self.depth)
        