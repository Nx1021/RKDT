import datetime
import os
import pickle
from typing import Generator, Callable, Iterable

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.OLDT import OLDT
from models.results import LandmarkDetectionResult, ImagePosture, compare_image_posture
from post_processer.PostProcesser import PostProcesser
from post_processer.error_calculate import ErrorCalculator, ErrorResult, match_roi
from post_processer.pnpsolver import PnPSolver
from .BasePredictor import BasePredictor
from .OLDTDataset import OLDTDataset, collate_fn
from .BaseLauncher import Launcher, BaseLogger

from utils.yaml import yaml_load



def is_image(arr):
    try:
        Image.fromarray(arr)
        return True
    except:
        return False

class IntermediateManager:
    '''
    中间输出管理器
    用于将中间输出在硬盘上读写
    每次保存一个对象，会在指定的目录下保存一个.pkl文件，并计数。
    '''
    def __init__(self, root_dir, sub_dir_name=""):
        self.root_dir = root_dir
        if sub_dir_name == "":
            current_time = datetime.datetime.now()
            self.sub_dir_name = current_time.strftime("%Y%m%d%H%M%S")
        else:
            self.sub_dir_name = sub_dir_name
        self.save_dir = os.path.join(self.root_dir, self.sub_dir_name)
        os.makedirs(self.save_dir, exist_ok=True)

        self.init_save_count() # 不同类型保存计数

    def init_save_count(self):
        '''
        不同类型的保存数量的计数
        '''
        self.save_count = {}

        classnames = os.listdir(self.save_dir)
        for classname in classnames:
            class_path = os.path.join(self.save_dir, classname)
            if os.path.isdir(class_path):
                file_num = len(os.listdir(class_path))
                self.save_count[classname] = file_num

    def load_objects_generator(self, class_name, method:Callable):
        '''
        逐个加载指定类别的所有对象的生成器方法
        返回一个生成器对象，逐个返回对象
        '''
        def load_objects_generator():
            class_dir = os.path.join(self.save_dir, class_name)

            # 检查指定类别的目录是否存在
            if not os.path.isdir(class_dir):
                return

            length = len(os.listdir(class_dir))
            for i in range(length):
                yield method(class_name, i)

        return load_objects_generator()

    def _save_object(self, class_name, obj, save_func:Callable):
        '''
        保存对象到指定的目录中，并根据save_func指定的方法进行保存
        save_func: 以特定方式存储的函数
        '''
        class_dir = os.path.join(self.save_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # 生成保存文件名，格式为 "class_name_count"
        count = self.save_count.get(class_name, 0) + 1
        file_name = f"{str(count).rjust(6, '0')}"

        # 使用save_func保存对象
        file_path = os.path.join(class_dir, file_name)
        save_func(file_path, obj)

        # 更新保存计数
        self.save_count[class_name] = count

    def _load_object(self, class_name, index:int, load_func:Callable):
        '''
        加载指定类别的所有对象或特定索引的对象
        如果提供了索引参数，则返回对应索引的对象
        否则，返回所有对象的列表
        load_func: 以特定方式加载对象的函数
        '''
        class_dir = os.path.join(self.save_dir, class_name)

        # 检查指定类别的目录是否存在
        if not os.path.isdir(class_dir):
            return []

        objects = []
        file_names = os.listdir(class_dir)
        if index >= 0 and index < len(file_names):
            file_path = os.path.join(class_dir, file_names[index])
            obj = load_func(file_path)
            objects.append(obj)
        else:
            raise IndexError

        return objects

    def save_image(self, class_name, image):
        '''
        保存图像到指定的目录中，并计数
        '''
        def save_image_func(file_path, image):
            file_path += ".png"
            cv2.imwrite(file_path, image)

        self._save_object(class_name, image, save_image_func)

    def load_image(self, class_name, index:int):
        '''
        加载指定类别的图像对象
        '''
        def load_image_func(file_path):
            return cv2.imread(file_path)

        return self._load_object(class_name, index, load_image_func)

    def save_pkl(self, class_name, obj):
        '''
        保存对象为.pkl文件到指定的目录中，并计数
        '''
        def save_pkl_func(file_path, obj):
            file_path += ".pkl"
            with open(file_path, "wb") as file:
                pickle.dump(obj, file)

        self._save_object(class_name, obj, save_pkl_func)

    def load_pkl(self, class_name, index:int):
        '''
        加载指定类别的.pkl文件为对象
        '''
        def load_pkl_func(file_path):
            with open(file_path, "rb") as file:
                return pickle.load(file)

        return self._load_object(class_name, index, load_pkl_func)

class OLDTPredictor(BasePredictor, Launcher):
    def __init__(self, model, cfg, log_remark, batch_size=32,  if_postprocess = True, if_calc_error = False, intermediate_from:str = ""):
        Launcher.__init__(self, model, batch_size, log_remark)
        BasePredictor.__init__(self, model, batch_size)

        # Enable GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model:OLDT = self.model.to(device)
        self.model.set_mode("predict")

        self.pnpsolver = PnPSolver(cfg)
        out_bbox_threshold = yaml_load(cfg)["out_bbox_threshold"]
        self.postprocesser = PostProcesser(self.pnpsolver, out_bbox_threshold)
        self.error_calculator = ErrorCalculator(self.pnpsolver, model.nc)

        self.if_postprocess = if_postprocess or if_calc_error # 如果要计算损失，必须先执行后处理
        self.if_calc_error = if_calc_error

        if intermediate_from and os.path.exists(os.path.join(self.log_root, intermediate_from, "intermediate_output")):
            intermediate_root = os.path.join(self.log_root, intermediate_from)
        else:
            intermediate_root = os.path.join(self.log_dir)
        self.intermediate_manager: IntermediateManager = IntermediateManager(intermediate_root, "intermediate_output")
        self.save_imtermediate = True

        self.gt_dir = "gt"
        self.predictions_dir = "list_" + LandmarkDetectionResult.__name__
        self.processed_dir = ImagePosture.__name__
        
        self.frametimer.set_batch_size(self.batch_size)

        self.postprocess_mode = "v" # or 'v'

        self.logger = BaseLogger(self.log_dir)
        self.logger.log(cfg)

    def _preprocess(self, inputs) -> list[cv2.Mat]:
        '''
        对输入进行处理，可以接收的输入：
        1、经过Dataloader加载的CustomDataset对象的输出:list[ImagePosture]
        2、单张图像：np.ndarray: 三通道uint8
        3、图像列表：Iterable[np.ndarray]: 三通道uint8

        return
        ----
        图像列表：list[np.ndarray]
        '''
        if isinstance(inputs, np.ndarray):
            if inputs.ndim == 3 and inputs.shape[2] == 3 and inputs.dtype == np.uint8:
                # 单张图像
                return [inputs]
            else:
                raise ValueError("输入必须是三通道的uint8类型的ndarray")
            
        elif isinstance(inputs, Iterable):
            image_list = []
            for obj in inputs:
                if isinstance(obj, np.ndarray) and obj.ndim == 3 and obj.shape[2] == 3 and obj.dtype == np.uint8:
                    # 图像列表
                    image_list.append(obj)
                elif isinstance(obj, ImagePosture):
                    image_list.append(obj.image)
                else:
                    raise ValueError("输入中的所有元素必须是三通道的uint8类型的ndarray")
            return image_list
        else:
            raise ValueError("输入类型不符合要求")

    def _postprocess(self, inputs) ->list[ImagePosture]:
        image_list:list[np.ndarray] = inputs[0]
        predictions:list[list[LandmarkDetectionResult]] = inputs[1]
        # self.postprocesser.process(image_list, predictions, 'v')
        return self.postprocesser.process(image_list, predictions, self.postprocess_mode )

    def _calc_error(self, gt:list[ImagePosture], pred:list[ImagePosture]):
        ### 匹配真值和预测的roi
        error_result:list[list[tuple[ErrorResult]]] = []
        for gt_imgpstr, pred_imgpstr in zip(gt, pred):
            matched = match_roi(gt_imgpstr, pred_imgpstr)
            image_error_result = []
            for m in matched:
                one_error_result = self.error_calculator.calc_one_error(m[0], m[1], m[2], ErrorCalculator.ALL)
                image_error_result.append(one_error_result)
            error_result.append(image_error_result)
        return error_result

    def preprocess(self, image) -> list[cv2.Mat]:
        return super().preprocess(image)

    def postprocess(self, inputs) ->list[ImagePosture]:
        return super().postprocess(inputs)

    def calc_error(self, gt: list, postprocessed: list)->list[list[tuple[ErrorResult]]]:
        return super().calc_error(gt, postprocessed)
    
    def predict_from_dataset(self, dataset):
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        
        with torch.no_grad():
            num_batches = len(data_loader)
            for batch in tqdm(data_loader, total=num_batches, leave=True):
                # Preprocessing
                inputs = self.preprocess(batch)
                if self.save_imtermediate:
                    for obj in batch:
                        self.intermediate_manager.save_pkl(self.gt_dir, obj)

                # Model inference
                predictions:list[list[LandmarkDetectionResult]] = self.inference(inputs)
                if self.save_imtermediate:
                    for obj in predictions:
                        self.intermediate_manager.save_pkl(self.predictions_dir, obj)

                # Postprocessing
                if self.if_postprocess:
                    processed:list[ImagePosture] = self.postprocess((inputs, predictions))
                    if self.save_imtermediate:
                        for obj in processed:
                            self.intermediate_manager.save_pkl(self.processed_dir, obj)

                # error
                if self.if_calc_error:
                    self.calc_error(batch, processed)
        with self.logger.capture_output("process record"):
            self.frametimer.print()
            self.error_calculator.print_result()

    def plot_outlier(self, error_result:list[list[tuple[ErrorResult]]], 
                     gt_list:list[ImagePosture],
                     pred_list:list[ImagePosture], 
                     metrics = ErrorCalculator.REPROJ):
        def save_figure(file_path, image):
            plt.gcf()
            file_path += ".svg"
            plt.savefig(file_path)
            plt.clf()

        for er, gt, pred in zip(error_result, gt_list, pred_list):
            try:
                target_metrics = [list(filter(lambda y: y.type == metrics, x))[0] for x in er] #one type error for one image
            except IndexError:
                continue
            passed = all([x.passed for x in target_metrics])
            if not passed and len(target_metrics) > 0:
                # compare_image_posture(gt, pred)
                # text = ""
                # for x in target_metrics:
                #     text += str(x.error) + "\n"
                # plt.text(0, 1, text, ha='left', va='top', transform=plt.gca().transAxes)
                # self.intermediate_manager._save_object("error_outlier", None, save_figure)
                self.intermediate_manager.save_pkl("error_outlier_raw", (gt, pred))

    def postprocess_from_intermediate(self, plot_outlier = False):
        predictions_generator:Generator[list[LandmarkDetectionResult]] = \
            self.intermediate_manager.load_objects_generator(self.predictions_dir, self.intermediate_manager.load_pkl)
        gt_generator:Generator[ImagePosture] =\
            self.intermediate_manager.load_objects_generator(self.gt_dir, self.intermediate_manager.load_pkl)
        total_num = self.intermediate_manager.save_count[self.gt_dir]
        for prediction, gt in tqdm(zip(predictions_generator, gt_generator), total=total_num, leave=True):
            image = gt[0].image
            processed:list[ImagePosture] = self.postprocess(([image], prediction))
            error_result:list[list[tuple[ErrorResult]]] = self.calc_error(gt, processed)
            if plot_outlier:
                self.plot_outlier(error_result, gt, processed)
        with self.logger.capture_output("process record"):
            self.frametimer.print()
            self.error_calculator.print_result()
        print("done")

    def calc_error_from_imtermediate(self):
        pass

    def predict_single_image(self, image):
        with torch.no_grad():
            inputs:list[cv2.Mat] = self.preprocess(image)
            # inputs = torch.from_numpy(np.expand_dims(preprocessed_image, axis=0)).to(device)
            predictions = self.inference(inputs)
            processed_predictions = self.postprocess((inputs, predictions))

        return processed_predictions

    def clear(self):
        self.frametimer.reset()
        self.error_calculator.clear()