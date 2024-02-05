import datetime
import os
import pickle
from typing import Generator, Callable, Iterable, Union

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.ops import generalized_box_iou
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.OLDT import OLDT
from models.loss import LandmarkLoss
from models.results import LandmarkDetectionResult, ImagePosture, compare_image_posture
from post_processer.PostProcesser import PostProcesser, \
    create_pnpsolver, create_model_manager

from posture_6d.derive import PnPSolver
from posture_6d.metric import MetricCalculator, MetricResult
from posture_6d.core.posture import Posture
from posture_6d.data.mesh_manager import get_bbox_connections
from .OLDTDataset import OLDTDataset, collate_fn
from .BaseLauncher import Launcher, BaseLogger

from utils.yaml import load_yaml
from models.utils import tensor_to_numpy


def match_roi(pred:ImagePosture, gt:ImagePosture) -> list[tuple[int, Posture, Posture]]:
    '''
    matched: list[tuple[class_id, pred_Posture, gt_Posture]]
    '''
    pred_image, pred_landmarks, pred_class_ids, pred_bboxes, pred_postures  = pred.split(get_trans_vecs=False)
    gt_image, gt_landmarks, gt_class_ids, gt_bboxes, gt_postures            = gt.split(get_trans_vecs=False)    
    ### 匹配，类别必须一致，bbox用giou评估
    M = len(gt_class_ids)
    N = len(pred_class_ids)
    if N == 0 or M == 0:
        return []
    # bbox
    gt_bboxes_tensor = torch.Tensor(np.array(gt_bboxes))
    pred_bboxes_tensor = torch.stack(pred_bboxes).to(gt_bboxes_tensor.device)
    cost_matrix_bbox = generalized_box_iou(pred_bboxes_tensor, gt_bboxes_tensor).numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix_bbox, maximize=True)

    cost_matrix_id = np.zeros((N, M), dtype=np.float32)  # 创建一个整数类型的全零矩阵
    for i in range(N):
        for j in range(M):
            if pred_class_ids[i] == gt_class_ids[j]:
                cost_matrix_id[i, j] = 2  # 不同元素的cost为1
    ###
    matched:list[tuple] = []
    for ri, ci in zip(row_ind, col_ind):
        if pred_class_ids[ri] == gt_class_ids[ci] and \
            pred_postures[ri] is not None and\
            gt_postures[ci] is not None:
            matched.append((gt_class_ids[ci], pred_postures[ri], gt_postures[ci]))
    return matched

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

class OLDTPredictor(Launcher):
    def __init__(self, model, cfg_file,
                  train_dataset:OLDTDataset = None, 
                  val_dataset:OLDTDataset = None, 
                  batch_size=32,  
                  if_postprocess = True, 
                  if_calc_error = False, 
                  intermediate_from:str = "", 
                  log_remark = ""):
        super().__init__(model, batch_size, log_remark)

        # Enable GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model:OLDT = self.model.to(device)
        self.model.set_mode("predict")

        self.train_dataset  = train_dataset
        self.val_dataset    = val_dataset

        pnpsolver = create_pnpsolver(cfg_file)
        mesh_manager = create_model_manager(cfg_file)
        out_bbox_threshold = load_yaml(cfg_file)["out_bbox_threshold"]
        self.postprocesser = PostProcesser(pnpsolver, mesh_manager, out_bbox_threshold)
        self.error_calculator = MetricCalculator(pnpsolver, mesh_manager)

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

        self.postprocess_mode = "v" # or 'v'

        self.logger = BaseLogger(self.log_dir)
        self.logger.log(cfg_file)

        self.loss = LandmarkLoss(cfg_file)

        self._use_depth = False

    @Launcher.timing(-1)
    def preprocess(self, inputs:Union[list[ImagePosture], np.ndarray, Iterable[np.ndarray]]) -> list[cv2.Mat]:
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

    @Launcher.timing(1)
    def inference(self, inputs:Iterable[np.ndarray]):
        results = self.model(inputs)
        return results

    @Launcher.timing(1)
    def postprocess(self, 
                    image_list:list[np.ndarray], 
                    predictions:list[list[LandmarkDetectionResult]],
                    depths = None) ->list[ImagePosture]:
        return self.postprocesser.process(image_list, predictions, depths = depths, mode = self.postprocess_mode)

    @Launcher.timing(-1)
    def calc_error(self, pred: Iterable[ImagePosture], gt: Iterable[ImagePosture])->list[list[tuple[MetricResult]]]:
        ### 匹配真值和预测的roi
        error_result:list[list[tuple[MetricResult]]] = []
        for pred_imgpstr, gt_imgpstr in zip(pred, gt):
            matched = match_roi(pred_imgpstr, gt_imgpstr)
            image_error_result = []
            for m in matched:

                one_error_result = self.error_calculator.calc_one_error(m[0], m[1], m[2], MetricCalculator.ALL)
                image_error_result.append(one_error_result)
            error_result.append(image_error_result)
        return error_result
    
    def predict_from_dataset(self, dataset:OLDTDataset, plot_outlier = False):
        dataset.set_use_depth(self._use_depth)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=self.num_workers)
        
        with torch.no_grad():
            num_batches = len(data_loader)
            iters:Iterable[list[ImagePosture]] = tqdm(data_loader, total=num_batches, leave=True)
            for idx, batch in enumerate(iters):
                # if idx % 100 != 0:
                #     continue

                # Preprocessing
                inputs = self.preprocess(batch)
                if self.save_imtermediate:
                    for obj in batch:
                        self.intermediate_manager.save_pkl(self.gt_dir, obj)

                # Model inference
                self.model.if_gather = True
                predictions:list[list[LandmarkDetectionResult]] = self.inference(inputs)
                if self.save_imtermediate:
                    for obj in predictions:
                        self.intermediate_manager.save_pkl(self.predictions_dir, obj)

                ###
                # for i in range(len(batch)):
                #     ip:ImagePosture = batch[i]
                #     predictions[i][0].bbox_n = torch.Tensor(ip.obj_list[0].bbox_n).to("cuda")
                ###

                # Postprocessing
                if self.if_postprocess:
                    # print([x.obj_list[0].tvec for x in batch if isinstance(x, ImagePosture)])
                    depths = [x.depth for x in batch if isinstance(x, ImagePosture)] if self._use_depth else None
                    processed:list[ImagePosture] = self.postprocess(inputs, predictions, depths = depths)
                    if self.save_imtermediate:
                        for obj in processed:
                            self.intermediate_manager.save_pkl(self.processed_dir, obj)

                # error
                if self.if_calc_error:
                    error_result = self.calc_error(processed, batch)
                    if plot_outlier:
                        self.plot_outlier(error_result, batch, processed)

                # self.plot_compare(batch, processed)

        with self.logger.capture_output("process record"):
            self.frame_timer.print()
            self.error_calculator.print_result()

    def predict_val(self, plot_outlier = False):
        self.clear()
        self.predict_from_dataset(self.val_dataset, plot_outlier)

    def predict_train(self, plot_outlier = False):
        self.clear()
        self.predict_from_dataset(self.train_dataset, plot_outlier)

    def plot_outlier(self, error_result:list[list[tuple[MetricResult]]], 
                     gt_list:list[ImagePosture],
                     pred_list:list[ImagePosture], 
                     metrics = MetricCalculator.REPROJ):
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
            passed = any([x.passed for x in target_metrics])
            if not passed and len(target_metrics) > 0:
                compare_image_posture(gt, pred)
                text = ""
                for x in target_metrics:
                    text += str(x.error) + "\n"
                plt.text(0, 1, text, ha='left', va='top', transform=plt.gca().transAxes)
                self.intermediate_manager._save_object("error_outlier", None, save_figure)
                # self.intermediate_manager.save_pkl("error_outlier_raw", (gt, pred))

    def plot_compare(self,  gt_list:list[ImagePosture],
                    pred_list:list[ImagePosture]):
        def save_figure(file_path, image):
            plt.gcf()
            file_path += ".svg"
            plt.savefig(file_path)
            plt.clf()

        def plot_bbox_3d(bbox_3d, color):
            plt.scatter(bbox_3d[:,0], bbox_3d[:,1], c = color, s=5)
            lines = get_bbox_connections(bbox_3d)
            for line in lines:
                plt.plot(line[0], line[1], c = color, linewidth=1)

        for gt, pred in zip(gt_list, pred_list):
            compare_image_posture(gt, pred)
            for gt_item, pred_item in zip(gt.obj_list, pred.obj_list):
                bbox_3d = self.postprocesser.mesh_manager.get_bbox_3d(gt_item.class_id)
                gt_bbox_3d_proj   = self.postprocesser.pnpsolver.calc_reproj(bbox_3d, gt_item.rvec,   gt_item.tvec)
                pred_bbox_3d_proj = self.postprocesser.pnpsolver.calc_reproj(bbox_3d, pred_item.rvec, pred_item.tvec)
                gt_color = "lawngreen"
                pred_color = "lightslategray"
                plot_bbox_3d(gt_bbox_3d_proj, gt_color)
                plot_bbox_3d(pred_bbox_3d_proj, pred_color)
            self.intermediate_manager._save_object("plot_compare", None, save_figure)
            # self.intermediate_manager.save_pkl("error_outlier_raw", (gt, pred))

    def postprocess_from_intermediate(self, plot_outlier = False):
        predictions_generator:Generator[list[LandmarkDetectionResult]] = \
            self.intermediate_manager.load_objects_generator(self.predictions_dir, self.intermediate_manager.load_pkl)
        gt_generator:Generator[ImagePosture] =\
            self.intermediate_manager.load_objects_generator(self.gt_dir, self.intermediate_manager.load_pkl)
        total_num = self.intermediate_manager.save_count[self.gt_dir]
        for prediction, gt in tqdm(zip(predictions_generator, gt_generator), total=total_num, leave=True):
            image = gt[0].image
            processed:list[ImagePosture] = self.postprocess(([image], prediction))
            error_result:list[list[tuple[MetricResult]]] = self.calc_error(processed, gt)
            if plot_outlier:
                self.plot_outlier(error_result, gt, processed)
        with self.logger.capture_output("process record"):
            self.frame_timer.print()
            self.error_calculator.print_result()
        print("done")

    def calc_error_from_imtermediate(self):
        pass

    def predict_single_image(self, image, depth = None):
        with torch.no_grad():
            inputs:list[cv2.Mat] = self.preprocess(image)
            # inputs = torch.from_numpy(np.expand_dims(preprocessed_image, axis=0)).to(device)
            predictions = self.inference(inputs)
            processed_prediction:ImagePosture = self.postprocess(inputs, predictions, depths = [depth])[0]

        return processed_prediction

    def clear(self):
        self.frame_timer.reset()
        self.error_calculator.clear()
