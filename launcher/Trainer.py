from launcher.OLDT_Dataset import transpose_data

import os
import shutil
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

import numpy as np
from models.OLDT import OLDT
from models.loss import LandmarkLossRecorder
from models.results import ImagePosture
from launcher.OLDT_Dataset import collate_fn
from launcher.utils import BaseLogger, Launcher
from utils.yaml import yaml_load
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime
import platform
import sys

import time

from typing import Union
sys_name = platform.system()
if sys_name == "Windows":
    TESTFLOW = False
else:
    TESTFLOW = False

class TrainFlow():
    def __init__(self, trainer:"Trainer", flowfile) -> None:
        self.trainer = trainer
        self.epoch = 0
        self.flow:dict = yaml_load(flowfile, False)
        self.stage_segment = list(self.flow.keys())
        if self.stage_segment[0] != 0:
            raise ValueError("the first stage must start at epoch 0!")
        self.scheduler = None

    @property
    def cur_stage(self):
        return sum([self.epoch >= x for x in self.stage_segment]) - 1

    def get_lr_func(self, lr_name, totol_step, initial_lr):
        for param_group in self.trainer.optimizer.param_groups:
            param_group['initial_lr'] = initial_lr
        if lr_name == "warmup":
            return LambdaLR(self.trainer.optimizer, 
                            lr_lambda=lambda step: min(step / totol_step, 1.0))
        if lr_name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(self.trainer.optimizer, 
                                                        totol_step)
    
    def enter_new_stage(self):
        stage_info = self.flow[self.stage_segment[self.cur_stage]]
        if stage_info is None:
            return
        totol_step = self.stage_segment[self.cur_stage + 1] - self.stage_segment[self.cur_stage]
        self.scheduler = self.get_lr_func(stage_info["lr_func"], totol_step, stage_info["lr"])
        if "cfg" in stage_info:
            self.trainer.inner_model.cfg.update(stage_info["cfg"])

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.epoch >= self.stage_segment[-1]:
            raise StopIteration  
        if self.scheduler is not None:
            self.scheduler.step()        
        if self.epoch in self.stage_segment:
            self.enter_new_stage()        
        self.epoch += 1 
        return self.epoch

class TrainLogger(BaseLogger):
    def __init__(self, log_dir):
        super().__init__(log_dir)
        self.writer = SummaryWriter(log_dir)

    def write_epoch(self, tag, value, step):
        if isinstance(value, torch.Tensor):
            value = value.item()
        # 写入 SummaryWriter
        self.writer.add_scalar(tag, value, step)

        # 写入文件
        log_file = os.path.join(self.log_dir, f"{tag}.txt")
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = "{:>4}  \t{}  \t{:<10.6f}\n".format(step, current_time, value)

        with open(log_file, 'a') as f:
            f.write(log_line)

class Trainer(Launcher):
    def __init__(self,
                 model:Union[OLDT, torch.nn.DataParallel],
                 train_dataset,
                 val_dataset,
                 criterion,
                 batch_size,
                 flowfile = "",
                 distribute = False,
                 test=False,
                 start_epoch = 0):
        super().__init__(model, batch_size)
        self.model:Union[OLDT, torch.nn.DataParallel] = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.flow = TrainFlow(self, flowfile)
        self.distribute = distribute
        self.test = test

        if self.distribute:
            # 初始化分布式环境
            dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456', world_size=torch.cuda.device_count(), rank=torch.cuda.current_device())
            self.model = torch.nn.parallel.DistributedDataParallel(self.model)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters())  # 初始化优化器
        self.criterion = criterion

        self.best_val_loss = float('inf')  # 初始化最佳验证损失为正无穷大

        self.test = test
        # 创建TensorBoard的SummaryWriter对象，指定保存日志文件的目录
        if not test:
            self.logger = TrainLogger(self.log_dir)

        self.cur_epoch = 0
        self.start_epoch = start_epoch

        self.freeze_modules = [self.inner_model.landmark_branches[0].pos_embed]

    @property
    def skip(self):
        return self.cur_epoch < self.start_epoch

    @property
    def inner_model(self):
        if isinstance(self.model, torch.nn.DataParallel):
            module:OLDT= self.model.module # type: ignore
            return module
        else:
            return self.model

    def pre_process(self, batch:list[ImagePosture]):
        images, keypoints, labels, bboxes_n, trans_vecs = transpose_data([x.split() for x in batch])
        keypoints = self._to_device(keypoints)
        labels = self._to_device(labels)
        bboxes_n = self._to_device(bboxes_n)
        return images, keypoints, labels, bboxes_n


    def _to_device(self, tensor_list:list[torch.Tensor]):
        return [torch.from_numpy(np.array(x)).to(self.device).to(torch.get_default_dtype()) for x in tensor_list]

    def check_has_target(self, gt_labels:list[torch.Tensor], active_class:list[int]):
        for lables in gt_labels:
            for l in lables:
                if l in active_class:
                    return True
        return False

    def forward_one_epoch(self, dataloader:DataLoader, backward = False):
        for module in self.freeze_modules:
            module.train(False)
        desc = "Train" if backward else "Val"
        ldmk_loss_mngr = LandmarkLossRecorder(desc)
        if self.skip:
            dataloader = range(len(dataloader))
        progress = tqdm(dataloader, desc=desc, leave=True)
        for image_posture in progress:
            if not self.skip:
                images, gt_landmarks, gt_labels, gt_bboxes_n = self.pre_process(image_posture)
                if self.check_has_target(gt_labels, self.inner_model.landmark_branch_classes):
                    # 前向传播
                    detection_results = self.model(images)
                    loss:torch.Tensor = self.criterion(gt_labels, gt_landmarks, gt_bboxes_n, detection_results, ldmk_loss_mngr)
                    if torch.isnan(loss).item():
                        print("loss nan, break!")
                        self.inner_model.save_branch_weights("./weights/", self.start_timestamp)
                        sys.exit()
                    # 反向传播和优化
                    if backward and ldmk_loss_mngr.buffer.detect_num > 0 and isinstance(loss, torch.Tensor):
                        self.optimizer.zero_grad()
                        loss.backward()
            
            if backward:
                self.optimizer.step()

            # 更新进度条信息
            progress.set_postfix({'Loss': "{:>8.4f}".format(ldmk_loss_mngr.loss())})
            if TESTFLOW:
                break
            ldmk_loss_mngr.buffer.clear()
            
        # 将val_loss写入TensorBoard日志文件
        if backward:
            self.logger.write_epoch("Learning rate", self.optimizer.param_groups[0]["lr"], self.cur_epoch)
        for key, value in ldmk_loss_mngr.to_dict().items():
            self.logger.write_epoch(key, value, self.cur_epoch)

        return ldmk_loss_mngr.loss()

    def train_one_epoch(self, dataloader):
        self.inner_model.set_mode("train")
        return self.forward_one_epoch(dataloader, True)

    def val_one_epoch(self, dataloader):
        self.inner_model.set_mode("val")
        with torch.no_grad():
            return self.forward_one_epoch(dataloader, False)

    def train(self):
        print("start to train... time:{}".format(self.start_timestamp))
        self.cur_epoch = 0
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)

        for epoch in self.flow:
            self.cur_epoch = epoch
            tqdm.write('\nEpoch {} start...'.format(self.cur_epoch))
            # 训练阶段
            train_loss = self.train_one_epoch(train_dataloader)

            # 验证阶段
            val_loss = self.val_one_epoch(val_dataloader)

            # 如果验证损失低于历史最小值，则保存模型权重
            if val_loss < self.best_val_loss and not self.skip:
                print("new best val_loss: {}, saving...".format(val_loss))
                self.best_val_loss = val_loss
                self.inner_model.save_branch_weights("./weights/", self.start_timestamp)

            # 更新进度条信息
            tqdm.write('Epoch {} - Train Loss: {:.4f} - Val Loss: {:.4f}'.format(self.cur_epoch, train_loss, val_loss))

        # 保存TensorBoard日志文件
        if not self.test:
            self.logger.writer.flush()
            self.logger.writer.close()
        if self.distribute:
            dist.destroy_process_group()

