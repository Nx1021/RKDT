from launcher.OLDT_Dataset import transpose_data

import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

import numpy as np
from models.OLDT import OLDT
from models.loss import LandmarkLoss
from models.results import ImagePosture
from launcher.OLDT_Dataset import collate_fn
import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime
import platform

from typing import Union
sys = platform.system()
if sys == "Windows":
    TESTFLOW = False
else:
    TESTFLOW = False


class LandmarkLossManager():
    def __init__(self) -> None:
        self.loss_sum = 0.0
        self.detect_num_sum = 0

    def update(self, loss, detect_num):
        self.loss_sum += loss
        self.detect_num_sum += detect_num

    def clear(self):
        self.loss_sum = 0.0
        self.detect_num_sum = 0

    @property
    def mean(self):
        if self.detect_num_sum > 0:
            return self.loss_sum / self.detect_num_sum
        else:
            return float('inf')

class Trainer:
    def __init__(self, 
                 model:Union[OLDT, torch.nn.DataParallel], 
                 train_dataset, 
                 val_dataset, 
                 criterion, 
                 batch_size, 
                 num_epochs:int, 
                 init_lr, 
                 warmup_epoch, 
                 distribute = False,
                 test=False):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.init_lr = init_lr
        self.epoch_step = int(np.ceil(len(train_dataset) / batch_size))
        self.total_steps = int(self.epoch_step * self.num_epochs)
        self.warmup_steps = int(warmup_epoch * self.epoch_step)
        self.test = test

        if distribute:
            # 初始化分布式环境
            dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456', world_size=torch.cuda.device_count(), rank=torch.cuda.current_device())
            self.model = torch.nn.parallel.DistributedDataParallel(self.model)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=init_lr)  # 初始化优化器
        self.warmup_scheduler = LambdaLR(self.optimizer, lr_lambda=lambda step: min(step / self.warmup_steps, 1.0))
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.total_steps - self.warmup_steps)
        self.criterion = criterion
        
        self.best_val_loss = float('inf')  # 初始化最佳验证损失为正无穷大

        current_time = datetime.datetime.now()
        self.start_timestamp = current_time.strftime("%Y%m%d%H%M%S")

        self.test = test
        # 创建TensorBoard的SummaryWriter对象，指定保存日志文件的目录
        if not test:
            self.log_dir = "./logs/" + self.start_timestamp + sys  # TensorBoard日志文件保存目录
            self.writer = SummaryWriter(log_dir=self.log_dir)  # 创建SummaryWriter对象

        self.cur_epoch = 0

    @property
    def inner_model(self):
        if isinstance(self.model, torch.nn.DataParallel):
            return self.model.module
        else:
            return self.model

    def pre_process(self, batch:list[ImagePosture]):
        images, keypoints, labels, bboxes_n, trans_vecs = transpose_data([x.split() for x in batch])
        keypoints = self._to_device(keypoints)
        labels = self._to_device(labels)
        bboxes_n = self._to_device(bboxes_n)
        return images, keypoints, labels, bboxes_n


    def _to_device(self, tensor_list:list[torch.Tensor]):
        return [torch.from_numpy(np.array(x)).to(self.device) for x in tensor_list]

    def check_has_target(self, gt_labels:list[torch.Tensor], active_class:list[int]):
        for lables in gt_labels:
            for l in lables:
                if l in active_class:
                    return True
        return False

    def train_once(self, dataloader):
        self.model.train()
        ldmk_loss_mngr = LandmarkLossManager()
        ldmk_last_loss_mngr = LandmarkLossManager()

        progress = tqdm(dataloader, desc='Training', leave=True)
        for image_posture in progress:
            images, gt_keypoints, gt_labels, gt_bboxes_n = self.pre_process(image_posture)
            if not self.check_has_target(gt_labels, self.inner_model.landmark_branch_classes):
                continue
            # 前向传播
            detection_results = self.model(images)
            loss, last_loss, detect_num = self.criterion(gt_keypoints, gt_bboxes_n, detection_results)

            # 反向传播和优化
            if detect_num > 0:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 更新学习率
                if self.cur_step < self.warmup_steps:
                    self.warmup_scheduler.step()
                else:
                    self.lr_scheduler.step()

                self.cur_step += 1

                # 计算批次的准确率和损失
                ldmk_loss_mngr.update(loss.item() * detect_num, detect_num)
                ldmk_last_loss_mngr.update(last_loss.item() * detect_num, detect_num)

                # 更新进度条信息
                progress.set_postfix({'Loss': "{:>8.4f}".format(ldmk_loss_mngr.mean), "lr": self.optimizer.param_groups[0]["lr"]})
            if TESTFLOW:
                break

            

        # 计算平均训练损失
        # if not self.test:
        #     print('Train Mean Loss: {:>8.4f}'.format(ldmk_loss_mngr.mean))

        # 将train_loss写入TensorBoard日志文件
        self.writer.add_scalar("Train Loss", ldmk_loss_mngr.mean, self.cur_epoch)
        self.writer.add_scalar("Train Last Decoder Loss", ldmk_last_loss_mngr.mean, self.cur_epoch)
        self.writer.add_scalar("Learning rate", self.optimizer.param_groups[0]["lr"], self.cur_epoch)

        return ldmk_loss_mngr.mean

    def val_once(self, dataloader):
        self.model.eval()
        ldmk_loss_mngr = LandmarkLossManager()
        ldmk_last_loss_mngr = LandmarkLossManager()

        progress = tqdm(dataloader, desc='Validation', leave=True)
        with torch.no_grad():
            for image_posture in progress:
                images, gt_keypoints, gt_labels, gt_bboxes_n = self.pre_process(image_posture)
                if not self.check_has_target(gt_labels, self.model.landmark_branch_classes):
                    continue
                # 前向传播
                detection_results = self.model(images)
                loss, last_loss, detect_num = self.criterion(gt_keypoints, gt_bboxes_n, detection_results)

                # 计算批次的损失
                if detect_num > 0:
                    ldmk_loss_mngr.update(loss.item() * detect_num, detect_num)
                    ldmk_last_loss_mngr.update(last_loss.item() * detect_num, detect_num)
                    progress.set_postfix({'Loss': "{:>8.4f}".format(ldmk_loss_mngr.mean)})
                if TESTFLOW:
                    break
                
        # # 计算平均验证损失
        # print('Val Mean Loss: {:>8.4f}'.format(ldmk_loss_mngr.mean))

        # 将val_loss写入TensorBoard日志文件
        if not self.test:
            self.writer.add_scalar("Val Loss", ldmk_loss_mngr.mean, self.cur_epoch)
            self.writer.add_scalar("Val Last Decoder Loss", ldmk_last_loss_mngr.mean, self.cur_epoch)

        return ldmk_loss_mngr.mean

    def train(self):
        print("start to train... time:{}".format(self.start_timestamp))
        self.cur_epoch = 0
        self.cur_step = 0
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)

        # 用于保存每个epoch的train_loss和val_loss
        train_losses = []
        val_losses = []

        for epoch in range(self.num_epochs):
            self.cur_epoch = epoch + 1
            tqdm.write('\nEpoch {} start...'.format(self.cur_epoch))
            # 训练阶段
            if not self.test:
                train_loss = self.train_once(train_dataloader)
            else:
                train_loss = 0.0

            # 验证阶段
            val_loss = self.val_once(val_dataloader)

            # 如果验证损失低于历史最小值，则保存模型权重
            if val_loss < self.best_val_loss:
                print("new best val_loss: {}, saving...".format(val_loss))
                self.best_val_loss = val_loss
                self.inner_model.save_branch_weights("./weights/", self.start_timestamp)

            # 更新进度条信息
            tqdm.write('Epoch {} - Train Loss: {:.4f} - Val Loss: {:.4f}'.format(self.cur_epoch, train_loss, val_loss))

            # 将train_loss和val_loss添加到列表中
            train_losses.append(train_loss)
            val_losses.append(val_loss)

        # 保存TensorBoard日志文件
        if not self.test:
            self.writer.flush()
            self.writer.close()

        dist.destroy_process_group()

