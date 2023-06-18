from typing import Union
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import datetime
import numpy as np
import sys

class Trainer:
    def __init__(self, 
                 model: Union[OLDT, torch.nn.DataParallel], 
                 train_dataset, 
                 val_dataset, 
                 criterion, 
                 batch_size, 
                 num_epochs: int, 
                 init_lr, 
                 warmup_epoch, 
                 distribute=False,
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

        self.optimizer = optim.Adam(self.model.parameters(), lr=init_lr)
        self.warmup_scheduler = LambdaLR(self.optimizer, lr_lambda=lambda step: min(step / self.warmup_steps, 1.0))
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, self.total_steps - self.warmup_steps)
        self.criterion = criterion
        
        self.best_val_loss = float('inf')

        current_time = datetime.datetime.now()
        self.start_timestamp = current_time.strftime("%Y%m%d%H%M%S")

        self.test = test

        if not test:
            self.log_dir = "./logs/" + self.start_timestamp + sys
            self.writer = SummaryWriter(log_dir=self.log_dir)

        self.cur_epoch = 0