import OLDT_setup

import matplotlib.pyplot as plt
import torch
import platform

from launcher.Trainer import Trainer
from launcher.OLDT_Dataset import OLDT_Dataset, transpose_data, collate_fn
from launcher.utils import get_gpu_with_lowest_memory_usage
from models.loss import LandmarkLoss
from models.OLDT import OLDT
import time

import os

# print("waiting...")
# time.sleep(60 * 60 *4)




torch.set_default_dtype(torch.float32)
# Load the model
# dm = DetectionModel(cfg='yolov8l.yaml')
# image = cv2.cvtColor(cv2.imread("test.jpg"), cv2.COLOR_BGR2RGB)
# image = torch.Tensor(image)
# image = torch.unsqueeze(image, 0)
# image = torch.transpose(image, 1, 3)
# dm(image)

# get_user_config_dir()
def clear_log():
    import os, shutil
    log_dir = "./logs/Trainer_logs"
    weights_dir = "./weights"

    weights_list = os.listdir(weights_dir) 
    weights_list = [os.path.splitext(f)[0][:14] for f in weights_list]
    weights_timestamp_list = list(filter(lambda x: x.isdigit(), weights_list))

    logs_list = os.listdir(log_dir)

    to_remove = []
    for logname in logs_list:
        valid = False
        for wt in weights_timestamp_list:
            valid = valid or logname[:14] == wt
        if not valid:
            to_remove.append(logname)

    to_remove = [os.path.join(log_dir, x) for x in to_remove]
    for d in to_remove:
        shutil.rmtree(d, ignore_errors=True)

    print("{} invalid logs cleared".format(len(to_remove)))

USE_DATA_IN_SERVER = True
SERVER_PERSONAL_DIR = "/home/nerc-ningxiao/"

if __name__ == '__main__':
    sys = platform.system()
    print("system:", sys)
    if sys == "Windows":
        clear_log()


    data_folder = './datasets/morrison'
    if USE_DATA_IN_SERVER and sys == "Linux":
        data_folder = os.path.join(SERVER_PERSONAL_DIR, data_folder)
        print("use data on the server: ", data_folder)
    yolo_weight_path = "weights/best.pt"
    cfg = "cfg/config.yaml"
    ###
    train_dataset = OLDT_Dataset(data_folder, "train")
    val_dataset = OLDT_Dataset(data_folder, "val")
    loss = LandmarkLoss(cfg)
    model = OLDT(yolo_weight_path, cfg, [0])  # 替换为你自己的模型
    load_brach_i = 0
    load_from = "./weights/20230707013957branch00.pt"
    # load_from = "./weights/20230710004623branch00.pt"
    # load_from = ""
    model.load_branch_weights(load_brach_i, load_from)
    num_epochs = 400
    warm_up = int(num_epochs * 0.2)
    learning_rate = 2.0 * 1e-4
    start_epoch = 1


    if sys == "Windows":
        batch_size = 2
        # model = torch.nn.DataParallel(model)
    elif sys == "Linux":
        batch_size = 16 # * torch.cuda.device_count()
        # model = torch.nn.DataParallel(model)
    else:
        raise SystemError
    trainer = Trainer(model, train_dataset, val_dataset, loss, batch_size, num_epochs, learning_rate, warm_up, 
                      distribute=False,
                      start_epoch = start_epoch)
    trainer.logger.log({
        "System": sys,
        "data_folder": data_folder,
        "yolo_weight_path": yolo_weight_path,
        "cfg": cfg,
        "num_epochs": num_epochs,
        "start_epoch": start_epoch,
        "warm_up": warm_up,
        "learning_rate": learning_rate,
        "batch_size": batch_size, 
        "load_from": {load_brach_i: load_from},
        "remark": "class loss has weight, no pn loss P_threshold = 0.4"                      
                              })
    trainer.logger.log(cfg)
    trainer.train()