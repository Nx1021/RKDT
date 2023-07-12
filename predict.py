import OLDT_setup

from launcher.Predictor import OLDTPredictor, IntermediateManager

from models.OLDT import OLDT
from launcher.OLDT_Dataset import OLDT_Dataset

import platform
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from PIL import Image
from typing import Iterable
import os

def find_record(weight_path, default = "cfg/config.yaml"):
    stamp = os.path.splitext(os.path.split(weight_path)[-1])[0][:14]
    dirs = os.listdir("./logs/Trainer_logs")
    for d in dirs:
        if d[:14] == stamp:
            cfg_path = os.path.join("./logs/Trainer_logs", d, "config.yaml")
            if os.path.exists(cfg_path):
                return cfg_path
    return default

if __name__ == '__main__':
    sys = platform.system()
    print("system:", sys)

    data_folder = './datasets/morrison'
    yolo_weight_path = "weights/best.pt"
    ###
    train_dataset = OLDT_Dataset(data_folder, "train")
    val_dataset = OLDT_Dataset(data_folder, "val")
    load_brach_i = 0
    load_from = "./weights/20230710202254branch00.pt"
    # cfg = find_record(load_from, "cfg/config.yaml")
    cfg = "cfg/config.yaml"
    print("config file: ", cfg)
    model = OLDT(yolo_weight_path, cfg, [load_brach_i])  # 替换为你自己的模型    
    model.load_branch_weights(load_brach_i, load_from)
    model.set_mode("predict")

    if sys == "Windows":
        batch_size = 4
        # model = torch.nn.DataParallel(model)
    elif sys == "Linux":
        batch_size = 32 # * torch.cuda.device_count()
        # model = torch.nn.DataParallel(model)

    remark = "new_variable_length"
    intermediate_from = ""
    predctor = OLDTPredictor(model, cfg, remark, batch_size, if_postprocess=True, if_calc_error=True, intermediate_from = intermediate_from)
    predctor.save_imtermediate = False
    predctor.logger.log({
        "System": sys,
        "data_folder": data_folder,
        "yolo_weight_path": yolo_weight_path,
        "cfg": cfg,
        "load_from": {load_brach_i: load_from},
        "remark": remark 
    })     
    if intermediate_from:
        predctor.postprocess_from_intermediate(plot_outlier=True)
    else:
        predctor.predict_from_dataset(val_dataset)


    # intermediate_manager = IntermediateManager("./logs/intermediate_output")
    # predctor = OLDTPredictor(model, cfg, batch_size, if_postprocess=False, if_calc_error=False, intermediate_manager = intermediate_manager)
    # predctor.postprocess_mode = 'v'
    # predctor.predict_from_dataset(val_dataset)