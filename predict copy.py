from __init__ import SCRIPT_DIR, DATASETS_DIR, CFG_DIR, WEIGHTS_DIR, LOGS_DIR, _get_sub_log_dir
from launcher.Predictor import OLDTPredictor, IntermediateManager
from launcher.Trainer import Trainer

from models.OLDT import OLDT
from launcher.OLDTDataset import OLDTDataset

import platform
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from PIL import Image
from typing import Iterable
import os

def find_record(weight_path, default = f"{CFG_DIR}/config.yaml"):
    stamp = os.path.splitext(os.path.split(weight_path)[-1])[0][:14]
    log_root = _get_sub_log_dir(Trainer)
    dirs = os.listdir(log_root)
    for d in dirs:
        if d[:14] == stamp:
            cfg_path = os.path.join(log_root, d, "config.yaml")
            if os.path.exists(cfg_path):
                return cfg_path
    return default

if __name__ == '__main__':
    sys = platform.system()
    print("system:", sys)

    data_folder = f"{DATASETS_DIR}/linemod/000000"
    yolo_weight_path = f"{WEIGHTS_DIR}/linemod_000000_best.pt"
    ###
    train_dataset = OLDTDataset(data_folder, "train")
    val_dataset = OLDTDataset(data_folder, "val")
    load_brach_i = 0
    load_from = f"{WEIGHTS_DIR}/20230725204657branch00.pt"
    # cfg = find_record(load_from, f"{CFG_DIR}/config.yaml")
    cfg = f"{CFG_DIR}/config_linemod_000001.yaml"
    print("config file: ", cfg)
    model = OLDT(yolo_weight_path, cfg, [load_brach_i])  # 替换为你自己的模型    
    model.load_branch_weights(load_brach_i, load_from)
    model.set_mode("predict")

    if sys == "Windows":
        batch_size = 16
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
        predctor.predict_from_dataset(train_dataset)


    # intermediate_manager = IntermediateManager(f"{LOGS_DIR}/intermediate_output")
    # predctor = OLDTPredictor(model, cfg, batch_size, if_postprocess=False, if_calc_error=False, intermediate_manager = intermediate_manager)
    # predctor.postprocess_mode = 'v'
    # predctor.predict_from_dataset(val_dataset)