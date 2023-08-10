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
import shutil
from utils.yaml import load_yaml, dump_yaml
from gen_mixed_linemod import MixLinemod_VocFormat

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

def modify_cfg():
    cfg = load_yaml(cfg_file)
    cfg["yolo_override"]["model"] = f"weights/{DATASET}/{SERIAL}_best.pt"
    dump_yaml(cfg_file, cfg)

    #
    cfg = load_yaml(f"{DATASETS_DIR}/{DATASET}.yaml")
    cfg["path"] = f"./{DATASET}/{SERIAL}"
    dump_yaml(f"{DATASETS_DIR}/{DATASET}.yaml", cfg)

SUBDATA_DIR = "linemod_mix/000000" #modify this

DATASET, SERIAL = os.path.split(SUBDATA_DIR)
yolo_weight_path = f"{WEIGHTS_DIR}/{SUBDATA_DIR}_best.pt"
cfg_file = f"{CFG_DIR}/oldt_{DATASET}.yaml"

USE_DATA_IN_SERVER = True
SERVER_DATASET_DIR = "/home/nerc-ningxiao/datasets/" + SUBDATA_DIR

if __name__ == "__main__":
    sys = platform.system()
    print("system:", sys)
    modify_cfg()

    data_folder = f"{DATASETS_DIR}/" + SUBDATA_DIR
    if USE_DATA_IN_SERVER and sys == "Linux":
        if not os.path.exists(SERVER_DATASET_DIR):
            # copy
            print(f"copy data to the server: {data_folder}, it may take a while...")
            shutil.copytree(data_folder, SERVER_DATASET_DIR)
            print(f"done")
        data_folder = SERVER_DATASET_DIR
        print("use data on the server: ", data_folder)
    
    ###
    train_dataset = OLDTDataset(data_folder, "train", MixLinemod_VocFormat)
    val_dataset = OLDTDataset(data_folder, "val", MixLinemod_VocFormat)

    load_brach_i = 0
    load_from = f"{WEIGHTS_DIR}/20230802012000branch00.pt"
    # cfg = find_record(load_from, f"{CFG_DIR}/config.yaml")
    print("config file: ", cfg_file)
    model = OLDT(yolo_weight_path, cfg_file, [load_brach_i])  # 替换为你自己的模型    
    model.load_branch_weights(load_brach_i, load_from)
    model.set_mode("train")

    if sys == "Windows":
        batch_size = 4
        # model = torch.nn.DataParallel(model)
    elif sys == "Linux":
        batch_size = 32 # * torch.cuda.device_count()
        # model = torch.nn.DataParallel(model)

    ###
    # flow = f"{CFG_DIR}/train_flow.yaml"
    # loss = LandmarkLoss(cfg_file)
    # trainer = Trainer(model, train_dataset, val_dataset, loss, batch_size,
    #                 flowfile= flow,
    #                 distribute=False,
    #                 start_epoch = 1
    #                 )
    # trainer.train()
    ###


    remark = "new_variable_length"
    intermediate_from = ""
    predctor = OLDTPredictor(model, cfg_file, remark, batch_size, 
                             if_postprocess=True, if_calc_error=True, 
                             intermediate_from = intermediate_from)
    predctor.save_imtermediate = False
    predctor.logger.log({
        "System": sys,
        "data_folder": data_folder,
        "yolo_weight_path": yolo_weight_path,
        "cfg": cfg_file,
        "load_from": {load_brach_i: load_from},
        "remark": remark 
    })     
    if intermediate_from:
        predctor.postprocess_from_intermediate(plot_outlier=True)
    else:
        predctor.predict_from_dataset(val_dataset)
        # predctor.clear()
        # predctor.predict_from_dataset(train_dataset)


    # intermediate_manager = IntermediateManager(f"{LOGS_DIR}/intermediate_output")
    # predctor = OLDTPredictor(model, cfg, batch_size, if_postprocess=False, if_calc_error=False, intermediate_manager = intermediate_manager)
    # predctor.postprocess_mode = 'v'
    # predctor.predict_from_dataset(val_dataset)