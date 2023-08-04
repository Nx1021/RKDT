from __init__ import SCRIPT_DIR, DATASETS_DIR, WEIGHTS_DIR, CFG_DIR, _get_sub_log_dir
import matplotlib.pyplot as plt
import torch
import platform

from launcher.Trainer import Trainer
from launcher.OLDTDataset import OLDTDataset, transpose_data, collate_fn
from models.loss import LandmarkLoss
from models.OLDT import OLDT
from gen_mixed_linemod import MixLinemod_VocFormat
from utils.yaml import load_yaml, dump_yaml
import time
import numpy as np

import os
import shutil

# print("waiting...")
# time.sleep(60 * 60 *4)

torch.set_default_dtype(torch.float32)

def clear_log():
    import os, shutil
    log_dir = _get_sub_log_dir(Trainer)
    weights_dir = WEIGHTS_DIR

    weights_list = os.listdir(weights_dir) 
    weights_list = [os.path.splitext(f)[0][:14] for f in weights_list]
    weights_timestamp_list = list(filter(lambda x: x.isdigit(), weights_list))

    try:
        logs_list = os.listdir(log_dir)
    except FileNotFoundError:
        return

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
    if sys == "Windows":
        clear_log()
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
    
    flow = f"{CFG_DIR}/train_flow.yaml"
    
    
    ###
    train_dataset = OLDTDataset(data_folder, "train", MixLinemod_VocFormat)
    val_dataset = OLDTDataset(data_folder, "val", MixLinemod_VocFormat)
    loss = LandmarkLoss(cfg_file)
    model = OLDT(yolo_weight_path, cfg_file, [0])  
    load_brach_i = 0
    load_from = f"{WEIGHTS_DIR}/20230802012000branch00.pt"
    # load_from = f"{WEIGHTS_DIR}/self.base_idx"
    model.load_branch_weights(load_brach_i, load_from)
    start_epoch = 40


    if sys == "Windows":
        batch_size = 2
        # model = torch.nn.DataParallel(model)
    elif sys == "Linux":
        batch_size = 16 # * torch.cuda.device_count()
        # model = torch.nn.DataParallel(model)
    else:
        raise SystemError

    # train_dataset.set_augment_para(3, np.pi / 6)
    trainer = Trainer(model, train_dataset, val_dataset, loss, batch_size,
                      flowfile= flow,
                      distribute=False,
                      start_epoch = start_epoch
                      )
    trainer.logger.log({
        "System": sys,
        "data_folder": data_folder,
        "yolo_weight_path": yolo_weight_path,
        "cfg": cfg_file,
        "start_epoch": start_epoch,
        "batch_size": batch_size, 
        "load_from": {load_brach_i: load_from},
        "remark": "linemod"                      
                              })
    trainer.logger.log(cfg_file)
    trainer.logger.log(flow)
    trainer.train()