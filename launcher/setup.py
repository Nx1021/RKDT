from __init__ import SCRIPT_DIR, DATASETS_DIR, WEIGHTS_DIR, CFG_DIR, _get_sub_log_dir
import matplotlib.pyplot as plt
import torch
import platform

from launcher.Trainer import Trainer
from launcher.Predictor import OLDTPredictor, IntermediateManager
from launcher.OLDTDataset import OLDTDataset, transpose_data, collate_fn
from models.loss import LandmarkLoss
from models.OLDT import OLDT
from gen_mixed_linemod import MixLinemod_VocFormat
from posture_6d.data.dataset_format import DatasetFormat, VocFormat
from utils.yaml import load_yaml, dump_yaml
import time
import numpy as np

import os
import shutil

from typing import Type, Union

# print("waiting...")
# time.sleep(60 * 60 *4)

def setup(  
        mode = "train",      
        sub_data_dir:str = "", 
        ldt_branches:dict[int, str] = {},
        batch_size:int = 16,   
        start_epoch:int = 1,             
        dataset_format: Union[str, Type[DatasetFormat]] = VocFormat,
        use_data_on_server = True, 
        flow_file:str = f"train_flow.yaml",
        server_dataset_dir = "",
        remark = ""):
    ### input check
    if isinstance(dataset_format, str):
        _globals = globals()
        dataset_format:Type = _globals[dataset_format]
    flow_file = f"{CFG_DIR}/{flow_file}" 
    assert mode in ["train", "predict"]
    assert isinstance(sub_data_dir, str), "sub_data_dir should be a string"
    assert isinstance(ldt_branches, dict), "ldt_branches should be a dict"
    assert isinstance(batch_size, int), "batch_size should be a int"
    assert isinstance(start_epoch, int), "start_epoch should be a int"
    assert isinstance(dataset_format, Type), "dataset_format should be a Type"
    assert isinstance(use_data_on_server, bool), "use_data_on_server should be a bool"
    assert isinstance(flow_file, str) and os.path.exists(flow_file), "flow_file should be a exist file"
    if use_data_on_server:
        assert os.path.exists(server_dataset_dir), "server_dataset_dir should be a exist dir"

    ### set paths
    sys = platform.system()
    print("system:", sys)    
    DATASET, SERIAL = os.path.split(sub_data_dir)
    cfg_file = f"{CFG_DIR}/oldt_{DATASET}.yaml"
    data_cfg_file = f"{DATASETS_DIR}/{DATASET}.yaml"
    yolo_weight_path = f"{WEIGHTS_DIR}/{sub_data_dir}_best.pt"    
    if not os.path.exists(cfg_file) or\
        not os.path.exists(data_cfg_file) or\
        not os.path.exists(yolo_weight_path):
        CFG = os.path.basename(CFG_DIR)
        WEIGHTS = os.path.basename(WEIGHTS_DIR)
        DATASETS = os.path.basename(DATASETS_DIR)
        info = modify_cfg.__doc__.format(OLDT = SCRIPT_DIR, 
                                  cfg = CFG, 
                                  weights = WEIGHTS, 
                                  datasets = DATASETS,
                                  DATASET = DATASET,
                                  SERIAL = SERIAL
                                  )
        raise FileNotFoundError(info)
    else:
        modify_cfg(cfg_file, DATASET, SERIAL)

    ### set data
    server_full_data_dir = os.path.join(server_dataset_dir, sub_data_dir)
    local_full_data_dir = os.path.join(DATASETS_DIR, sub_data_dir)
    copy = False
    if use_data_on_server and sys == "Linux":
        if not os.path.exists(server_full_data_dir):
            # copy
            assert os.path.exists(local_full_data_dir), f"local data dir {local_full_data_dir} not exist"
            print(f"copy data to the server: {server_full_data_dir}, it may take a while...")
            voc = dataset_format(local_full_data_dir)
            voc.set_elements_cachemode(True)
            voc.copyto(server_full_data_dir, cover=True)
            # shutil.copytree(local_full_data_dir, server_full_data_dir)
            copy = True
        data_folder = server_full_data_dir
        print("use data on the server: ", data_folder)
    else:
        data_folder = local_full_data_dir
        print("use data on the local: ", data_folder)
    
    ### setup model
    train_dataset = OLDTDataset(data_folder, "train", dataset_format) 
    val_dataset = OLDTDataset(data_folder, "val", dataset_format)

    train_dataset.vocformat.set_elements_cachemode(True)
    val_dataset.vocformat.set_elements_cachemode(True)

    # if copy:
    #     print(f"copy data to the server: {server_full_data_dir}, it may take a while...")
    #     train_dataset.vocformat.set_elements_cachemode(True)
    #     train_dataset.vocformat.copyto(server_full_data_dir, cover=True)
    #     print(f"copy done")


    model = OLDT(yolo_weight_path, cfg_file, list(ldt_branches.keys()))  
    for load_brach_i, load_from in ldt_branches.items():
        if load_from == "":
            continue
        if '.' not in load_from:
            load_from += '.pt'
        load_from = f"{WEIGHTS_DIR}/{load_from}"
        model.load_branch_weights(load_brach_i, load_from)

    if mode == "train":
        loss = LandmarkLoss(cfg_file)

        # train_dataset.set_augment_para(3, np.pi / 6)
        trainer = Trainer(model, train_dataset, val_dataset, loss, batch_size,
                        flow_file= flow_file,
                        distribute=False,
                        start_epoch = start_epoch
                        )
        trainer.logger.log({
            "System": sys,
            "data_folder": data_folder,
            "yolo_weight_path": yolo_weight_path,
            "cfg": cfg_file,
            "batch_size": batch_size, 
            "load_from": ldt_branches,
            "start_epoch": start_epoch,
            "remark": remark                      
                                })
        trainer.logger.log(cfg_file)
        trainer.logger.log(flow_file)
        return trainer        
    else:
        predctor = OLDTPredictor(model, train_dataset, val_dataset, cfg_file, batch_size, 
                                if_postprocess=True, if_calc_error=True)
        predctor.save_imtermediate = False
        predctor.logger.log({
            "System": sys,
            "data_folder": data_folder,
            "yolo_weight_path": yolo_weight_path,
            "cfg": cfg_file,
            "batch_size": batch_size,             
            "load_from": ldt_branches,
            "remark": remark 
        })     
        return predctor



def modify_cfg(cfg_file, DATASET, SERIAL):
    '''
    you should have files like:
    - {OLDT}
      | - {cfg}
      |  | - oldt_{DATASET}.yaml
      | - {weights}
      |  | - {DATASET}
      |  |  | - {SERIAL}_best.pt
      | - {datasets}
      |  | - {DATASET}.yaml
      |  | - {DATASET}
      |  |  | - {SERIAL}
    '''
    cfg = load_yaml(cfg_file)
    cfg["yolo_override"]["model"] = f"weights/{DATASET}/{SERIAL}_best.pt"
    dump_yaml(cfg_file, cfg)

    #
    cfg = load_yaml(f"{DATASETS_DIR}/{DATASET}.yaml")
    cfg["path"] = f"./{DATASET}/{SERIAL}"
    dump_yaml(f"{DATASETS_DIR}/{DATASET}.yaml", cfg)

