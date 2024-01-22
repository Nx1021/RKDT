from __init__ import SCRIPT_DIR, DATASETS_DIR, CFG_DIR, WEIGHTS_DIR, LOGS_DIR, _get_sub_log_dir
from launcher.Predictor import OLDTPredictor, IntermediateManager
from launcher.Trainer import Trainer

from models.OLDT import OLDT
from launcher.OLDTDataset import OLDTDataset
from launcher.setup import setup

import platform
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from PIL import Image
from typing import Iterable
import os
import shutil
from utils.yaml import load_yaml, dump_yaml

from posture_6d.data.dataset_example import VocFormat_6dPosture

if __name__ == "__main__":
    # setup
    cfg_file = f"{CFG_DIR}/oldt_linemod_mix_new.yaml"

    # torch.cuda.set_device("cuda:0")
    
    torch.cuda.set_device("cuda:0") # 0, 4, 5 is in 1

    for idx in [10, 11]: # 8, 9, 10, 11
        setup_paras = load_yaml(cfg_file)["setup"]

        sys = platform.system()
        if sys == "Windows":
            batch_size = 4
            # model = torch.nn.DataParallel(model)
        elif sys == "Linux":
            batch_size = 32 # * torch.cuda.device_count()
            # model = torch.nn.DataParallel(model)
        # setup_paras["ldt_branches"] = {i: f"linemod_mix_new/{weights[i]}"}
        setup_paras["ldt_branches"] = {idx: ""}
        setup_paras["batch_size"] = batch_size
        setup_paras["sub_data_dir"] = f"linemod_mix_new/"

        trainer = setup("detection",

                        # detection_base_weight=f"{WEIGHTS_DIR}/linemod_mix/{str(idx).rjust(6, '0')}_best.pt" ,
                        detection_base_weight=None ,
                        detection_active_class_id= [idx], 
                        **setup_paras)
        
        # trainer.train_dataset.vocformat.spliter_group.split_mode = "aug_posture"
        # print(trainer.train_dataset.vocformat.spliter_group.get_cluster("default"))
        # trainer.train_dataset.vocformat.spliter_group.add_spliter("obj_{}_posture".format(str(idx).rjust(2, "0")), subsets=["train", "val"])

        # spliter = trainer.train_dataset.vocformat.spliter_group.get_cluster("obj_{}_posture".format(str(idx).rjust(2, "0")))
        # spliter.exclusive = False
        # print(spliter)

        # trainer.train()
        # trainer = None

    # for i in range(1, 15):
    #     print(i)
    #     for name, mapname in zip(["bbox_3ds", "depth_scale", "intr", "landmarks", "trans_vecs", "visib_fracts"], 
    #                              ["bbox_3d",  "depth_scale", "intr", "landmarks", "extr_vecs",  "visib_fracts"]):
    #         os.remove(f"/home/nerc-ningxiao/datasets/linemod_mix/{str(i).rjust(6, '0')}/{name}/{mapname}.datamap")
    #         shutil.copy(f"{DATASETS_DIR}/linemod_mix/{str(i).rjust(6, '0')}/{name}/{mapname}.datamap", 
    #                     f"/home/nerc-ningxiao/datasets/linemod_mix/{str(i).rjust(6, '0')}/{name}/{mapname}.datamap")
        # voc:VocFormat_6dPosture = VocFormat_6dPosture(f"{DATASETS_DIR}/linemod_mix/{str(i).rjust(6, '0')}")
        
        # voc_server:VocFormat_6dPosture = VocFormat_6dPosture(f"/home/nerc-ningxiao/datasets/linemod_mix/{str(i).rjust(6, '0')}")
        # voc_server.copy_from_simplified(voc, cover=True, force=True)