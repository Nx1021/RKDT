from __init__ import SCRIPT_DIR, DATASETS_DIR, CFG_DIR, WEIGHTS_DIR, LOGS_DIR, _get_sub_log_dir, SERVER_DATASET_DIR
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

if __name__ == "__main__":
    cfg_file = f"{CFG_DIR}/oldt_morrison_real_voc.yaml"
    sys = platform.system()
    # torch.cuda.set_device("cuda:0")

    setup_paras = load_yaml(cfg_file)["setup"]

    sys = platform.system()
    if sys == "Windows":
        batch_size = 2
        # model = torch.nn.DataParallel(model)
    elif sys == "Linux":
        batch_size = 32 # * torch.cuda.device_count()
        # model = torch.nn.DataParallel(model)

    for i in range(7, 9):
        # setup_paras["ldt_branches"] = {i: "20231015064605branch_ldt_02.pt"}
        setup_paras["ldt_branches"] = {i: ""}
        setup_paras["batch_size"] = batch_size
        setup_paras["sub_data_dir"] = f"morrison_real_voc/"

        trainer = setup("train",
                        detection_base_weight=f"{WEIGHTS_DIR}/morrison_real_voc/best.pt" ,
                            **setup_paras)
        trainer.train_dataset.vocformat.spliter_group.split_mode = "posture"
        trainer.val_dataset.vocformat.spliter_group.split_mode = "posture"
        # format.posture_spliter.set_split_mode(f"obj_{str(i).rjust(2, '0')}")
        # format.gen_posture_log(0.5)
        # trainer.train_dataset.vocformat.spliter_group.copyto(os.path.join(setup_paras["server_dataset_dir"], "morrison_real_voc", "ImageSets"))
        trainer.train()

        trainer = None
