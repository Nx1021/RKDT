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
from posture_6d.data.dataset_format import Mix_VocFormat

if __name__ == "__main__":
    cfg_file = f"{CFG_DIR}/oldt_linemod_mix.yaml"
    # torch.cuda.set_device("cuda:0")
    for i in [2, 14]:
        setup_paras = load_yaml(cfg_file)["setup"]

        sys = platform.system()
        if sys == "Windows":
            batch_size = 4
            # model = torch.nn.DataParallel(model)
        elif sys == "Linux":
            batch_size = 32 # * torch.cuda.device_count()
            # model = torch.nn.DataParallel(model)
        setup_paras["ldt_branches"] = {i: ""}
        setup_paras["batch_size"] = batch_size
        setup_paras["sub_data_dir"] = f"linemod_mix/{str(i).rjust(6, '0')}"

        trainer = setup("train", **setup_paras)
        trainer.train()

        trainer = None