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
    # cfg_file = f"{CFG_DIR}/oldt_morrison_mix.yaml"
    cfg_file = f"{CFG_DIR}/oldt_morrison_mix_single.yaml"
    # setup_paras = load_yaml(cfg_file)["setup"]

    # sys = platform.system()
    # if sys == "Windows":
    #     batch_size = 4
    #     # model = torch.nn.DataParallel(model)
    # elif sys == "Linux":
    #     batch_size = 32 # * torch.cuda.device_count()
    #     # model = torch.nn.DataParallel(model)
    # setup_paras["sub_data_dir"] = "morrison_mix/"
    # setup_paras["ldt_branches"] = {0: "20230923080858branch_ldt_00.pt"}
    # setup_paras["batch_size"] = batch_size

    # predictor = setup("predict", 
    #     detection_base_weight=f"{WEIGHTS_DIR}/morrison_mix_single/best.pt" ,
    #     **setup_paras)
    # dataset:Mix_VocFormat = predictor.train_dataset.vocformat
    # dataset.spliter_group.set_cur_spliter_name("posture")
    # predictor.predict_val()

    for i in range(0,1):
        setup_paras = load_yaml(cfg_file)["setup"]

        sys = platform.system()
        if sys == "Windows":
            batch_size = 4
            # model = torch.nn.DataParallel(model)
        elif sys == "Linux":
            batch_size = 32 # * torch.cuda.device_count()
            # model = torch.nn.DataParallel(model)
        setup_paras["ldt_branches"] = {i: "20231006050449branch_ldt_00.pt"}
        setup_paras["batch_size"] = batch_size
        setup_paras["sub_data_dir"] = f"morrison_mix_single/{str(i).rjust(6, '0')}"

        predictor = setup("predict",
                        detection_base_weight=f"{WEIGHTS_DIR}/morrison_mix_single/best.pt" ,
                         **setup_paras)
        predictor.train_dataset.vocformat.spliter_group.set_cur_spliter_name("posture")
        predictor.val_dataset.vocformat.spliter_group.set_cur_spliter_name("posture")
        # format.posture_spliter.set_split_mode(f"obj_{str(i).rjust(2, '0')}")
        # format.gen_posture_log(0.5)
        # trainer.train_dataset.vocformat.spliter_group.copyto(os.path.join(setup_paras["server_dataset_dir"], "morrison_mix_single", "ImageSets"))
        predictor.predict_val(plot_outlier = False)


        predictor = None