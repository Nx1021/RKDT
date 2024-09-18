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

if __name__ == "__main__":
    cfg_file = f"{CFG_DIR}/oldt_linemod_mix.yaml"
    setup_paras = load_yaml(cfg_file)["setup"]

    sys = platform.system()
    if sys == "Windows":
        batch_size = 16
        # model = torch.nn.DataParallel(model)
    elif sys == "Linux":
        batch_size = 32 # * torch.cuda.device_count()
        # model = torch.nn.DataParallel(model)

    # weights = {
    #     0: "20240102071157branch_ldt_00.pt",
    #     1: "20230815074456branch_ldt_01.pt",
    #     2: "20230831080343branch_ldt_02.pt",
    #     3: "20230818011104branch_ldt_03.pt",
    #     4: "20230819081237branch_ldt_04.pt",
    #     5: "20230819081450branch_ldt_05.pt",
    #     6: "20230823005717branch_ldt_06.pt",
    #     # 7: "20230826185557branch_ldt_07.pt",
    #     7: "20240105130341branch_ldt_07.pt",
    #     8: "20230823010935branch_ldt_08.pt",
    #     9: "20230826200842branch_ldt_09.pt",
    #     10: "20230823011027branch_ldt_10.pt",
    #     11: "20230826191856branch_ldt_11.pt",
    #     # 12: "20230823011323branch_ldt_12.pt",
    #     12: "20240108081240branch_ldt_12.pt",
    #     # 13: "20230826165015branch_ldt_13.pt",
    #     13: "20240105192423branch_ldt_13.pt",
    #     14: "20230902185318branch_ldt_14.pt"
    # }

    weights = {
        0: "20240122032206branch_ldt_00.pt",
        1: "20230815074456branch_ldt_01.pt",
        2: "20230831080343branch_ldt_02.pt",
        3: "20230818011104branch_ldt_03.pt",
        4: "20240131022319branch_ldt_04.pt",
        5: "20240126020130branch_ldt_05.pt",
        6: "20230823005717branch_ldt_06.pt",
        # 7: "20230826185557branch_ldt_07.pt",
        7: "20240203015411branch_ldt_07.pt",
        8: "20240205152649branch_ldt_08.pt",
        9: "20240207152358branch_ldt_09.pt",
        10: "20240209162253branch_ldt_10.pt",
        11: "20230826191856branch_ldt_11.pt",
        # 12: "20230823011323branch_ldt_12.pt",
        12: "20240108081240branch_ldt_12.pt",
        # 13: "20230826165015branch_ldt_13.pt",
        13: "20240105192423branch_ldt_13.pt",
        14: "20230902185318branch_ldt_14.pt"
    }

    wp = "linemod_mix" #"linemod_mix"

    for k, v in weights.items():
        if (k == 2) or (k == 6):
            continue
        setup_paras["sub_data_dir"] = "linemod_mix/{}".format(str(k).rjust(6, '0'))
        setup_paras["ldt_branches"] = {k: f"{wp}/{v}"}
        setup_paras["batch_size"] = batch_size
        setup_paras["use_data_on_server"] = setup_paras["use_data_on_server"] and sys == "Linux"

        predictor = setup("predict", 
                        detection_base_weight=f"{WEIGHTS_DIR}/{wp}/{str(k).rjust(6, '0')}_best.pt" ,
                        **setup_paras)
        predictor.train_dataset.vocformat.spliter_group.split_mode = "posture"
        predictor.val_dataset.vocformat.spliter_group.split_mode = "posture"
        # predictor.postprocesser._use_bbox_area_assumption = True
        predictor.postprocesser._use_bbox_area_assumption = False
        predictor._use_depth = False
        predictor.postprocesser.depth_scale = 1.0
        predictor.if_calc_error = True
        # predictor.num_workers = 0
        predictor.predict_val()

        print(np.mean(predictor._tvec_error, axis=0))
        print(np.std(predictor._tvec_error, axis=0))

