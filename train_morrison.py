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
    cfg_file = f"{CFG_DIR}/oldt_morrison_mix_single.yaml"
    sys = platform.system()
    # torch.cuda.set_device("cuda:0")
    if sys == "Linux":
        # for i in range(0, 9):
        #     for elem in ['bbox_3ds', 'depths', 'depth_scale', 'images', 'intr', 'labels', 'landmarks', 'masks', 'trans_vecs', 'visib_fracts']:
        #         orig_image_map_path = f"{DATASETS_DIR}/morrison_mix_single/{str(i).rjust(6, '0')}/{elem}/data_info_map.elmm"
        #         server_image_map_path = f"/home/nerc-ningxiao/datasets/morrison_mix_single/{str(i).rjust(6, '0')}/{elem}/data_info_map.elmm"
        #         if os.path.exists(server_image_map_path):
        #             os.remove(server_image_map_path)
        #         shutil.copy(orig_image_map_path, server_image_map_path)
        # for i in [1, 4, 2,3,5,6,7]:
        #     server_dir = f"{SERVER_DATASET_DIR}/morrison_mix_single/{str(i).rjust(6, '0')}"
        #     if not os.path.exists(server_dir):
        #         vm = Mix_VocFormat(f"{DATASETS_DIR}/morrison_mix_single/{str(i).rjust(6, '0')}")
        #         vm.images_elements.cache_priority = False
        #         vm.depth_elements.cache_priority = False
        #         vm.masks_elements.cache_priority = False
        #         vm.labels_elements.cache_priority = False
        #         vm.copyto(f"{SERVER_DATASET_DIR}/morrison_mix_single/{str(i).rjust(6, '0')}")
        pass
            # orig_spliter_dir = f"{DATASETS_DIR}/morrison_mix_single/{str(i).rjust(6, '0')}/ImageSets"
            # server_spliter_dir = f"/home/nerc-ningxiao/datasets/morrison_mix_single/{str(i).rjust(6, '0')}/ImageSets"
            # if os.path.exists(server_spliter_dir):
            #     shutil.rmtree(server_spliter_dir)
            # shutil.copytree(orig_spliter_dir, server_spliter_dir)


    # for i in [3]:
    #     if sys == "Linux":
    #         server_dir = f"{SERVER_DATASET_DIR}/morrison_mix_single/{str(i).rjust(6, '0')}"
    #         if not os.path.exists(server_dir):
    #             vm = Mix_VocFormat(f"{DATASETS_DIR}/morrison_mix_single/{str(i).rjust(6, '0')}")
    #             vm.images_elements.cache_priority = False
    #             vm.depth_elements.cache_priority = False
    #             vm.masks_elements.cache_priority = False
    #             vm.labels_elements.cache_priority = False
    #             vm.copyto(f"{SERVER_DATASET_DIR}/morrison_mix_single/{str(i).rjust(6, '0')}")
    #         vm = Mix_VocFormat(f"{DATASETS_DIR}/morrison_mix_single/{str(i).rjust(6, '0')}")
    #         vm.spliter_group.copyto(f"{SERVER_DATASET_DIR}/morrison_mix_single/{str(i).rjust(6, '0')}/ImageSets", cover=True)

    #     setup_paras = load_yaml(cfg_file)["setup"]

    #     sys = platform.system()
    #     if sys == "Windows":
    #         batch_size = 2
    #         # model = torch.nn.DataParallel(model)
    #     elif sys == "Linux":
    #         batch_size = 32 # * torch.cuda.device_count()
    #         # model = torch.nn.DataParallel(model)
    #     setup_paras["ldt_branches"] = {i: ""}
    #     setup_paras["batch_size"] = batch_size
    #     setup_paras["sub_data_dir"] = f"morrison_mix_single/{str(i).rjust(6, '0')}"

    #     trainer = setup("train",
    #                     detection_base_weight=f"{WEIGHTS_DIR}/morrison_mix_single/best_{str(i).rjust(2, '0')}.pt" ,
    #                      **setup_paras)
    #     trainer.train_dataset.vocformat.spliter_group.set_cur_spliter_name("posture")
    #     trainer.val_dataset.vocformat.spliter_group.set_cur_spliter_name("posture")
    #     # format.posture_spliter.set_split_mode(f"obj_{str(i).rjust(2, '0')}")
    #     # format.gen_posture_log(0.5)
    #     # trainer.train_dataset.vocformat.spliter_group.copyto(os.path.join(setup_paras["server_dataset_dir"], "morrison_mix_single", "ImageSets"))
    #     trainer.train()

    #     trainer = None

    # region real_voc
    if sys == "Linux":
        server_dir = f"{SERVER_DATASET_DIR}/morrison_voc_real/"
        if not os.path.exists(server_dir):
            vm = Mix_VocFormat(f"{DATASETS_DIR}/morrison_voc_real/")
            vm.images_elements.cache_priority = False
            vm.depth_elements.cache_priority = False
            vm.masks_elements.cache_priority = False
            vm.labels_elements.cache_priority = False
            vm.copyto(f"{SERVER_DATASET_DIR}/morrison_voc_real")
        vm = Mix_VocFormat(f"{DATASETS_DIR}/morrison_voc_real")
        vm.spliter_group.copyto(f"{SERVER_DATASET_DIR}/morrison_voc_real/ImageSets", cover=True)

    setup_paras = load_yaml(cfg_file)["setup"]

    sys = platform.system()
    if sys == "Windows":
        batch_size = 2
        # model = torch.nn.DataParallel(model)
    elif sys == "Linux":
        batch_size = 32 # * torch.cuda.device_count()
        # model = torch.nn.DataParallel(model)

    for i in range(9):
        setup_paras["ldt_branches"] = {i: ""}
        setup_paras["batch_size"] = batch_size
        setup_paras["sub_data_dir"] = f"morrison_voc_real"

        trainer = setup("train",
                        detection_base_weight=f"{WEIGHTS_DIR}/morrison_voc_real/best.pt" ,
                            **setup_paras)
        trainer.train_dataset.vocformat.spliter_group.set_cur_spliter_name("posture")
        trainer.val_dataset.vocformat.spliter_group.set_cur_spliter_name("posture")
        # format.posture_spliter.set_split_mode(f"obj_{str(i).rjust(2, '0')}")
        # format.gen_posture_log(0.5)
        # trainer.train_dataset.vocformat.spliter_group.copyto(os.path.join(setup_paras["server_dataset_dir"], "morrison_voc_real", "ImageSets"))
        trainer.train()

        trainer = None

    # endregion