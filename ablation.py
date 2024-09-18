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
import pickle
import pandas as pd
from utils.yaml import load_yaml, dump_yaml
from post_processer.PostProcesser import create_mesh_manager, create_pnpsolver, PostProcesser
from posture_6d.core.utils import JsonIO



cfg_file = f"{CFG_DIR}/oldt_linemod_mix.yaml"
setup_paras = load_yaml(cfg_file)["setup"]

sys = platform.system()
if sys == "Windows":
    batch_size = 4
    # model = torch.nn.DataParallel(model)
elif sys == "Linux":
    batch_size = 32 # * torch.cuda.device_count()
    # model = torch.nn.DataParallel(model)

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

def _setup(k, weight):
    setup_paras["sub_data_dir"] = "linemod_mix/{}".format(str(k).rjust(6, '0'))
    setup_paras["ldt_branches"] = {k: f"{wp}/{weight}"}
    setup_paras["batch_size"] = batch_size

    predictor = setup("predict", 
                    detection_base_weight=f"{WEIGHTS_DIR}/{wp}/{str(k).rjust(6, '0')}_best.pt" ,
                    **setup_paras)
    predictor.train_dataset.vocformat.spliter_group.split_mode = "posture"
    predictor.val_dataset.vocformat.spliter_group.split_mode = "posture"

    predictor.save_imtermediate = True
    predictor.save_raw_input = False
    predictor.save_raw_output = True
    predictor.save_processed_output = False
    predictor.postprocesser._use_bbox_area_assumption = True
    predictor._use_depth = False
    predictor.postprocesser.depth_scale = 1.0   

    return predictor

def calc_median():
    for k, v in weights.items():
        predictor = _setup(k, v)
        predictor.predict_val()

def get_raw_output_generator(directory):
    assert os.path.exists(directory), f"directory {directory} not exist"
    for f in os.listdir(directory):
        if f.endswith(".pkl"):
            with open(os.path.join(directory, f), "rb") as f:
                yield pickle.load(f)
        
def ablation_vote_threshold(mode = 'v'):
    for k, v in weights.items():
        predictor = _setup(k, v)

        raw_output_dir = f"{LOGS_DIR}/OLDTPredictor_logs/raw_output_{str(k).rjust(2, '0')}/intermediate_output/list_LandmarkDetectionResult"

        if mode == 'v':
            for threshold in np.linspace(0.0, 0.5, 51, endpoint=True):
                predictor.postprocesser.voting_threshold = threshold

                raw_output_generator = get_raw_output_generator(raw_output_dir)
                predictor.clear()
                predictor.predict_from_dataset(predictor.val_dataset, ex_raw_output=raw_output_generator, result_suffix="_vote_{:f<1.3}".format(threshold))
        elif mode == 'e':
            raw_output_generator = get_raw_output_generator(raw_output_dir)
            predictor.clear()
            predictor.postprocess_mode = 'e'
            predictor.predict_from_dataset(predictor.val_dataset, ex_raw_output=raw_output_generator, result_suffix="_exclu")

def ablation_depth():
    for k, v in weights.items():
        if k == 2 or k == 6:
            continue
        predictor = _setup(k, v)

        raw_output_dir = f"{LOGS_DIR}/OLDTPredictor_logs/raw_output_{str(k).rjust(2, '0')}/intermediate_output/list_LandmarkDetectionResult"

        raw_output_generator = get_raw_output_generator(raw_output_dir)
        predictor.clear()
        predictor._use_depth = True
        predictor.postprocesser.depth_scale = 1.0
        predictor.predict_from_dataset(predictor.val_dataset, ex_raw_output=raw_output_generator)

def _parse_vote_threshold_result(directory):
    assert os.path.exists(directory), f"directory {directory} not exist"
    # 获取文件列表
    files = os.listdir(directory)
    # 过滤得到所有以process record_vote_{}.txt形式的文件
    files = [f for f in files if f.startswith("process record_vote_") and f.endswith(".txt")]

    total_nums:list[int] = []
    reproj_nums:list[float] = []
    add_nums:list[float] = []
    class_id = -1
    vote_thresholds:list[float] = []

    for f in files:
        # 解析文件名中包含的浮点数
        vote_threshold = float(f.removeprefix("process record_vote_").removesuffix(".txt"))
        vote_thresholds.append(vote_threshold)

        # 打开并读取
        with open(os.path.join(directory, f), "r") as f:
            lines = f.readlines()
        # 分别找到第一个以"total number"\"reproj"\"ADD(s)"开头的行
        total_number_line = ""
        reproj_line = ""
        add_line = ""
        for line in lines:
            if line.startswith("total number"):
                total_number_line = line 
                break
        
        for line in lines:
            if line.startswith("reproj"):
                reproj_line = line 
                break
        for line in lines:
            if line.startswith("ADD(s)"):
                add_line = line 
                break
        # 从这三行中提取数据
        total_number_line = total_number_line.removeprefix("total number")
        reproj_line = reproj_line.removeprefix("reproj")
        add_line = add_line.removeprefix("ADD(s)")

        # 将以空格划分的多个数字转换为列表
        total_number = list(map(int, total_number_line.split()))
        reproj = list(map(int, reproj_line.split()))
        add = list(map(int, add_line.split()))

        # 找到第一个不为0的数的index
        if class_id == -1:
            class_id = total_number.index(sum(total_number))
        
        total_nums.append(total_number[class_id])
        reproj_nums.append(reproj[class_id])
        add_nums.append(add[class_id])
    
    # 按vote_thresholds排序
    vote_thresholds, total_nums, reproj_nums, add_nums = zip(*sorted(zip(vote_thresholds, total_nums, reproj_nums, add_nums)))
    # 用pandas表格输出4行数据
    df = pd.DataFrame({
        "vote_threshold": vote_thresholds,
        "total_number": total_nums,
        "reproj": reproj_nums,
        "ADD(s)": add_nums
    })
    # 打印
    print("class id:", class_id)
    print(df)

    return class_id, vote_thresholds, total_nums, reproj_nums, add_nums
    

if __name__ == "__main__":
    ablation_depth()
