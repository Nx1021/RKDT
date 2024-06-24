from __init__ import SCRIPT_DIR, DATASETS_DIR, CFG_DIR, WEIGHTS_DIR, LOGS_DIR, _get_sub_log_dir

from models.results import LandmarkDetectionResult
from models.utils import tensor_to_numpy
from launcher.setup import setup

import platform
import numpy as np
from typing import Iterable
import os
import shutil
from utils.yaml import load_yaml, dump_yaml
import pickle
import tqdm

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def predict_and_save():
    cfg_file = f"{CFG_DIR}/oldt_linemod_mix.yaml"
    setup_paras = load_yaml(cfg_file)["setup"]

    sys = platform.system()
    if sys == "Windows":
        batch_size = 4
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

    wp = "linemod_mix_new" #"linemod_mix_new"

    for k, v in weights.items():
        if k != 0:
            continue
        setup_paras["sub_data_dir"] = "linemod_mix/{}".format(str(k).rjust(6, '0'))
        setup_paras["ldt_branches"] = {k: f"{wp}/{v}"}
        setup_paras["batch_size"] = batch_size

        predictor = setup("predict", 
                        detection_base_weight=f"{WEIGHTS_DIR}/{wp}/{str(k).rjust(6, '0')}_best.pt" ,
                        **setup_paras)
        predictor.train_dataset.vocformat.spliter_group.split_mode = "posture"
        predictor.val_dataset.vocformat.spliter_group.split_mode = "posture"
        # predictor.postprocesser._use_bbox_area_assumption = True
        predictor._use_depth = False
        predictor.postprocesser.depth_scale = 1.0

        predictor.save_imtermediate = True
        predictor.save_raw_input = False
        predictor.save_processed_output = False
        predictor.save_raw_output = True
        predictor.predict_val()

def extract(directory:str, q_numbers = [0]):
    '''
    coord: [N,2]
    idx: [N]
    '''
    results = {}

    for file in tqdm.tqdm(os.listdir(directory)):
        with open(os.path.join(directory, file), "rb") as f:
            # 加载pickle文件
            data:list[LandmarkDetectionResult] = pickle.load(f, encoding='gbk')
            for d in data:
                for q_idx in q_numbers:
                    results.setdefault(q_idx, [[], []])
                    results[q_idx][0].append(tensor_to_numpy(d.landmarks_n[0, q_idx]))
                    results[q_idx][1].append(np.argmax(tensor_to_numpy(d.landmarks_probs[0, q_idx])))
    
    for k,v in results.items():
        results[k] = [np.array(v[0]), np.array(v[1])]

    return results

def plot(coords:np.ndarray, idx:np.ndarray, hide_ticks = True, hide_title = False):
    mask_ = idx != 24

    mask_ = mask_ * (coords[:,0] < 1) * (coords[:,1] < 1) * (coords[:,0] > 0) * (coords[:,1] > 0)

    coords = coords[mask_]
    idx = idx[mask_]
    plt.scatter(coords[:, 0], coords[:, 1], c=idx, vmin=0, vmax=24, cmap="rainbow", s = 1.5)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    # plt.gca().axis('equal')
    plt.gca().set_aspect('equal', adjustable='box')

    if not hide_title:
        plt.title(f"Slot {k}")

    if hide_ticks:
        plt.xticks([])
        plt.yticks([])



if __name__ == "__main__":
    path = r"E:\shared\code\OLDT\logs\OLDTPredictor_logs\20240415134757Windows\intermediate_output\list_LandmarkDetectionResult"
    results = extract(path, [4*x for x in range(24)])
    fig = plt.figure(figsize=(16, 12))
    # 绘制子图
    for i, (k, v) in enumerate(results.items()):
        plt.subplot(4, 6, i+1)
        plot(v[0], v[1])

    # 绘制通用的colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax  = fig.add_axes([0.90, 0.10, 0.02, 0.8 ])#位置[左,下,右,上]
    norm = mcolors.Normalize(vmin=0, vmax=23)
    cmap = plt.cm.get_cmap('rainbow')
    colors = cmap([norm(x) for x in range(24)])  # 这里的0.5就是你指定的归一化值
    cb_img = np.array(colors).reshape(24, 1, 4)[:,:,:3]
    cb_img = (cb_img * 255).astype(np.uint8)
    plt.imshow(cb_img, aspect='auto')
    plt.yticks(range(24), range(24))
    plt.xticks([])

    plt.show()