from launcher.Predictor import OLDTPredictor, IntermediateManager

from models.OLDT import OLDT
from launcher.OLDT_Dataset import OLDT_Dataset

import platform
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from PIL import Image
from typing import Iterable


if __name__ == '__main__':
    sys = platform.system()
    print("system:", sys)

    data_folder = './datasets/morrison'
    yolo_weight_path = "weights/best.pt"
    cfg = "cfg/train_yolo.yaml"
    ###
    train_dataset = OLDT_Dataset(data_folder, "train")
    val_dataset = OLDT_Dataset(data_folder, "val")
    model = OLDT(yolo_weight_path, cfg, [0])  # 替换为你自己的模型
    model.load_branch_weights(0, "./weights/20230614142113branch00.pt")

    if sys == "Windows":
        batch_size = 4
        # model = torch.nn.DataParallel(model)
    elif sys == "Linux":
        batch_size = 64 # * torch.cuda.device_count()
        # model = torch.nn.DataParallel(model)
    # intermediate_manager = IntermediateManager("./intermediate_output", "voting")
    # predctor = OLDTPredictor(model, cfg, batch_size, if_postprocess=True, if_calc_error=True, intermediate_manager = intermediate_manager)
    # predctor.postprocess_mode = 'e'
    # # predctor.predict_from_dataset(val_dataset)
    # predctor.postprocess_from_intermediate(plot_outlier=True)

    intermediate_manager = IntermediateManager("./intermediate_output")
    predctor = OLDTPredictor(model, cfg, batch_size, if_postprocess=True, if_calc_error=True, intermediate_manager = None)
    predctor.postprocess_mode = 'v'
    predctor.predict_from_dataset(val_dataset)