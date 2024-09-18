from __init__ import SCRIPT_DIR, DATASETS_DIR, CFG_DIR, WEIGHTS_DIR, LOGS_DIR, _get_sub_log_dir
from launcher.Predictor import OLDTPredictor, IntermediateManager
from launcher.Trainer import Trainer

from models.OLDT import OLDT
from post_processer import PostProcesser, create_pnpsolver, create_mesh_manager
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
import cv2
import time
from tqdm import tqdm

class MyDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.files = os.listdir(data_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        img = cv2.imread(os.path.join(self.data_dir, file))
        return img

def collate_fn(batch):
    return batch

if __name__ == "__main__":
    bs = 1
    dataset = MyDataset(r"datasets\linemod\000006\rgb")
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=0, collate_fn = collate_fn)

    yolo_weight_path = r"weights\linemod_mix\000005_best.pt"
    cfg_file = r"cfg\oldt_linemod_mix.yaml"
    model = OLDT(yolo_weight_path,  cfg_file, [5])
    model.load_branch_weights(5, r"weights\linemod_mix\20240126020130branch_ldt_05.pt")

    pnpsolver = create_pnpsolver(cfg_file)
    mesh_manager = create_mesh_manager(cfg_file)
    out_bbox_threshold = load_yaml(cfg_file)["out_bbox_threshold"]
    postprocesser = PostProcesser(pnpsolver, mesh_manager, out_bbox_threshold)
    model([dataset[x] for x in range(bs)])

    start = time.time()
    for imgs in tqdm(dataloader):
        predictions = model(imgs)
        x = postprocesser.process(imgs, predictions)
    end = time.time()
    print(f"Time: {end - start}")
    print(len(dataset))