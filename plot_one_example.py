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

from posture_6d.data.dataset_example import VocFormat_6dPosture, Spliter, ViewMeta
from posture_6d.data.mesh_manager import MeshManager
from matplotlib import pyplot as plt
import open3d as o3d

if __name__ == "__main__":

    # morrison_real_voc = VocFormat_6dPosture(r"E:\shared\code\OLDT\datasets\morrison_real_voc", lazy=True)

    # vm:ViewMeta = morrison_real_voc[2]

    # vm.masks = None
    # vm.color = vm.color[:, :, ::-1]

    # bbox_3d = vm.bbox_3d
    # vm.bbox_3d = None
    # vm.plot()
    # plt.show()

    # label = vm.labels
    # vm.labels = None
    # vm.bbox_3d = bbox_3d
    # vm.landmarks = None
    # vm.plot()
    # plt.show()

    mm = MeshManager(r"E:\shared\code\OLDT\datasets\morrison_models")
    obj_0 = mm.export_meta(0)

    kps_mesh = o3d.geometry.TriangleMesh()
    for ldmk in obj_0.ldmk_3d:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=4)
        sphere.translate(ldmk)
        sphere.paint_uniform_color([0.0, 1.0, 0.0])
        kps_mesh += sphere
    
    o3d.io.write_triangle_mesh("kps_mesh.ply", kps_mesh)

