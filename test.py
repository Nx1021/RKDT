import numpy as np
import sys
from __init__ import DATASETS_DIR, CFG_DIR
from posture_6d.dataset_format import VocFormat, LinemodFormat, _LinemodFormat_sub1, Elements
from posture_6d.mesh_manager import MeshManager
from post_processer.pnpsolver import create_model_manager
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
# from launcher.OLDTDataset import OLDTDataset, collate_fn
# from torch.utils.data import DataLoader
import cv2
import pickle

mm = MeshManager("E:\shared\code\OLDT\datasets\linemod\models", {
    0 : "obj_000001.ply",
    1 : "obj_000002.ply",
    2 : "obj_000003.ply",
    3 : "obj_000004.ply",
    4 : "obj_000005.ply",
    5 : "obj_000006.ply",
    6 : "obj_000007.ply",
    7 : "obj_000008.ply",
    8 : "obj_000009.ply",
    9 : "obj_000010.ply",
    10: "obj_000011.ply",
    11: "obj_000012.ply",
    12: "obj_000013.ply",
    13: "obj_000014.ply",
    14: "obj_000015.ply",
}, load_all= True)
meshmetas = mm.get_meta_dict()

data_num = 0


lm_vf = VocFormat(f"{DATASETS_DIR}/linemod/{str(0).rjust(6, '0')}")
for viewmeta in tqdm(lm_vf.read_from_disk()):
    viewmeta.show()
    plt.show()

for class_id in range(0, 15):
    lm_lf = LinemodFormat(f"{DATASETS_DIR}/linemod_orig/{str(class_id+1).rjust(6, '0')}")    
    lm_vf = VocFormat(f"{DATASETS_DIR}/linemod/{str(class_id).rjust(6, '0')}", lm_lf.data_num)
    with lm_vf.start_to_write(True):   
        for viewmeta in tqdm(lm_lf.read_from_disk()):
            viewmeta.modify_class_id([(class_id+1, class_id)])
            viewmeta.visib_fract = {class_id: 1.0}
            viewmeta.calc_by_base(meshmetas)
            lm_vf.write_to_disk(viewmeta)
# vf2.masks_elements.close()

npme = Elements(vf2, "masks_np", np.load, None, '.npy')

# for data_i, masks in tqdm(npme):
#     sub_set = vf2.decide_set(data_i)
#     m_ = npme.read(data_i)[0]
#     masks = {1: masks[0]}
#     sub_set = vf2.decide_set(data_i)
#     vf2.masks_elements.write(data_i, masks, sub_set)
#     vf2.masks_elements.read(data_i, sub_set)

for data_i, viewmeta in tqdm(enumerate(vf2.read_from_disk())):
    # vf2.masks_elements.read(data_i)
    # sub_set = vf2.decide_set(data_i)
    # viewmeta.masks = {1: npme.read(data_i, sub_set)[0]}
    print(data_i)
    # dict_ = viewmeta.serialize()
    # sub_set = vf2.decide_set(data_i)
    # pme.write(data_i, dict_["masks"], appdir=sub_set)
    

for data_i, viewmeta in tqdm(enumerate(vf2.read_from_disk())):
    vf2.serialized_element.write(data_i, viewmeta)
    # viewmeta.calc_by_base(meshmetas, True)
    # exchange_xy(viewmeta.landmarks)
    # exchange_xy(viewmeta.bbox_3d)
    # viewmeta.show()
    
    # vf2.write_to_disk(viewmeta)

# from .grasp_coord import Gripper