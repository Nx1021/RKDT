import numpy as np
import sys
from __init__ import DATASETS_DIR, CFG_DIR
from posture_6d.data.dataset_format import VocFormat, LinemodFormat, _LinemodFormat_sub1, Elements
from posture_6d.data.mesh_manager import MeshManager
from posture_6d.utils import JsonIO
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
# from launcher.OLDTDataset import OLDTDataset, collate_fn
# from torch.utils.data import DataLoader
import cv2
import pickle


# mm = MeshManager("E:\shared\code\OLDT\datasets\morrison_models", {
#     0 : "bar_clamp.ply",
#     1 : "gearbox.ply",
#     2 : "nozzle.ply",
#     3 : "part1.ply",
#     4 : "part3.ply",
#     5 : "pawn.ply",
#     6 : "turbine_housing.ply",
#     7 : "vase.ply",
#     8 : "ape.ply"
# }, load_all= True)
# meshmetas = mm.get_meta_dict()

lm_lf000019 = _LinemodFormat_sub1(r"E:\shared\code\ObjectDatasetTools\LINEMOD\000019\aug_output")
vf = VocFormat(r"E:\shared\code\OLDT\datasets\morrison3", clear = False)
with vf.start_to_write():
    for viewmeta in tqdm(lm_lf000019.read_from_disk()):
        viewmeta.filter_unvisible()
        viewmeta.modify_class_id([(20, 0),
                                  (21, 1),
                                  (22, 2),
                                  (23, 3),
                                  (24, 4),
                                  (25, 5),
                                  (26, 6),
                                  (27, 7),
                                  (28, 8),])
        vf.write_to_disk(viewmeta)

data_num = 0


lm_vf = VocFormat(f"{DATASETS_DIR}/morrison")
viewmeta = lm_vf.read_one(6413)
viewmeta.plot()


lm_vf.close_all(False)
lm_vf.serialized_element.close()
for viewmeta in tqdm(lm_vf.read_from_disk()):
    pass

lm_vf.set_all_read_only()
lm_vf.serialized_element.open()
lm_vf.serialized_element.set_read_only(False)
with lm_vf.start_to_write():
    for data_i, viewmeta in enumerate(tqdm(lm_vf.read_from_disk())):
        lm_vf.serialized_element.write(data_i, viewmeta)
    # new_labels.write(i, viewmeta)
    # if i % 100 == 0:
    #     viewmeta.plot()
    #     plt.show()

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
    # viewmeta.plot()
    
    # vf2.write_to_disk(viewmeta)

# from .grasp_coord import Gripper