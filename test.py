import numpy as np
# import sys
from __init__ import DATASETS_DIR, CFG_DIR
from posture_6d.dataset_format import VocFormat, LinemodFormat, _LinemodFormat_sub1, Elements
# from posture_6d.mesh_manager import MeshManager
# from post_processer.pnpsolver import create_model_manager
from tqdm import tqdm
import matplotlib.pyplot as plt

# from launcher.OLDTDataset import OLDTDataset, collate_fn
# from torch.utils.data import DataLoader
import time
import cv2
import pickle
from scipy.sparse import coo_array
import scipy.sparse as sp

# mm = MeshManager("E:\shared\code\OLDT\datasets\linemod\models", {
#     0 : "obj_000001.ply",
#     1 : "obj_000002.ply",
#     2 : "obj_000003.ply",
#     3 : "obj_000004.ply",
#     4 : "obj_000005.ply",
#     5 : "obj_000006.ply",
#     6 : "obj_000007.ply",
#     7 : "obj_000008.ply",
#     8 : "obj_000009.ply",
#     9 : "obj_000010.ply",
#     10: "obj_000011.ply",
#     11: "obj_000012.ply",
#     12: "obj_000013.ply",
#     13: "obj_000014.ply",
#     14: "obj_000015.ply",
# }, load_all= True)
# meshmetas = mm.get_meta_dict()

# data_num = 0


lm_vf = VocFormat(f"{DATASETS_DIR}/morrison")
# mask = cv2.imread(r".png", cv2.IMREAD_GRAYSCALE).astype(np.uint8)
# mask = np.load(".npy")
# mask = np.transpose(mask, (2, 0, 1))
print("start")
start = time.time()
lm_vf.cache_elements.open()
# lm_vf.cache_elements.set_read_only(False)
for i, viewmeta in tqdm(enumerate(lm_vf.read_from_disk())):
    pass
    # lm_vf.cache_elements.write(i, viewmeta)
# for i in tqdm(range(1000)):
#     viewmeta = lm_vf.cache_elements.read(i)
#     if i > 1000:
#         break
# for i in range(1000):
#     # a = cv2.imread(r"E:\shared\code\OLDT\datasets\morrison\images\train\000001.jpg")
#     np.load("rgb.npy")
#     # a = cv2.imread(r"E:\shared\code\OLDT\datasets\morrison\depths\train\000001.png")
#     np.load("depth.npy")

stop = time.time()
print(stop - start)
viewmeta = lm_vf.read_one(6413)
viewmeta.plot()

