import numpy as np
# import sys
from __init__ import DATASETS_DIR, CFG_DIR
from posture_6d.dataset_format import VocFormat, LinemodFormat, _LinemodFormat_sub1, Elements, JsonDict
from posture_6d.viewmeta import ViewMeta
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
from post_processer.pnpsolver import PnPSolver
from post_processer.PostProcesser import PostProcesser

for set_i in range(15):
    lm_vf = VocFormat(f"{DATASETS_DIR}/linemod/{str(set_i).rjust(6, '0')}")

    for i in range(lm_vf.data_num):
        paths:dict[str, str] = lm_vf.get_element_paths_of_one(i)
        labels_array = np.loadtxt(paths["labels"]).reshape(-1, 5)
        ids = labels_array[:, 0]

        bbox_3ds_array = np.loadtxt(paths["bbox_3ds"]).reshape(-1, 8 * 2)
        bbox_3ds_array = np.concatenate([ids.reshape(-1, 1), bbox_3ds_array], axis=1)
        np.savetxt(paths["bbox_3ds"], bbox_3ds_array, fmt = "%.8f", delimiter='\t')

        landmarks_array = np.loadtxt(paths["landmarks"]).reshape(-1, 24 * 2)
        landmarks_array = np.concatenate([ids.reshape(-1, 1), landmarks_array], axis=1)
        np.savetxt(paths["landmarks"], landmarks_array, fmt = "%.8f", delimiter='\t')

        trans_vecs_array = np.loadtxt(paths["trans_vecs"]).reshape(-1, 2 * 3)
        trans_vecs_array = np.concatenate([ids.reshape(-1, 1), trans_vecs_array], axis=1)
        np.savetxt(paths["trans_vecs"], trans_vecs_array, fmt = "%.8f", delimiter='\t')

        visib_fracts_array = np.loadtxt(paths["visib_fracts"]).reshape(-1, 1)
        visib_fracts_array = np.concatenate([ids.reshape(-1, 1), visib_fracts_array], axis=1)
        np.savetxt(paths["visib_fracts"], visib_fracts_array, fmt = "%.8f", delimiter='\t')

lm_vf_test = VocFormat(f"{DATASETS_DIR}/linemod_test")
# lm_vf.close_all()
# lm_vf.images_elements.open()
# lm_vf.labels_elements.open()
# lm_vf.landmarks_elements.open()
# lm_vf.extr_vecs_elements.open()

# with lm_vf_test.start_to_write():
#     for i, viewmeta in tqdm(enumerate(lm_vf.read_from_disk())):
#         if i > 10:
#             break
#         v:ViewMeta = viewmeta
#         lm_vf_test.write_to_disk(v)

for i, viewmeta in tqdm(enumerate(lm_vf_test.read_from_disk())):
    viewmeta.plot()
    plt.show()