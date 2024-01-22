
# %%
from __init__ import DATASETS_DIR, CFG_DIR
# from posture_6d.data.dataset_format import VocFormat, Mix_VocFormat, Spliter
from posture_6d.data.dataset_example import VocFormat_6dPosture, Spliter, ViewMeta, BopFormat
from posture_6d.data.spliter import SpliterGroup, FilesHandle
# from gen_mixed_linemod import MixLinemod_VocFormat
# from posture_6d.data.mesh_manager import MeshManager
# from posture_6d.data.dataset_format import ClusterIONotExecutedWarning
# from posture_6d.posture import Posture
# from posture_6d.derive import PnPSolver
# import numpy as np
# from tqdm import tqdm

# from gen_mixed_linemod import MixLinemod_VocFormat
from posture_6d.data.mesh_manager import MeshManager
# from posture_6d.data.viewmeta import ViewMeta
from posture_6d.core.utils import Table
from posture_6d.core.posture import Posture
from posture_6d.core import CameraIntr
from posture_6d.derive import calc_bbox_2d_proj, calc_Z_rot_angle_to_horizontal
import numpy as np
import cv2
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt
import torch
import warnings
from typing import TypeVar

from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import os

from typing import Any, Union, Callable, TypeVar, Generic, Iterable, Type
from functools import partial

import posture_6d.data.IOAbstract as IOAbstract
IOAbstract.IO_TEST_MODE = False
IOAbstract.IO_TEST_PRINT_INTERNAL = 1

# %%
new_mix = VocFormat_6dPosture(f"{DATASETS_DIR}/linemod_mix_new")
# server = VocFormat_6dPosture("/home/nerc-ningxiao/datasets/linemod_mix_new")
force = True
new_mix.extr_vecs_elements.file_to_cache(force = force)
new_mix.intr_elements.file_to_cache(force = force)
new_mix.depth_scale_elements.file_to_cache(force = force)
new_mix.bbox_3ds_elements.file_to_cache(force = force)
new_mix.landmarks_elements.file_to_cache(force = force)
new_mix.visib_fracts_element.file_to_cache(force = force)

# server.copy_from_simplified(new_mix, cover=True, force=True)

# print(new_mix.spliter_group.get_cluster("default"))
# for i in range(15):
#     new_mix.spliter_group.add_spliter("obj_{}_posture".format(str(i).rjust(2, "0")), subsets=["train", "val"])

#     spliter = new_mix.spliter_group.get_cluster("obj_{}_posture".format(str(i).rjust(2, "0")))
#     spliter.exclusive = False

#     print(spliter)


# # for obj in new_mix.clusters:
# #     obj.delete_unlink_files_in_dir()


# # new_mix.build_partly(list(range(57819, 58782)))

# # with new_mix.get_writer().allow_overwriting():
# #     for i in tqdm(range(57819, 58819)):
# #         new_mix.remove(i)

# # for spliter in new_mix.spliter_group.clusters:
# #     with spliter.get_writer().allow_overwriting():
# #         for i in range(57819, 58819):
# #             spliter.remove_elem_idx(i)
# # %%
# # base_num = 0
# # for i in range(15):
# #     org_mix_splitergroup = SpliterGroup(f"{DATASETS_DIR}/linemod_mix/{str(i).rjust(6, '0')}/split_group", "")
# #     org_mix_splitergroup.add_spliter("posture", ["train", "val"])
# #     org_mix_splitergroup.add_spliter("aug_posture", ["train", "val"])
# #     org_mix_splitergroup.add_spliter("reality", ["real", "sim"])
# #     org_mix_spliter:Spliter = org_mix_splitergroup.get_cluster("posture")
# #     num = len(org_mix_splitergroup.get_cluster("reality").get_idx_list("real"))

# #     reality_spliter = org_mix_splitergroup.get_cluster("reality")
# #     real_list = list(reality_spliter.get_idx_list("real"))

# #     aug_posture_spliter = org_mix_splitergroup.get_cluster("aug_posture")
# #     posture_train_list = list(aug_posture_spliter.get_idx_list("train"))

# #     posture_spliter = org_mix_splitergroup.get_cluster("posture")
# #     posture_val_list = list(posture_spliter.get_idx_list("val"))

# #     new_spliter = new_mix.spliter_group.get_cluster("obj_{}_posture".format(str(i).rjust(2, "0")))
# #     new_spliter.exclusive = False
# #     print(new_spliter)

# #     val_idx = org_mix_spliter.get_idx_list("val")
# #     print(len(val_idx))

# #     with new_spliter.get_writer().allow_overwriting(): 
# #         new_spliter.clear()
# #         for real_idx in tqdm(real_list):
# #             if real_idx in posture_train_list:
# #                 new_spliter.set_one(real_idx + base_num, "train", True)
# #             if real_idx in posture_val_list:
# #                 new_spliter.set_one(real_idx + base_num, "val", True)
# #         new_spliter.cache_to_file()

# #     print(new_spliter)
# #     base_num += num
# #     print(base_num)

# # # %%

# mesh_manager = MeshManager(f"{DATASETS_DIR}/linemod/models")
# meta_dict = mesh_manager.get_meta_dict()

# # # %%
# # spliter_00 = new_mix.spliter_group.get_cluster("obj_{}_posture".format(str(0).rjust(2, "0")))
# # with spliter_00.get_writer().allow_overwriting():
# #     spliter_00.set_one(0, "val", True)
# # print(spliter_00.query_fileshandle(0).synced)

# # %%
# # 54819
# # for i in range(4,7):
# #     pbr_dataset = BopFormat(f"E:/文件/工作文件/数控中心/机器人抓取/公开数据集/train_pbr/{str(i).rjust(6, '0')}", "")

# #     for fh in pbr_dataset.rgb_cluster.query_all_fileshandle():
# #         fh:FilesHandle
# #         fh.read_func = lambda file_path: cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
# #     for fh in pbr_dataset.depth_cluster.query_all_fileshandle():
# #         fh:FilesHandle
# #         fh.read_func = lambda file_path: cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH)
# #     for fh in pbr_dataset.mask_cluster.query_all_fileshandle():
# #         fh:FilesHandle
# #         fh.read_func = lambda file_path: cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

# #     with new_mix.get_writer().allow_overwriting():
# #         for src_i, meta in tqdm(enumerate(pbr_dataset)):
# #             meta:ViewMeta
# #             meta.modify_class_id(list(zip(range(1, 16), range(15))))
# #             try:
# #                 meta.calc_by_base(meta_dict)
# #             except:
# #                 continue

# #             new_mix.append(meta)

# #             for id_, visib in meta.visib_fracts.items():
# #                 if visib > 0.4:
# #                     new_mix.spliter_group.get_cluster("obj_{}_posture".format(str(id_).rjust(2, "0"))).set_one(new_mix.i_upper-1, "train", True)
        
# #         # new_mix.spliter_group.get_cluster("obj_{}_posture".format(str(0).rjust(2, "0"))).cache_to_file(force = True)

# # upper 60,820
# lmo = BopFormat(r"E:\shared\code\OLDT\datasets\linemod_Occlusion")

# with new_mix.get_writer().allow_overwriting():
#     for src_i, meta in tqdm(enumerate(lmo)):
#         meta:ViewMeta
#         meta.modify_class_id(list(zip(range(1, 16), range(15))))
#         try:
#             meta.calc_by_base(meta_dict)
#         except:
#             continue

#         new_mix.append(meta)

#         for id_, visib in meta.visib_fracts.items():
#             if visib > 0.4:
#                 if src_i % 5 == 0:
#                     new_mix.spliter_group.get_cluster("obj_{}_posture".format(str(id_).rjust(2, "0"))).set_one(new_mix.i_upper-1, "train", True)
#                 new_mix.spliter_group.get_cluster("obj_{}_posture".format(str(id_).rjust(2, "0"))).set_one(new_mix.i_upper-1, "val", True)
    
#     # new_mix.spliter_group.get_cluster("obj_{}_posture".format(str(0).rjust(2, "0"))).cache_to_file(force = True)


# # for i in range(1, 15):
# #     new_mix_spliter = Spliter(new_mix.spliter_group, "obj_{}_posture".format(str(i).rjust(2, "0")), subsets=["train", "val"])
# #     new_mix_spliter.exclusive = False
# #     new_mix.spliter_group.add_cluster(new_mix_spliter)

# #     mix = VocFormat_6dPosture(f"{DATASETS_DIR}/linemod_mix/{str(i).rjust(6, '0')}")
# #     if i ==0:
# #         real_list = list(range(3708))
# #     else:
# #         reality_spliter = mix.spliter_group.get_cluster("reality")
# #         real_list = list(reality_spliter.get_idx_list("real"))

# #     aug_posture_spliter = mix.spliter_group.get_cluster("aug_posture")
# #     posture_train_list = list(aug_posture_spliter.get_idx_list("train"))

# #     posture_spliter = mix.spliter_group.get_cluster("posture")
# #     posture_val_list = list(posture_spliter.get_idx_list("val"))

# #     with new_mix.get_writer():
# #         for real_idx in tqdm(real_list):
# #             new_mix.append(mix[real_idx])

# #             if real_idx in posture_train_list:
# #                 new_mix_spliter.set_one(new_mix.i_upper-1, "train", True)
# #             if real_idx in posture_val_list:
# #                 new_mix_spliter.set_one(new_mix.i_upper-1, "val", True)


# # %%
