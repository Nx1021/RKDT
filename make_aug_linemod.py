from MyLib.posture_6d.data.dataset_example import VocFormat_6dPosture, UnifiedFilesHandle, UnifiedFileCluster, BopFormat
from MyLib.posture_6d.data.dataset import SpliterGroup, Spliter, Dataset, FilesHandle
from MyLib.posture_6d.data.viewmeta import ViewMeta, MeshMeta
from MyLib.posture_6d.data.mesh_manager import MeshManager
from MyLib.posture_6d.derive import PnPSolver, calc_equivalent_poses_of_symmetrical_objects, Posture
import numpy as np
from __init__ import SCRIPT_DIR, DATASETS_DIR
import matplotlib.pyplot as plt
import glob
import os

# bp = BopFormat("E:\shared\code\OLDT\datasets\linemod_Occlusion")
voc = VocFormat_6dPosture("E:\shared\code\OLDT\datasets\linemod_o")

# spliter = voc.spliter_group.get_cluster("posture")
# spliter.set_one_subset("val", list(range(len(voc))))

with voc.get_writer().allow_overwriting():
    for i, x in voc.items():
        x:ViewMeta
        x.modify_class_id(list(zip(range(1, 16), range(15))))
        voc.write(i, x)

# reality_nums    = {
#     1: 3641,
#     2: 3698,
#     3: 3602,
#     4: 3587,
#     5: 3536,
#     6: 3719,
#     7: 3563,
#     8: 3761,
#     9: 3758,
#     10: 3659,
#     11: 3710,
#     12: 3455,
#     13: 3680,
#     14: 3728,
#                    }
# total_nums      = {
#     1: 10206,
#     2: 10263,
#     3: 10167,
#     4: 10152,
#     5: 10101,
#     6: 10284,
#     7: 10128,
#     8: 10326,
#     9: 10323,
#     10: 10224,
#     11: 10275,
#     12: 10020,
#     13: 10245,
#     14: 10293,
#                    }

# for i in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
#     sg = SpliterGroup(r"E:\shared\code\OLDT\datasets\linemod_mix\{}\split_group".format(str(i).rjust(6, '0')), "", split_paras={
#         "posture": ["train", "val"],
#         "default": ["train", "val"],
#         "reality": ["real", "sim"],
#         "basis": ["basic", "augment"]})


#     reality_num = reality_nums[i]
#     total_num = total_nums[i]

#     default_spliter = sg.get_cluster("default")
#     reality_spliter = sg.get_cluster("reality")
#     basis_spliter = sg.get_cluster("basis")
#     posture_spliter = sg.get_cluster("posture")

#     #
#     default_spliter.clear(force=True)
#     train_paths = glob.glob(os.path.join(r"E:\shared\code\OLDT\datasets\linemod_mix\{}\images\train".format(str(i).rjust(6, '0')), "**/*"), recursive=True)
#     train_idx = [int(os.path.splitext(os.path.basename(x))[0]) for x in train_paths]
#     val_paths = glob.glob(os.path.join(r"E:\shared\code\OLDT\datasets\linemod_mix\{}\images\val".format(str(i).rjust(6, '0')), "**/*"), recursive=True)
#     val_idx = [int(os.path.splitext(os.path.basename(x))[0]) for x in val_paths]

#     default_spliter.set_one_subset("train", train_idx)
#     default_spliter.set_one_subset("val",   val_idx)
#     print(default_spliter.get_nums())

#     #
#     reality_spliter.clear(force=True)
#     reality_spliter.set_one_subset("real", list(range(reality_num+1)))
#     reality_spliter.set_one_subset("sim", list(range(reality_num+1, total_num+1)))
#     print(reality_spliter.get_nums())

#     #
#     basic_seq = list([x for x in range(reality_num+1) if x % 3 == 0])
#     basis_spliter.clear(force=True)
#     basis_spliter.set_one_subset("basic",   basic_seq)
#     basis_spliter.set_one_subset("augment", np.setdiff1d(np.arange(total_num+1), basic_seq).tolist())
#     print(basis_spliter.get_nums())

#     #
#     posture_spliter.clear(force=True)
#     posture_val = np.random.choice(basic_seq, size = int(len(basic_seq) * 0.85), replace=False).tolist()
#     posture_train = np.setdiff1d(np.arange(total_num+1), posture_val).tolist()
#     posture_spliter.set_one_subset("val", posture_val)
#     posture_spliter.set_one_subset("train", posture_train)
#     print(posture_spliter.get_nums())

#     aug_posture_spliter = Spliter(sg, "aug_posture")
#     sg.add_cluster(aug_posture_spliter)
#     aug_posture_spliter.clear(force=True)
#     aug_posture_spliter.exclusive = False

#     aug_posture_spliter.set_one_subset("val", posture_spliter.get_idx_dict()["val"])
#     train_list:list = list(posture_spliter.get_idx_list("train"))
#     val_list = posture_spliter.get_idx_list("val")

#     # 随机选70%的数据作为训练集
#     train_list_aug = np.random.choice(val_list, int(len(val_list)*0.7), replace=False).tolist()
#     # 添加到train_dict，使用字典生成式组成{num: True}的形式
#     train_list.extend(train_list_aug)

#     aug_posture_spliter.set_one_subset("train", train_list)

#     print(aug_posture_spliter.get_nums())

#     # posture_spliter = sg.get_cluster("posture")
#     # aug_posture_spliter = Spliter(sg, "aug_posture")
#     # print(aug_posture_spliter.get_nums())
#     # aug_posture_spliter.exclusive = False
#     # sg.add_cluster(aug_posture_spliter)

#     # aug_posture_spliter.set_one_subset("val", posture_spliter.get_idx_dict()["val"])
#     # train_list:list = list(posture_spliter.get_idx_list("train"))
#     # val_list = posture_spliter.get_idx_list("val")

#     # # 随机选70%的数据作为训练集
#     # train_list_aug = np.random.choice(val_list, int(len(val_list)*0.7), replace=False).tolist()
#     # # 添加到train_dict，使用字典生成式组成{num: True}的形式
#     # train_list.extend(train_list_aug)

#     # aug_posture_spliter.set_one_subset("train", train_list)

#     # print(aug_posture_spliter.get_nums())

# for i in range(15):
#     if i in [2, 5, 12, 13]:
#         continue
#     voc = VocFormat_6dPosture(os.path.join(DATASETS_DIR, r"linemod_mix\{}".format(str(i).rjust(6, '0'))))

#     voc.bbox_3ds_elements.file_to_cache(force=True)
#     voc.depth_scale_elements.file_to_cache(force=True)
#     voc.intr_elements.file_to_cache(force=True)
#     voc.landmarks_elements.file_to_cache(force=True)
#     voc.extr_vecs_elements.file_to_cache(force=True)
#     voc.visib_fracts_element.file_to_cache(force=True)