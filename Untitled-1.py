# %%
from __init__ import DATASETS_DIR, CFG_DIR, SERVER_DATASET_DIR
from posture_6d.data.dataset_example import VocFormat_6dPosture, UnifiedFilesHandle
from posture_6d.data.dataset import SpliterGroup, Spliter
from posture_6d.core.utils import deserialize_object, serialize_object
import os
import numpy as np


linemod_mix_path = os.path.join(DATASETS_DIR, "linemod_mix")


# %%
for i in range(15):
    lm = VocFormat_6dPosture(os.path.join(linemod_mix_path, str(i).rjust(6, '0')))


    # %%
    print(lm.SPLIT_PARA)
    base_txt = os.path.join(linemod_mix_path, "base_log.txt")
    aug_txt = os.path.join(linemod_mix_path, "isolate_log.txt")
    posture_train_txt = os.path.join(linemod_mix_path, "oldt_train.txt")
    posture_val_txt   = os.path.join(linemod_mix_path, "oldt_val.txt")
    train_txt = os.path.join(linemod_mix_path, "train.txt")
    val_txt   = os.path.join(linemod_mix_path, "val.txt")

    base_array = np.loadtxt(base_txt, dtype=str)
    aug_array = np.loadtxt(aug_txt, dtype=str)
    posture_train_array = np.loadtxt(posture_train_txt, dtype=str)
    posture_val_array = np.loadtxt(posture_val_txt, dtype=str)
    train_array = np.loadtxt(train_txt, dtype=str)
    val_array = np.loadtxt(val_txt, dtype=str)


    # %%

    base_dict = {int(x): True for x in base_array}
    aug_dict = {int(x): True for x in aug_array}
    posture_train_dict = {int(x): True for x in posture_train_array}
    posture_val_dict = {int(x): True for x in posture_val_array}
    train_dict = {int(x): True for x in train_array}
    val_dict = {int(x): True for x in val_array}


    # %%
    lm.spliter_group.get_cluster("default").set_one_subset("train", train_dict)
    lm.spliter_group.get_cluster("default").set_one_subset("val", val_dict)


    # %%

    lm.spliter_group.get_cluster("reality").set_all_by_rate((1.0, 0.0))

    lm.spliter_group.get_cluster("basis").set_one_subset("basic", base_dict)
    lm.spliter_group.get_cluster("basis").set_one_subset("augment", aug_dict)

    lm.spliter_group.get_cluster("posture").set_one_subset("train", posture_train_dict)
    lm.spliter_group.get_cluster("posture").set_one_subset("val", posture_val_dict)



