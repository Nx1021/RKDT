from __init__ import DATASETS_DIR, CFG_DIR, SERVER_DATASET_DIR
from posture_6d.data.dataset_example import VocFormat_6dPosture, UnifiedFilesHandle
from posture_6d.data.dataset import SpliterGroup, Spliter
from posture_6d.core.utils import deserialize_object, serialize_object
import os

linemod_mix_path = os.path.join(DATASETS_DIR, "linemod_test")

lm = VocFormat_6dPosture(linemod_mix_path)

lm.spliter_group.get_cluster("posture").set_one_subset("train", {0: True, 1:True})
