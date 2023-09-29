# 在还没有指定TypeVar类型，其类型为bound指定的类型时： Generic、参数正确类型输入、属性注释缺一不可
# 在子类指定了Generic的类型后，其类型被绑定，子类的子类无法再修改已经绑定的TypeVar
# 如果子类未绑定，则可以在子类的子类再指定
from __init__ import DATASETS_DIR, CFG_DIR
from posture_6d.data.dataset_format import VocFormat, Mix_VocFormat, Spliter

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
from posture_6d.data.dataset_format import deserialize_object
from posture_6d.data.viewmeta import ViewMeta
import numpy as np
import cv2
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from pympler import asizeof
from scipy.linalg import logm
import warnings
from typing import TypeVar

from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import os
from scipy.optimize import fsolve

from typing import Any, Union, Callable, TypeVar, Generic, Iterable, Type
import glob

# top_dir = r"E:\shared\code\OLDT\datasets\morrison_mix_single\000000"
# dirs = ["labels", "masks", "visib_fracts"]
# for d in dirs:
#     _d = os.path.join(top_dir, d)
#     dict_ = deserialize_object(os.path.join(_d, "data_info_map.elmm"))
#     paths = glob.glob(os.path.join(_d, "**/*"))
#     names = [int(os.path.splitext(os.path.basename(p))[0]) for p in paths]
#     sub_dirs = [os.path.split(os.path.dirname(p))[-1] for p in paths]

#     for i, n in enumerate(names):
#         if sub_dirs[i] != dict_[n]["dir"]:
#             p = paths[i]
#             os.remove(p)

obj_i = 0
vm = Mix_VocFormat(f"{DATASETS_DIR}/morrison_mix_single/000000")
vm.labels_elements.default_image_size = (640, 480)
#### 1
# vm.labels_elements.default_image_size = (640, 480)
# with vm.writer.allow_overwriting():
#     for data_i, v in vm.items():
#         # bbox = v.labels
#         masks = v.masks
#         color = v.color
#         # # 排除黑色区域，重新计算mask
#         # for i, mask in masks.items():
#         #     visib = v.visib_fract[i]
#         #     if np.sum(mask) == 0 or visib == 0:
#         #         continue
#         #     bin_color = np.zeros(color.shape[:2], np.uint8)
#         #     gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
#         #     bin_color[gray > 30] = 1
#         #     valid_mask = bin_color * mask

#         #     orig_pixel_num = np.sum(mask) / visib
#         #     new_visib = np.sum(valid_mask) / orig_pixel_num

#         #     v.masks[i] = valid_mask
#         #     v.visib_fract[i] = new_visib

#         v.labels = v.calc_bbox2d_from_mask(v.masks)
        
#         subdir = vm.images_elements.data_info_map[data_i].dir
#         # vm.masks_elements.write(data_i, v.masks, subdir=subdir)
#         # vm.visib_fract_elements.write(data_i, v.visib_fract, subdir=subdir)
#         vm.labels_elements.write(data_i, v.labels, subdir=subdir)


### 2
# to_remove = []
# for data_i, visib in vm.visib_fract_elements.items():
#     if obj_i not in visib:
#         to_remove.append(data_i)
#     elif visib[obj_i] < 0.7:
#         to_remove.append(data_i)

# with vm.writer.allow_overwriting():
#     for data_i in to_remove:
#         vm.remove_one(data_i)

# for c in vm.elements_map.values():
#     # c._data_info_map.rebuild(True, False)
#     c.save_data_info_map()

# for i in range(len(vm.data_overview_table.row_names)):
#     if i not in vm.masks_elements.data_info_map:
#         vm.data_overview_table.remove_row(i)
# vm.save_overview()

### 3
# with vm.writer.allow_overwriting():
#     vm.make_continuous()
# vm.labels_elements.continuous

vm.save_elements_data_info_map()

### 4
# for c in vm.elements_map.values():
#     c._data_info_map.clear()
#     c._data_info_map.rebuild(True, False)
# for v in vm[::500]:
#     v.plot()
#     plt.show()

# print(1)