from __init__ import DATASETS_DIR, CFG_DIR, SERVER_DATASET_DIR
from posture_6d.data.dataset_example import VocFormat_6dPosture, UnifiedFilesHandle
from posture_6d.data.dataset import SpliterGroup, Spliter
from posture_6d.core.utils import deserialize_object, serialize_object
import os

linemod_mix_path = os.path.join(DATASETS_DIR, r"linemod_mix\000000")

lm = VocFormat_6dPosture(linemod_mix_path)



# import numpy as np
# from scipy.spatial import ConvexHull
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import time
# from scipy.optimize import root
# from grasp_coord.calculator import ForceClosureEvaluator



# start = time.time()

# # for i in range(1000):
# #     # 创建一个随机的三维点集
# #     points = np.random.rand(20**3, 3)

# #     # 计算凸包
# #     hull = ConvexHull(points)

# fce = ForceClosureEvaluator(np.pi/6)
# rlt = fce.get_v_sub_space_tol_angle(np.array([(1- 0.1961**2)**(1/2), 0.0, 0.1961]))
# print(rlt)

# stop = time.time()
# print(stop - start)