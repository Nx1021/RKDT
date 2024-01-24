from __init__ import DATASETS_DIR, CFG_DIR, SERVER_DATASET_DIR
from posture_6d.data.dataset_example import VocFormat_6dPosture, UnifiedFilesHandle, ViewMeta
from posture_6d.data.dataset import SpliterGroup, Spliter
from posture_6d.core.utils import deserialize_object, serialize_object
import os
import matplotlib.pyplot as plt
import numpy as np

# morrison = VocFormat_6dPosture(os.path.join(DATASETS_DIR, r"morrison_real_voc"))

# for x in morrison:
#     x:ViewMeta
#     x.landmarks = None
#     x.plot()
#     plt.show()

import numpy as np
from scipy import stats
from scipy.stats import t
# 给定的数据
data = [1.0, 0.9, 1.0, 1.0, 1.0, 0.9, 1.0, 0.8]

# data = [2, 4, 6, 8, 10, 12, 14, 16]

# 计算均值
mean_value = np.mean(data)

# 使用95%的置信度来计算置信区间
confidence_level = 0.95
degrees_freedom = len(data) - 1
t_critical = t.ppf((1 + confidence_level) / 2, degrees_freedom)

# 计算标准误差
std_error = np.std(data, ddof=1) / np.sqrt(len(data))

# 计算置信区间的下限和上限
lower_bound = mean_value - t_critical * std_error
upper_bound = mean_value + t_critical * std_error

print(f"均值: {mean_value}")
print(f"95%置信区间: ({t_critical * std_error})")

# linemod_mix_path = os.path.join(DATASETS_DIR, r"linemod_mix\000000")

# lm = VocFormat_6dPosture(linemod_mix_path)



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
