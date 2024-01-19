import __init__

import numpy as np
import open3d as o3d
import scipy.ndimage as ndimage
import os
import cv2
import time
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from sko.GA import GA
import itertools

from grasp_coord.scene import Scene
from grasp_coord.object_pcd import ObjectPcd, create_ObjectPcd_from_file
from grasp_coord.gripper import Gripper, MyThreeFingerGripper
from grasp_coord.calculator import Voxelized, SphereAngle
from grasp_coord.calculator import CandiCoordCalculator, Triangle_Hough
from grasp_coord import MODELS_DIR, SCRIPT_DIR



if __name__ == "__main__":
    # sa = SphereAngle(5000)
    # ax = plt.axes(projection='3d')  # 设置三维轴
    # ax.scatter(sa.vecs[:,0], sa.vecs[:,1], sa.vecs[:,2], s=5, marker="o", c='r')
    # plt.show()
    # hough = Triangle_Hough(1, 30, 0, 2)
    # image = cv2.imread("hough_test.png", cv2.IMREAD_GRAYSCALE)
    # hough.run(image, ifplot=True)

    # create_ObjectPcd_from_file(0)
    # create_ObjectPcd_from_file(0)

    # log_file = os.path.join(SCRIPT_DIR, "calc_candi_grasp_log.txt")

    gripper = MyThreeFingerGripper() 
    # for vs, rn, fa in zip([3,2,3,3,3], [500, 500, 2000, 500, 500], [np.pi/6, np.pi/6, np.pi/6, np.pi/4, np.pi/18]):
    #     for i in range(8):
    #         start_time = time.time()

    #         calc = CandiCoordCalculator(create_ObjectPcd_from_file(i), gripper)
    #         calc.voxel_size = voxel_size = vs
    #         calc.rot_num = rot_num = rn
    #         calc.set_friction_angle(fa)
    #         calc.save_name = "_candi_grasp_posture" + "_vs_" + str(voxel_size) + "_rn_" + str(rot_num) + "_fa_" + str(int(round((calc.friction_angle * 180 / np.pi))))
    #         global_grasp_poses = calc.calc_candidate_coord(False, False)

    #         end_time = time.time()

    #         with open(log_file, "a") as f:
    #             f.write(calc.modelinfo.name + calc.save_name + " ")
    #             f.write(str(round(end_time - start_time, 1)) + " ")
    #             f.write(str(len(global_grasp_poses)) + "\n")
        # calc._voxelize_test()

    calc = CandiCoordCalculator(create_ObjectPcd_from_file(0), gripper)
    # ### 保存结果
    save_path = os.path.join(MODELS_DIR, "bar_clamp_candi_grasp_posture_vs_3_rn_500_fa_30" + ".npy")
    global_grasp_poses = np.load(save_path, allow_pickle= True)
    
    calc.modelinfo.candi_coord_parameter = global_grasp_poses

    ### 显示结果（可选的）
    # calc.modelinfo.draw_all_in_one(gripper)
    # calc.modelinfo.draw_all(gripper)
    calc.modelinfo.mesh.compute_triangle_normals()
    o3d.visualization.draw_geometries([calc.modelinfo.mesh], width=1280, height=720)

    # gripper = MyThreeFingerGripper() 
    # for i in range(0, 9):
    #     calc = CandiCoordCalculator(create_ObjectPcd_from_file(i), gripper)
    #     # calc.calc_candidate_coord(False, True)
    #     calc._voxelize_test()