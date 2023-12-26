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



if __name__ == "__main__":
    # sa = SphereAngle(5000)
    # ax = plt.axes(projection='3d')  # 设置三维轴
    # ax.scatter(sa.vecs[:,0], sa.vecs[:,1], sa.vecs[:,2], s=5, marker="o", c='r')
    # plt.show()
    # hough = Triangle_Hough(1, 30, 0, 2)
    # image = cv2.imread("hough_test.png", cv2.IMREAD_GRAYSCALE)
    # hough.run(image, ifplot=True)

    gripper = MyThreeFingerGripper() 
    for i in range(0, 1):
        calc = CandiCoordCalculator(create_ObjectPcd_from_file(i), gripper)
        calc.calc_candidate_coord(False, False)
        # calc._voxelize_test()

    # gripper = MyThreeFingerGripper() 
    # for i in range(0, 9):
    #     calc = CandiCoordCalculator(create_ObjectPcd_from_file(i), gripper)
    #     # calc.calc_candidate_coord(False, True)
    #     calc._voxelize_test()