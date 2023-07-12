import OLDT_setup

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
from grasp_coord.calculator import Voxelized
from grasp_coord.calculator import CandiCoordCalculator, Triangle_Hough



if __name__ == "__main__":
    gripper = MyThreeFingerGripper() 
    for i in range(0, 9):
        calc = CandiCoordCalculator(create_ObjectPcd_from_file(i), gripper)
        calc.calc_candidate_coord(False, True)