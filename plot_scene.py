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

s = Scene()

s.load_log(r"E:\shared\code\OLDT\logs\grasping_running\log_20240123182051\scene_log_20240123182051.json")
s.show_scene_in_log(261, save_path = r"E:\shared\code\OLDT\logs\grasping_running\log_20240123182051")

