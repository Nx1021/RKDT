from MyLib.posture import Posture
from post_processer.model_manager import ModelInfo

import numpy as np
import open3d as o3d
import scipy.ndimage as image
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
    # obj = create_ObjectPcd_from_file(0)
    # gripper = MyThreeFingerGripper()
    # obj.draw_all(gripper)
    # rvecs, tvecs, us, _, _ = obj.parse_candidate_coord()
    # for i in range(10):
    #     obj.plot(gripper=gripper, gripper_posture_O=Posture(rvec=rvecs[i], tvec=tvecs[i]), u = us[i])


    scene = Scene()

    for i in range(3):
        obj = create_ObjectPcd_from_file(i)
        scene.add_object(obj, Posture(rvec=np.array([0., 0,0]), tvec=np.array([400.0 + np.random.randint(-100, 100), np.random.randint(-100, 100), 0])))
    start = time.time()
    for obj in scene.object_list:
        success, gp, u = scene.calc_grasp_posture(obj)
        if success:
            scene.gripper.set_u(u)
            scene.gripper.posture_WCS = gp
            scene.show_scene()
        else:
            print("find no feasible grasp coordinate")
        # for obj in scene.object_list:
        #     success, gp, u = scene.calc_grasp_posture(obj)
        #     if success:
        #         scene.gripper.set_u(u)
        #         scene.gripper.posture_inR = gp
        #         # scene.show_scene()
        #     else:
        #         print("find no feasible grasp coordinate")
    stop = time.time()
    print(stop - start)

    # gripper = Gripper()

    # object = create_ObjectPcd_from_file(0)
    # start = time.time()
    # voxelized = Voxelized.from_mesh(object.mesh, 3)
    # print(time.time() - start)

    gripper = MyThreeFingerGripper() 

    # test_image = cv2.imread("hough_test.png", cv2.IMREAD_GRAYSCALE)
    # hough = Triangle_Hough(1, 40, 10, 2)
    # hough.run(test_image, True)

    # start = time.time()
    # hough._loop_method(test_image)
    # print(time.time() - start)
    # calc = CandiCoordCalculator(create_ObjectPcd_from_file(0), gripper)
    for i in range(1):
        calc = CandiCoordCalculator(create_ObjectPcd_from_file(i), gripper)
        calc.calc_candidate_coord()