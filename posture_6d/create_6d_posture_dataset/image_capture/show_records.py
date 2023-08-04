import cv2
import sys
import png
import os
import numpy as np
import glob
import json
import matplotlib.pyplot as plt
import open3d as o3d
from mpl_toolkits import mplot3d
from excude_pipeline import *
from utils.compute_gt_poses import GtPostureComputer
from utils.camera import convert_depth_frame_to_pointcloud

if __name__ == "__main__":
    argv = sys.argv
    argv = ("", "LINEMOD\\{}".format(DATADIR)) 
    try:
        if argv[1] == "all":
            folders = glob.glob("LINEMOD/*/")
        elif argv[1] +"\\" in glob.glob("LINEMOD/*/"):
            folders = [argv[1] +"\\"]
        else:
            exit()
    except:
        exit()
    inter = 1
    for directory in folders:
        with open(directory+'intrinsics_cali.json', 'r') as f:
            camera_intrinsics = json.load(f)
        data_i = 0
        while True:
            if data_i % inter != 0 :
                pass
            else:          
                name = str(data_i).rjust(6, "0")
                # cad
                img_file = os.path.join(directory, RGB_DIR, "{}.jpg").format(name)
                cad = cv2.imread(img_file)
                if cad is None:
                    break
                cad = cv2.cvtColor(cad, cv2.COLOR_BGR2RGB)  
                # depth
                depth_file = os.path.join(directory, DEPTH_DIR, "{}.png").format(name)
                reader = png.Reader(depth_file)
                pngdata = reader.read()
                depth = np.array(tuple(map(np.uint16, pngdata[2])))
                points = convert_depth_frame_to_pointcloud(depth, camera_intrinsics)
                points = np.reshape(points, (-1, 3))
                colors = np.reshape(cad, (-1, 3))
                pcd = o3d.geometry.PointCloud()
                pcd.points = utility.Vector3dVector(points)
                pcd.colors = utility.Vector3dVector(colors/255)

                scene_aruco_3d, ids = GtPostureComputer.get_aruco_coord(cad, depth, camera_intrinsics)
                scene_aruco_3d = np.reshape(scene_aruco_3d, (-1,3))
                sphere_list = []
                for aruco_p in scene_aruco_3d:
                    sphere = o3d.geometry.TriangleMesh.create_sphere(0.005)
                    sphere.transform(Posture(tvec=aruco_p).trans_mat)
                    sphere.paint_uniform_color(np.array([0, 0, 1.0]))
                    sphere_list.append(sphere)
                o3d.visualization.draw_geometries([pcd] + sphere_list, width=1280, height=720)        

                # norm_depth_image = depth.astype(np.float32)
                # show_thre = 2000
                # norm_depth_image[norm_depth_image > show_thre] = show_thre


                # # norm_depth_image = (norm_depth_image)/(norm_depth_image.max())
                # # norm_depth_image = (norm_depth_image*255).astype(np.uint8)
                # # depth_colormap = cv2.applyColorMap(norm_depth_image, cv2.COLORMAP_JET)
                # plt.subplot(1,2,1)
                # plt.imshow(cad)
                # plt.title(str(data_i))
                # plt.subplot(1,2,2)
                # plt.imshow(norm_depth_image, vmin=0, vmax=show_thre)
                # plt.title(str(data_i))
                # # plt.subplot(1,3,3)
                # # ax = plt.axes(projection='3d')
                # # ax.scatter(scene_aruco_3d[:,0], scene_aruco_3d[:,1], scene_aruco_3d[:,2], marker = 'o')
                # plt.title(str(data_i))
                # plt.show()

            data_i += 1