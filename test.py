# 在还没有指定TypeVar类型，其类型为bound指定的类型时： Generic、参数正确类型输入、属性注释缺一不可
# 在子类指定了Generic的类型后，其类型被绑定，子类的子类无法再修改已经绑定的TypeVar
# 如果子类未绑定，则可以在子类的子类再指定
from __init__ import DATASETS_DIR, CFG_DIR
from MyLib.posture_6d.data.dataset_format import VocFormat, Mix_VocFormat, Spliter

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
from MyLib.posture_6d.core.utils import Table
from MyLib.posture_6d.core.posture import Posture
from MyLib.posture_6d.core import CameraIntr
from MyLib.posture_6d.derive import calc_bbox_2d_proj, calc_Z_rot_angle_to_horizontal
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




def plot_frame(camera_extrinsics, new_camera_extrinsics = None):
    import open3d as o3d
    import numpy as np

    # 创建一个Open3D可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 创建世界坐标系的坐标轴
    world_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0)

    # 将世界坐标系添加到可视化窗口
    vis.add_geometry(world_coordinate_frame)

    # 创建相机坐标系的坐标轴
    camera_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0)

    # 将相机坐标系的坐标轴转换到相机外参矩阵的位置
    camera_coordinate_frame.transform(camera_extrinsics)

    # 将相机坐标系添加到可视化窗口
    vis.add_geometry(camera_coordinate_frame)

    if new_camera_extrinsics is not None:
        new_camera_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=120.0)
        new_camera_coordinate_frame.transform(new_camera_extrinsics)
        vis.add_geometry(new_camera_coordinate_frame)

    # 创建世界坐标系的坐标轴
    trans_world_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0, origin=camera_extrinsics[:3, 3])

    # 将世界坐标系添加到可视化窗口
    vis.add_geometry(trans_world_coordinate_frame)

    # 将平面添加到可视化窗口
    vis.add_geometry(trans_world_coordinate_frame)

    # 设置渲染参数
    render_option = vis.get_render_option()
    render_option.point_size = 20

    # 运行可视化窗口
    vis.run()
    vis.destroy_window()
from abc import ABC, abstractmethod
# mm = MeshManager(f"{DATASETS_DIR}/morrison_models")

# meshmeta_dict = mm.get_meta_dict()

# with vm_single.writer.allow_overwriting():
#     for i in vm_single.keys():
#         new_dir = "train" if i in new_split[0] else "val"
#         for c in list(vm_single.elements_map.values()):
#             c.change_dir(i, new_dir)
# # TODO 建立split和data_i_map的关系

vm = Mix_VocFormat(f"{DATASETS_DIR}/morrison_mix_single/000000")

for v in vm:
    v.plot()
# vm_test = Mix_VocFormat(f"{DATASETS_DIR}/morrison_mix_test")
# for c in vm_test.clusters:
#     c._update_cluster_all()
# with vm_test.writer:
#     for v in tqdm(vm[:10]):
#         vm_test.write_to_disk(v)
#         vm_test.record_data_type(vm_test.data_num - 1, True, False)

std_T_dir = r"E:\shared\code\OLDT\datasets\morrison_models\std_posture"

std_infos = []
valid_data:dict[int, list[int]] = {}
for f in os.listdir(std_T_dir):
    T_O2S = np.load(os.path.join(std_T_dir, f))
    T_O2S[:3, 3] = 0
    T_O2S[3, :3] = 0
    class_id = int(f[:6])
    name = f[7:-4]
    std_infos.append((class_id, name, T_O2S))
    valid_data.setdefault(class_id, [])


camera_intr = CameraIntr(vm.intr_elements[0], 640, 480)

for obj_i in range(9):
    vm_single = Mix_VocFormat(f"{DATASETS_DIR}/morrison_mix_single/{str(obj_i).rjust(6, '0')}")
    # vm_single.clear(force=True)
    for c in vm_single.clusters:
        c._update_cluster_all()
    T_O2S = std_infos[obj_i][2]
    # with vm_single.writer:
    split = vm.posture_spliter.get_split(f"obj_{str(obj_i).rjust(2, '0')}")
    data_i_list = split["train"] + split["val"]
    data_i_list.sort()
    meshmeta = mm.export_meta(obj_i)
    # 计算相机的滚转角
    with vm_single.writer:
        for data_i in tqdm(data_i_list):
            if data_i in vm.basis_spliter.get_idx_list(vm.BASIS_SUBSETS[1]):
                continue
            trans_dict = vm.extr_vecs_elements[data_i]
            rvec, tvec = trans_dict[obj_i]
            T_O2C = Posture(rvec=rvec, tvec=tvec).trans_mat
            T_S2C = T_O2S @ np.linalg.inv(T_O2C)
            # roll = np.arctan2(T_S2C[1, 0], T_S2C[0, 0])
            theta = calc_Z_rot_angle_to_horizontal(T_S2C) # R @ T_O2S @ np.linalg.inv(T_O2C)

            v = vm[data_i]
            new_v = v.rotate(-theta)
            new_v.labels = None
            new_v.calc_by_base(meshmeta_dict)

            vm_single.write_to_disk(new_v)
            vm_single.spliter_group.record_set( vm_single.data_i_upper - 1 ,
                vm.spliter_group.query_set(data_i)
            )


        # T_C2S = np.linalg.inv(T_O2C) @ T_O2S
        # if T_S2C[2, 3] > 0:
        #     # vm.spliter_group.cur_training_spliter.set_one(data_i, "train", f"obj_{str(class_id).rjust(2, '0')}")
        #     valid_data[class_id].append(data_i)

    # for data_i in tqdm():
    #     if obj_i in vm.visib_fract_elements[data_i]:
    #         vm_single.write_to_disk(vm[data_i])
    #         vm_single.record_data_type(vm.data_num - 1, True, False)

# for i in range(15):
#     vm = Mix_VocFormat(f"{DATASETS_DIR}/linemod_mix/{str(i).rjust(6, '0')}")
#     real_split = np.loadtxt(os.path.join(vm.directory, "base_log.txt"), dtype=np.int32).tolist()
#     sim_split = np.loadtxt(os.path.join(vm.directory, "isolate_log.txt"), dtype=np.int32).tolist()

#     posture_train_split = np.loadtxt(os.path.join(vm.directory, "oldt_train.txt"), dtype=np.int32).tolist()
#     posture_val_split = np.loadtxt(os.path.join(vm.directory, "oldt_val.txt"), dtype=np.int32).tolist()

#     default_train_split = np.loadtxt(os.path.join(vm.directory, "train.txt"), dtype=np.int32).tolist()
#     default_val_split = np.loadtxt(os.path.join(vm.directory, "val.txt"), dtype=np.int32).tolist()

#     base_split = [x for x in real_split if x%3 == 0] + sim_split
#     aug_split = [x for x in real_split if x%3 != 0]

#     [x.clear_idx() for x in vm.spliter_group.clusters]

#     vm.reality_spliter.get_idx_list(vm.REALITY_SUBSETS[0]).extend(real_split)
#     vm.reality_spliter.get_idx_list(vm.REALITY_SUBSETS[1]).extend(sim_split)

#     vm.posture_spliter.get_idx_list("train").extend(posture_train_split)
#     vm.posture_spliter.get_idx_list("val").extend(posture_val_split)

#     vm.default_spliter.get_idx_list("train").extend(default_train_split)
#     vm.default_spliter.get_idx_list("val").extend(default_val_split)

#     vm.basis_spliter.get_idx_list(vm.BASIS_SUBSETS[0]).extend(base_split)
#     vm.basis_spliter.get_idx_list(vm.BASIS_SUBSETS[1]).extend(aug_split)

#     vm.spliter_group.save()


# vm = Mix_VocFormat(f"{DATASETS_DIR}/morrison_mix/")
# # vm.save_elements_data_info_map()
# vm.set_elements_cache_priority(True)
# vm.spliter_group.set_cur_spliter_name("posture")
# for i in range(9):
#     vm.spliter_group.cur_training_spliter.set_split_mode(f"obj_{str(i).rjust(2, '0')}")
#     base = vm.spliter_group.cur_training_spliter.get_idx_list("train") + vm.spliter_group.cur_training_spliter.get_idx_list("val")
#     print(len(base))
#     for di in list(base):
#         vf = vm.visib_fract_elements[di][i]
#         if vf < 0.6:
#             base.remove(di)
#     print(len(base))
#     vm.gen_posture_log(0.5, base)
# vm.spliter_group.cur_training_spliter.set_split_mode("obj_00")
# for data_i in vm.train_idx_array[::100]:
#     print(data_i)
#     v = vm[data_i]
#     v.plot()
#     plt.show()


# with vm.writer.allow_overwriting():
#     for data_i in range(20000, 40000):
#         if data_i in vm.images_elements.keys():
#             vm.remove_one(data_i)
        # v = vm[data_i]
        # aug_v = v.change_brightness(np.random.randint(-100, 100))
        # aug_v = v.change_saturation(np.random.randint(-100, 100))
        # vm.write_to_disk(aug_v)
        # vm.record_data_type(vm.data_num - 1, True, False)

# std_T_dir = r"E:\shared\code\OLDT\datasets\morrison_models\std_posture"

# std_infos = []
# valid_data:dict[int, list[int]] = {}
# for f in os.listdir(std_T_dir):
#     T_O2S = np.load(os.path.join(std_T_dir, f))
#     T_O2S[:3, 3] = 0
#     class_id = int(f[:6])
#     name = f[7:-4]
#     std_infos.append((class_id, name, T_O2S))
#     valid_data.setdefault(class_id, [])

# vm.spliter_group.set_cur_spliter_name("posture")

# for data_i, trans_dict in tqdm(vm.extr_vecs_elements.items(), total=len(vm)):
#     for class_id, name, T_O2S in std_infos:
#         if class_id in trans_dict:
#             rvec, tvec = trans_dict[class_id]
#             T_O2C = Posture(rvec=rvec, tvec=tvec).trans_mat
#             # P_OinO = P_CinC = P_SinS = np.array([[0,0,0,1], [100,0,0,1], [0,100,0,1], [0,0,100,1]])
#             # P_OinC = P_OinO @ T_O2C.T # (T_O2C @ P_OinO.T).T
#             # P_OinS = P_OinO @ T_O2S.T # (T_O2S @ P_OinO.T)
#             # P_CinO = P_CinC @ np.linalg.inv(T_O2C).T
#             # P_SinO = P_SinS @ np.linalg.inv(T_O2S).T

#             # P_CinS = (P_CinO @ T_O2S.T)
#             # P_SinS = (P_SinO @ T_O2S.T)
#             # T_S2C = P_CinS.T @ np.linalg.inv(P_CinC.T)
#             T_S2C = T_O2S @ np.linalg.inv(T_O2C)
#             # T_C2S = np.linalg.inv(T_O2C) @ T_O2S
#             if T_S2C[2, 3] > 0:
#                 # vm.spliter_group.cur_training_spliter.set_one(data_i, "train", f"obj_{str(class_id).rjust(2, '0')}")
#                 valid_data[class_id].append(data_i)
#     # if data_i > 100:
#     #     break
# for class_id, data_ids in valid_data.items():
#     split_mode = f"obj_{str(class_id).rjust(2, '0')}"
#     vm.spliter_group.cur_training_spliter.add_split_mode(split_mode, exist_ok=True)
#     vm.spliter_group.cur_training_spliter.set_split_mode(split_mode)
#     vm.gen_posture_log(0.15, data_ids)
# vm.spliter_group.save()




# vm.labels_elements.default_image_size = (640, 480)
# with vm.labels_elements.writer.allow_overwriting():
#     vm.labels_elements.unzip_cache()

# for i in range(3, 15):
#     d = f"{DATASETS_DIR}/linemod_mix/{str(i).rjust(6, '0')}"
#     vc = MixLinemod_VocFormat(d)
#     vc.save_elements_cache()
#     vc.set_elements_cache_priority(True)
#     vc.copyto(f"F:\\{str(i).rjust(6, '0')}")
    # vc.labels_elements.save_cache(image_size=[(640, 480) for _ in range(len(vc.labels_elements))])
    # vc.labels_elements.cache_priority = True

    # # vc.bbox_3ds_elements.save_cache()
    # vc.bbox_3ds_elements.cache_priority = True

    # # vc.depth_scale_elements.save_cache()
    # vc.depth_scale_elements.cache_priority = True

    # # vc.intr_elements.save_cache()
    # vc.intr_elements.cache_priority = True

    # # vc.landmarks_elements.save_cache()
    # vc.landmarks_elements.cache_priority = True

    # # vc.extr_vecs_elements.save_cache()
    # vc.extr_vecs_elements.cache_priority = True

    # # vc.visib_fract_elements.save_cache()
    # vc.visib_fract_elements.cache_priority = True


# SERVER_DATASET_DIR = "/home/nerc-ningxiao/datasets/linemod_mix/000000"
# d_to = f"{SERVER_DATASET_DIR}/labels"
# d = f"{DATASETS_DIR}/linemod_mix/000000/labels"
# try:
#     shutil.rmtree(d_to)
# except:
#     pass
# shutil.copytree(d, d_to)

# mm = MeshManager(f"{DATASETS_DIR}/linemod/models")
# pnp = PnPSolver(r"E:\shared\code\OLDT\datasets\linemod\models\default_K.txt")
# for i in range(1, 15):
#     d = f"{DATASETS_DIR}/linemod_mix/{str(i).rjust(6, '0')}"
#     vc = MixLinemod_VocFormat(d)
    
#     for v in vc:
#         v:ViewMeta = v
#         for k in v.extr_vecs:
#             rvec, tvec = v.extr_vecs[k]
#             ldmk = v.landmarks[k] #[24, 2]
#             ldmk_3d = mm.get_ldmk_3d(k) # [24, 3]
#             new_rvec, new_tvec = pnp.solvepnp(ldmk, ldmk_3d)            
#             reproj = pnp.calc_reproj(ldmk_3d, new_rvec, new_tvec)
#             print(np.max(reproj - ldmk))

#         v.plot()
#         plt.show()

# mm = MeshManager(f"{DATASETS_DIR}/linemod/models")
# pnp = PnPSolver(r"E:\shared\code\OLDT\datasets\linemod\models\default_K.txt")
# for i in range(1, 15):
#     d = f"{DATASETS_DIR}/linemod_mix/{str(i).rjust(6, '0')}"
#     vc = MixLinemod_VocFormat(d)
#     vc.close_all()
#     vc.allow_overwrite = True
#     label_elements = vc.labels_elements
#     extr_vecs_elements = vc.extr_vecs_elements
#     landmarks_elements = vc.landmarks_elements

#     label_elements.open()
#     extr_vecs_elements.open()
#     landmarks_elements.open()
#     extr_vecs_elements.set_writable()
#     label_elements.set_writable()

#     for data_i in tqdm(landmarks_elements.keys()):
#         label_dict = {}
#         extr_dict = {}

#         ldmk_dict = landmarks_elements[data_i]

#         for k in ldmk_dict.keys():
#             ldmk = ldmk_dict[k]
#             ldmk_3d = mm.get_ldmk_3d(k)
#             rvec, tvec = pnp.solvepnp(ldmk, ldmk_3d)   
#             extr_dict[k] = np.array([np.squeeze(rvec), np.squeeze(tvec)])

#             points = mm.get_model_pcd(k)
#             proj:np.ndarray = pnp.calc_reproj(points, rvec, tvec) #[N, 2]
#             bbox = np.array(
#                 [np.min(proj[:, 0]), np.min(proj[:, 1]), np.max(proj[:, 0]), np.max(proj[:, 1])]
#             ) # [xmin, ymin, xmax, ymax]
#             # normalize
#             label_dict[k] = bbox
#         appdir, appname = landmarks_elements.auto_path(data_i, return_app=True)
#         label_elements.write(data_i, label_dict, appdir=appdir, appname=appname, image_size=(640, 480))
#         extr_vecs_elements.write(data_i, extr_dict, appdir=appdir, appname=appname)
#         # label_elements[data_i] = label_dict
