# 在还没有指定TypeVar类型，其类型为bound指定的类型时： Generic、参数正确类型输入、属性注释缺一不可
# 在子类指定了Generic的类型后，其类型被绑定，子类的子类无法再修改已经绑定的TypeVar
# 如果子类未绑定，则可以在子类的子类再指定
from __init__ import DATASETS_DIR, CFG_DIR
# from posture_6d.data.dataset_format import VocFormat

# from gen_mixed_linemod import MixLinemod_VocFormat
# from posture_6d.data.mesh_manager import MeshManager
# from posture_6d.data.dataset_format import ClusterIONotExecutedWarning
# from posture_6d.posture import Posture
# from posture_6d.derive import PnPSolver
# import numpy as np
# from tqdm import tqdm

# from gen_mixed_linemod import MixLinemod_VocFormat
# from posture_6d.data.mesh_manager import MeshManager
# from posture_6d.data.viewmeta import ViewMeta
from MyLib.posture_6d.data.dataset_format import Mix_VocFormat
import numpy as np
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt
import torch
from pympler import asizeof
from scipy.linalg import logm
import warnings
from typing import TypeVar

from tqdm import tqdm
import time

from typing import Any, Union, Callable, TypeVar, Generic, Iterable


# vm = Mix_VocFormat(f"{DATASETS_DIR}/morrison_mix")
# with vm.writer.allow_overwriting():
#     pass

# vm.gen_posture_log()
# vm.spliter_group.cur_training_spliter
# vm.spliter_group.split_mode_list
# vm.spliter_group.set_split_mode("posture")
# vm.visib_fract_elements
# vm.children

# class A():
#     def __new__(cls, *arg, **kw):
#         print("paras:", *arg, **kw)
#         print("A new")
#         return super().__new__(cls)
    
#     def __init_subclass__(cls) -> None:
#         print("A init_subclass")
#         print(cls.__name__)
#         def f(obj, *arg, **kw):
#             print("f")
#             cls._org_init__(obj, *arg, **kw)
#         cls._org_init__ = cls.__init__
#         cls.__init__ = f
#         super().__init_subclass__()

#     def __init__(self, x = 1) -> None:
#         print("A init")
#         print(x)

    # def __getattribute__(self, __name: str) -> Any:
    #     print(__name)
    #     return super().__getattribute__(__name)
    
# class B(A):
#     def __init__(self) -> None:
#         print("B init")
#         super().__init__()

# class C(B):
#     pass

# a = A(2)
 
from typing import Type
T = TypeVar('T', bound="MyClass")

class MyClass(Generic[T]):
    def __init__(self, value):
        self.value:type(self) = value

class MyBaseClass(MyClass['MyBaseClass'], Generic[T]):
    pass

class MySubClass(MyBaseClass):
    pass

MyClass().value
MyBaseClass(1).value
MySubClass(1).value

vm_test = Mix_VocFormat(f"{DATASETS_DIR}/morrison_mix_test")
vm_test = Mix_VocFormat(f"{DATASETS_DIR}/morrison_mix_test")



with vm_test.writer.allow_overwriting():
    for i in range(10):
        vm_test[i] = vm[i]
        vm_test.record_data_type(i, True, False)

# vm.labels_elements.default_image_size = (640, 480)
# with vm.labels_elements.writer.allow_overwriting():
#     vm.labels_elements.unzip_cache()

# for i in range(3, 15):
#     d = f"{DATASETS_DIR}/linemod_mix/{str(i).rjust(6, '0')}"
#     vc = MixLinemod_VocFormat(d)
#     vc.save_elements_cache()
#     vc.set_elements_cachemode(True)
#     vc.copyto(f"F:\\{str(i).rjust(6, '0')}")
    # vc.labels_elements.save_cache(image_size=[(640, 480) for _ in range(len(vc.labels_elements))])
    # vc.labels_elements.cache_mode = True

    # # vc.bbox_3ds_elements.save_cache()
    # vc.bbox_3ds_elements.cache_mode = True

    # # vc.depth_scale_elements.save_cache()
    # vc.depth_scale_elements.cache_mode = True

    # # vc.intr_elements.save_cache()
    # vc.intr_elements.cache_mode = True

    # # vc.landmarks_elements.save_cache()
    # vc.landmarks_elements.cache_mode = True

    # # vc.extr_vecs_elements.save_cache()
    # vc.extr_vecs_elements.cache_mode = True

    # # vc.visib_fract_elements.save_cache()
    # vc.visib_fract_elements.cache_mode = True


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
