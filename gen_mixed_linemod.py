from __init__ import DATASETS_DIR, CFG_DIR
import os
import cv2
import numpy as np
from posture_6d.data.dataset_format import LinemodFormat, VocFormat, FileCluster
from posture_6d.data.mesh_manager import MeshManager
from posture_6d.data.viewmeta import ViewMeta
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt

BASE_AUG_RATIO = 3
ISOLATE_AUG_RATIO = 5

def aug_viewmeta(viewmeta:ViewMeta):
    angle = np.random.rand() * 2 * np.pi/6 - np.pi/6
    brt = np.random.rand() * 100 - 50
    stu = np.random.rand() * 100 -50
    viewmeta = viewmeta.rotate(angle)
    viewmeta = viewmeta.change_brightness(brt)
    viewmeta = viewmeta.change_saturation(stu)
    return viewmeta

class BackgroundLoader:
    def __init__(self, *dirs):
        self.dirs = dirs

    def gen(self, w, h):
        resized_images = []
        while True:
            for d in self.dirs:
                for f in os.listdir(d):
                    image = cv2.imread(os.path.join(d, f))
                    resized_image = cv2.resize(image, (w, h))
                    yield resized_image

class MixLinemod_Modes(enumerate):
    DETECTION = 0
    OLDT = 1

class MixLinemod_VocFormat(VocFormat):
    def __init__(self, directory, data_num=0, split_rate=0.75, clear=False) -> None:
        super().__init__(directory, data_num, split_rate, clear)
        self.mode = MixLinemod_Modes.OLDT

    def _init_clusters(self):
        super()._init_clusters()
        self.base_logfile       = "base_log.txt"
        self.isolate_logfile    = "isolate_log.txt"
        self.oldt_train_file    = "oldt_train.txt"
        self.oldt_val_file      = "oldt_val.txt"
        self.base_iso_split_files = FileCluster(self, "", True,
                                                FileCluster.SingleFile(self.base_logfile,    self.loadsplittxt_func, self.savesplittxt_func),
                                                FileCluster.SingleFile(self.isolate_logfile, self.loadsplittxt_func, self.savesplittxt_func))
        self.oldt_split_files = FileCluster(self, "", True,
                                            FileCluster.SingleFile(self.oldt_train_file, self.loadsplittxt_func, self.savesplittxt_func),
                                            FileCluster.SingleFile(self.oldt_val_file,   self.loadsplittxt_func, self.savesplittxt_func))
        self.load_data_log()
        self.load_oldt_log()    

    def set_mode(self, mode:MixLinemod_Modes, train:bool):
        self.mode   = mode

    @property
    def train_idx_array(self):
        if self.mode == MixLinemod_Modes.DETECTION:
            return self.detection_train_idx_array
        elif self.mode == MixLinemod_Modes.OLDT:
            return self.oldt_train_idx_array
        else:
            raise ValueError("Unknown mode")
        
    @property
    def val_idx_array(self):
        if self.mode == MixLinemod_Modes.DETECTION:
            return self.detection_val_idx_array
        elif self.mode == MixLinemod_Modes.OLDT:
            return self.oldt_val_idx_array
        else:
            raise ValueError("Unknown mode")

    def load_data_log(self):
        if self.base_iso_split_files.all_exist:
            self.base_idx, self.isolate_idx = self.base_iso_split_files.read_all()
        else:
            self.base_idx       = np.array([], dtype=np.int32)
            self.isolate_idx    = np.array([], dtype=np.int32)

        # if os.path.exists(self.base_logfile):
        #     self.base_idx = np.loadtxt(self.base_logfile, dtype=np.int32).reshape(-1)
        # else:
        #     self.base_idx = np.array([], dtype=np.int32)
        # if os.path.exists(self.isolate_logfile):
        #     self.isolate_idx = np.loadtxt(self.isolate_logfile, dtype=np.int32).reshape(-1)
        # else:
        #     self.isolate_idx = np.array([], dtype=np.int32)
    
    def load_oldt_log(self):
        if self.oldt_split_files.all_exist:
            self.oldt_train_idx_array, self.oldt_val_idx_array = self.oldt_split_files.read_all()
        else:
            base_idx = self.base_idx.copy()
            base_idx = base_idx[base_idx % BASE_AUG_RATIO == 0] 
            np.random.shuffle(base_idx)
            self.oldt_val_idx_array = base_idx[:int(len(base_idx)*0.85)]
            self.oldt_train_idx_array = np.setdiff1d(
                np.union1d(self.base_idx, self.isolate_idx),
                self.oldt_val_idx_array
                )
            self.oldt_split_files.write_all((self.oldt_train_idx_array, self.oldt_val_idx_array))


        # if os.path.exists(self.oldt_train_file) and os.path.exists(self.oldt_val_file):
        #     self.oldt_train_idx_array = np.loadtxt(self.oldt_train_file, dtype=np.int32).reshape(-1)
        #     self.oldt_val_idx_array = np.loadtxt(self.oldt_val_file, dtype=np.int32).reshape(-1)
        # else:
        #     base_idx = self.base_idx.copy()
        #     base_idx = base_idx[base_idx % base_aug_ratio == 0] 
        #     np.random.shuffle(base_idx)
        #     self.oldt_val_idx_array = base_idx[:int(len(base_idx)*0.85)]
        #     self.oldt_train_idx_array = np.setdiff1d(
        #         np.union1d(self.base_idx, self.isolate_idx),
        #         self.oldt_val_idx_array
        #         )
        #     with open(self.oldt_train_file, "w") as f:
        #         f.writelines("".join([str(i)+"\n" for i in self.oldt_train_idx_array]))
        #     with open(self.oldt_val_file, "w") as f:
        #         f.writelines("".join([str(i)+"\n" for i in self.oldt_val_idx_array]))
    
    def save_log(self):
        self.base_iso_split_files.write_all((self.base_idx, self.isolate_idx))
        # with open(self.base_logfile, "w") as f:
        #     f.writelines("".join([str(i)+"\n" for i in self.base_idx]))
        # with open(self.isolate_logfile, "w") as f:
        #     f.writelines([str(i)+"\n" for i in self.isolate_idx])
    
    def clear(self, ignore_warning=False):
        super().clear(ignore_warning)
        self.base_idx = np.array([], dtype=np.int32)
        self.isolate_idx = np.array([], dtype=np.int32)
        self.base_iso_split_files.clear()
        # with open(self.base_logfile, "w") as f:
        #     pass
        # with open(self.isolate_logfile, "w") as f:
        #     pass

    def stop_writing(self):
        self.save_log()
        super().stop_writing()
    
    
def aug_isolate(viewmeta:ViewMeta, bgg):
    viewmeta = viewmeta.rotate()


if __name__ == '__main__':
    bgloader = BackgroundLoader(r"E:\shared\code\yolo_v8\datasets\VOC\images\test2007", 
                            r"E:\shared\code\yolo_v8\datasets\VOC\images\train2007",
                            r"E:\shared\code\yolo_v8\datasets\VOC\images\train2012",
                            r"E:\shared\code\yolo_v8\datasets\VOC\images\val2007",
                            r"E:\shared\code\yolo_v8\datasets\VOC\images\val2012").gen(640, 480)
    mm = MeshManager("E:/shared/code/OLDT/datasets/linemod/models", {0: "obj_000001.ply",
                                                                    1: "obj_000002.ply",
                                                                    2: "obj_000003.ply",
                                                                    3: "obj_000004.ply",
                                                                    4: "obj_000005.ply",
                                                                    5: "obj_000006.ply",
                                                                    6: "obj_000007.ply",
                                                                    7: "obj_000008.ply",
                                                                    8: "obj_000009.ply",
                                                                    9: "obj_000010.ply",
                                                                    10: "obj_000011.ply",
                                                                    11: "obj_000012.ply",
                                                                    12: "obj_000013.ply",
                                                                    13: "obj_000014.ply",
                                                                    14: "obj_000015.ply"})    
    mesh_dict = mm.get_meta_dict()
    for di in range(1, 15):
        isolate = LinemodFormat(r"E:\shared\code\OLDT\datasets\linemod_isolate\{}".format(str(di+1).rjust(6, "0")))
        lmvf = VocFormat(r"E:\shared\code\OLDT\datasets\linemod\{}".format(str(di).rjust(6, "0")))
        lmmix_vf = MixLinemod_VocFormat(r"E:\shared\code\OLDT\datasets\linemod_mix\{}".format(str(di).rjust(6, "0")), ISOLATE_AUG_RATIO*isolate.data_num + BASE_AUG_RATIO*lmvf.data_num)
        lmmix_vf.close_all(False)
        
        with lmmix_vf.writer:
            for viewmeta in tqdm(lmvf.read_from_disk()):
                for _ in range(BASE_AUG_RATIO):
                    lmmix_vf.base_idx.append(lmmix_vf.data_num)
                    if _ != 0:
                        v = copy.deepcopy(aug_viewmeta(viewmeta))
                    else:
                        v = copy.deepcopy(viewmeta)
                    lmmix_vf.write_to_disk(v)
                
        
        with lmmix_vf.writer:
            for viewmeta in tqdm(isolate.read_from_disk()):
                viewmeta.modify_class_id([(di+1, di)])
                viewmeta.visib_fract = {di: 1.0}
                viewmeta.calc_by_base(mesh_dict)
                ### use background from backgroundloader
                ###
                for _ in range(ISOLATE_AUG_RATIO):
                    lmmix_vf.isolate_idx.append(lmmix_vf.data_num)
                    if _ != 0:
                        v = copy.deepcopy(aug_viewmeta(viewmeta))
                    else:
                        v = copy.deepcopy(viewmeta)
                    bg = next(bgloader)
                    mask = list(v.masks.values())[0].astype(np.bool8)
                    bg[mask] = v.color[mask]
                    v.color = bg
                    lmmix_vf.write_to_disk(v)
