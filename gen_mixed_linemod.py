from __init__ import DATASETS_DIR, CFG_DIR
import os
import cv2
import numpy as np
from posture_6d.dataset_format import LinemodFormat, VocFormat
from posture_6d.mesh_manager import MeshManager
from posture_6d.viewmeta import ViewMeta
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt

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

class MixLinemod_VocFormat(VocFormat):
    class _Writer(VocFormat._Writer):
        def __exit__(self, exc_type, exc_value: Exception, traceback):
            format_obj:MixLinemod_VocFormat = self.format_obj
            format_obj.save_log()
            return super().__exit__(exc_type, exc_value, traceback)

    def __init__(self, directory, data_num=0, split_rate=0.75, clear=False) -> None:
        super().__init__(directory, data_num, split_rate, clear)
        self.base_logfile = f"{self.directory}/base_log.txt"
        self.isolate_logfile = f"{self.directory}/isolate_log.txt"
        self.load_log()
    
    def load_log(self):
        if os.path.exists(self.base_logfile):
            with open(self.base_logfile, "r") as f:
                self.base_log = [int(i) for i in f.readlines()]
        else:
            self.base_log = []
        if os.path.exists(self.isolate_logfile):
            with open(self.isolate_logfile, "r") as f:
                self.isolate_log = [int(i) for i in f.readlines()]
        else:
            self.isolate_log = []
    
    def save_log(self):
        with open(self.base_logfile, "w") as f:
            f.writelines("".join([str(i)+"\n" for i in self.base_log]))
        with open(self.isolate_logfile, "w") as f:
            f.writelines([str(i)+"\n" for i in self.isolate_log])
    
    def clear(self, ignore_warning=False):
        super().clear(ignore_warning)
        self.base_log.clear()
        self.isolate_log.clear()
        with open(self.base_logfile, "w") as f:
            pass
        with open(self.isolate_logfile, "w") as f:
            pass
    
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
    base_aug_ratio = 3
    isolate_aug_ratio = 5
    for di in range(1, 15):
        isolate = LinemodFormat(r"E:\shared\code\OLDT\datasets\linemod_isolate\{}".format(str(di+1).rjust(6, "0")))
        lmvf = VocFormat(r"E:\shared\code\OLDT\datasets\linemod\{}".format(str(di).rjust(6, "0")))
        lmmix_vf = MixLinemod_VocFormat(r"E:\shared\code\OLDT\datasets\linemod_mix\{}".format(str(di).rjust(6, "0")), isolate_aug_ratio*isolate.data_num + base_aug_ratio*lmvf.data_num)
        lmmix_vf.close_all(False)
        
        with lmmix_vf.start_to_write():
            for viewmeta in tqdm(lmvf.read_from_disk()):
                for _ in range(base_aug_ratio):
                    lmmix_vf.base_log.append(lmmix_vf.data_num)
                    if _ != 0:
                        v = copy.deepcopy(aug_viewmeta(viewmeta))
                    else:
                        v = copy.deepcopy(viewmeta)
                    lmmix_vf.write_to_disk(v)
                
        
        with lmmix_vf.start_to_write():
            for viewmeta in tqdm(isolate.read_from_disk()):
                viewmeta.modify_class_id([(di+1, di)])
                viewmeta.visib_fract = {di: 1.0}
                viewmeta.calc_by_base(mesh_dict)
                ### use background from backgroundloader
                ###
                for _ in range(isolate_aug_ratio):
                    lmmix_vf.isolate_log.append(lmmix_vf.data_num)
                    if _ != 0:
                        v = copy.deepcopy(aug_viewmeta(viewmeta))
                    else:
                        v = copy.deepcopy(viewmeta)
                    bg = next(bgloader)
                    mask = list(v.masks.values())[0].astype(np.bool8)
                    bg[mask] = v.color[mask]
                    v.color = bg
                    lmmix_vf.write_to_disk(v)
