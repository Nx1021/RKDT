from __init__ import CFG_DIR, SCRIPT_DIR, WEIGHTS_DIR
import launcher
import os
import shutil
from models import yolo8_patch 
from utils.yaml import load_yaml, dump_yaml
from ultralytics.yolo.utils.torch_utils import select_device
from ultralytics.yolo.engine import trainer
from ultralytics.yolo.engine.model import TASK_MAP
from ultralytics import YOLO
import torch


cfgfile = f"{CFG_DIR}/yolo_linemod_mix.yaml"
cfg = load_yaml(cfgfile)
dataset_cfg = load_yaml(cfg["data"])
weights_copy_path = os.path.join(WEIGHTS_DIR, dataset_cfg["path"] + "_best.pt")
TASK_MAP['detect'][1] = yolo8_patch.get_MyTrainer(weights_copy_path)

def clear_invalid():
    dir_ = os.path.join(SCRIPT_DIR, "runs/detect")
    for d in os.listdir(dir_):
        weights_dir = os.path.join(dir_, d, "weights")
        if os.path.exists(weights_dir) and len(os.listdir(weights_dir)) > 0:
            pass
        else:
            to_rm = os.path.join(dir_, d)
            print(to_rm, " is removed")
            shutil.rmtree(to_rm)

if __name__ == '__main__':
    torch.cuda.set_device("cuda:0")
    clear_invalid()
    for i in [1,3]:
        cfgfile = f"{CFG_DIR}/yolo_linemod_mix.yaml"
        cfg = load_yaml(cfgfile)
        dataset_cfg = load_yaml(cfg["data"])
        dataset_cfg["path"] = "./linemod_mix/{}".format(str(i).rjust(6, '0'))
        dump_yaml(cfg["data"], dataset_cfg)
        weights_copy_path = os.path.join(WEIGHTS_DIR, dataset_cfg["path"] + "_best.pt")
        TASK_MAP['detect'][1] = yolo8_patch.get_MyTrainer(weights_copy_path)
    # Load a model
    # model = YOLO("yolov8l.yaml")  # build a new model from scratch
        model = YOLO(f"{SCRIPT_DIR}/weights/yolov8l.pt")  # load a pretrained model (recommended for training)

        model.train(cfg = cfgfile)  # train the model
    # metrics = model.val()  # evaluate model performance on the validation set
    # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    # path = model.export(format="onnx")  # export the model to ONNX format
