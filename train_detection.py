from __init__ import CFG_DIR, SCRIPT_DIR, WEIGHTS_DIR
import launcher
import os
import shutil
from models import yolo8_patch 
from ultralytics.yolo.utils.torch_utils import select_device
from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics import YOLO
from utils.yaml import load_yaml

def save_model_decorator(save_model_func, cfgfile):
    cfg = load_yaml(cfgfile)
    dataset_cfg = load_yaml(cfg["data"])
    weights_copy_path = dataset_cfg["path"] + "_best.pt"
    def wrapper(obj:BaseTrainer):
        best_last_time = os.path.getmtime(obj.best)
        rlt = save_model_func(obj)
        if best_last_time != os.path.getmtime(obj.best):
            shutil.copy(obj.best, weights_copy_path)
        return rlt
    return wrapper

        
    
if __name__ == '__main__':
    # Load a model
    # model = YOLO("yolov8l.yaml")  # build a new model from scratch
    model = YOLO(f"{SCRIPT_DIR}/weights/yolov8l.pt")  # load a pretrained model (recommended for training)
    model.train(cfg = f"{CFG_DIR}/yolo_linemod_mix.yaml")  # train the model
    trainer:BaseTrainer = model.trainer
    trainer.save_model = save_model_decorator(BaseTrainer.save_model, f"{CFG_DIR}/yolo_linemod_mix.yaml")
    metrics = model.val()  # evaluate model performance on the validation set
    results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    path = model.export(format="onnx")  # export the model to ONNX format
