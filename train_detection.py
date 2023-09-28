from __init__ import CFG_DIR, SCRIPT_DIR, WEIGHTS_DIR, DATASETS_DIR
import launcher
from launcher.setup import setup
from models import yolo8_patch
from ultralytics import YOLO
import torch
import platform
from utils.yaml import load_yaml, dump_yaml
from posture_6d.data.dataset_format import Mix_VocFormat
import os

if __name__ == "__main__":
    yolo_cfg_file = f"{CFG_DIR}/yolo_morrison_mix_single.yaml"
    cfg = load_yaml(yolo_cfg_file)
    # dataset_cfg = load_yaml(data_cfg_file)
    # weights_copy_path = yolo_weight_path#os.path.join(WEIGHTS_DIR, DATASET, SERIAL + "_best.pt")
    # TASK_MAP['detect'][1] = yolo8_patch.get_MyTrainer(weights_copy_path)

    # if detection_base_weight is None:
    #     detection_base_weight = f"{SCRIPT_DIR}/weights/yolov8l.pt"

    model = YOLO(f"{SCRIPT_DIR}/runs/detect/train46/weights/best.pt")  # load a pretrained model (recommended for training)
    model.train(cfg = yolo_cfg_file)  # train the model


# if __name__ == '__main__':
#     cfg_file = f"{CFG_DIR}/oldt_morrison_mix.yaml"
#     # torch.cuda.set_device("cuda:0")
#     for i in [2]:
#         setup_paras = load_yaml(cfg_file)["setup"]

#         sys = platform.system()
#         if sys == "Windows":
#             batch_size = 2
#             # model = torch.nn.DataParallel(model)
#         elif sys == "Linux":
#             batch_size = 32 # * torch.cuda.device_count()
#             # model = torch.nn.DataParallel(model)
#         setup_paras["ldt_branches"] = {i: ""}
#         setup_paras["batch_size"] = batch_size
#         setup_paras["sub_data_dir"] = f"linemod_mix/{str(i).rjust(6, '0')}"
#         setup("detection", **setup_paras)

# if __name__ == '__main__':
#     cfg_file = f"{CFG_DIR}/oldt_morrison_mix.yaml"
#     # torch.cuda.set_device("cuda:0")
#     setup_paras = load_yaml(cfg_file)["setup"]

#     sys = platform.system()
#     if sys == "Windows":
#         batch_size = 2
#         # model = torch.nn.DataParallel(model)
#     elif sys == "Linux":
#         batch_size = 32 # * torch.cuda.device_count()
#         # model = torch.nn.DataParallel(model)
#     setup_paras["ldt_branches"] = {}
#     setup_paras["batch_size"] = batch_size
#     setup_paras["sub_data_dir"] = f"morrison_mix/"
    
#     setup("detection", detection_base_weight=f"{SCRIPT_DIR}/runs/detect/train44/weights/best.pt", **setup_paras)
        
# if __name__ == '__main__':
#     cfg_file = f"{CFG_DIR}/oldt_linemod_mix.yaml"
#     # torch.cuda.set_device("cuda:0")
#     for i in [7, 8, 9, 10, 11, 12, 13, 14]:
#         setup_paras = load_yaml(cfg_file)["setup"]

#         sys = platform.system()
#         if sys == "Windows":
#             batch_size = 2
#             # model = torch.nn.DataParallel(model)
#         elif sys == "Linux":
#             batch_size = 32 # * torch.cuda.device_count()
#             # model = torch.nn.DataParallel(model)
#         setup_paras["ldt_branches"] = {i: ""}
#         setup_paras["batch_size"] = batch_size
#         setup_paras["sub_data_dir"] = f"linemod_mix/{str(i).rjust(6, '0')}"
#         setup("detection", **setup_paras)


if __name__ == '__main__':
    cfg_file = f"{CFG_DIR}/oldt_morrison_mix_single.yaml"
    # torch.cuda.set_device("cuda:0")
    setup_paras = load_yaml(cfg_file)["setup"]

    sys = platform.system()
    if sys == "Windows":
        batch_size = 2
        # model = torch.nn.DataParallel(model)
    elif sys == "Linux":
        batch_size = 64 # * torch.cuda.device_count()
        # model = torch.nn.DataParallel(model)
    setup_paras["ldt_branches"] = {}
    setup_paras["batch_size"] = batch_size
    setup_paras["sub_data_dir"] = f"morrison_mix_single/"
    
    for i in range(9):
        voc:Mix_VocFormat = Mix_VocFormat(f"{DATASETS_DIR}/morrison_mix_single/{str(i).rjust(6, '0')}")
        voc.set_elements_cache_priority(True)
        voc.labels_elements.cache_priority = False
        voc.images_elements.cache_priority = False
        voc.depth_elements.cache_priority = False
        voc.masks_elements.cache_priority = False
        server_full_data_dir = os.path.join(setup_paras["server_dataset_dir"], "morrison_mix_single", str(i).rjust(6, '0'))
        voc.copyto(server_full_data_dir, cover=True)

    setup("detection", detection_base_weight=f"{SCRIPT_DIR}/runs/detect/train46/weights/best.pt", **setup_paras)
       