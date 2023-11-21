import sys
import platform
import os
import shutil


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if platform.system() == "Linux":
    # 切换工作目录到当前脚本所在的目录
    os.chdir(SCRIPT_DIR)
sys.path.append(SCRIPT_DIR)

CFG_DIR         = f"{SCRIPT_DIR}/cfg"
DATASETS_DIR    = f"{SCRIPT_DIR}/datasets"
LOGS_DIR        = f"{SCRIPT_DIR}/logs"
WEIGHTS_DIR     = f"{SCRIPT_DIR}/weights"
SERVER_DATASET_DIR = "/home/nerc-ningxiao/datasets"

def _get_sub_log_dir(type):
    return f"{LOGS_DIR}/{type.__name__}_logs/"

try:
    import MyLib.posture_6d
    # sys.path.insert(0, MyLib.__path__[0])
    # shutil.rmtree(f"{SCRIPT_DIR}/posture_6d")
    # shutil.copytree(MyLib.posture_6d.__path__[0], f"{SCRIPT_DIR}/posture_6d")
except ModuleNotFoundError:
    pass

###
try:
    from .models.results import ImagePosture, ObjPosture
except:
    pass

def build_predictor(yolo_weight_path, branches_weight:dict[int, str], cfg_path):
    from models.OLDT import OLDT
    from launcher.Predictor import OLDTPredictor
    model = OLDT(yolo_weight_path, cfg_path, list(branches_weight.keys()))  # 替换为你自己的模型    
    for load_brach_i, load_from in branches_weight.items():
        model.load_branch_weights(load_brach_i, load_from)
    model.set_mode("predict")

    remark = "new_variable_length"
    predctor = OLDTPredictor(model, cfg_path)
    predctor.save_imtermediate = False

    return predctor