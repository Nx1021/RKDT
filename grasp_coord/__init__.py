from utils.yaml import load_yaml
from __init__ import SCRIPT_DIR, LOGS_DIR, CFG_DIR
import os

cfg = load_yaml(f"{CFG_DIR}/oldt_morrison_real_voc.yaml")
MODELS_DIR = os.path.join(SCRIPT_DIR, cfg["models_dir"])
NUM_MODEL = len(cfg["pcd_models"])
PCD_MODELS = cfg["pcd_models"]

