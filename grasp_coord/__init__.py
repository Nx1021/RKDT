from utils.yaml import yaml_load
from __init__ import SCRIPT_DIR, LOGS_DIR, CFG_DIR

cfg = yaml_load(f"{CFG_DIR}/config.yaml")
MODELS_DIR = cfg["models_dir"]
NUM_MODEL = len(cfg["pcd_models"])
PCD_MODELS = cfg["pcd_models"]

