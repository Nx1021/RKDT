from utils.yaml import yaml_load

cfg = yaml_load("./cfg/config.yaml")
MODELS_DIR = cfg["models_dir"]
NUM_MODEL = len(cfg["pcd_models"])
PCD_MODELS = cfg["pcd_models"]

