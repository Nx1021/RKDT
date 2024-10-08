import ruamel.yaml
import os
from . import SCRIPT_DIR

loaded_cfg = {}

def load_yaml(path='data.yaml', assingle=False) -> dict:
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        (dict): YAML data and file name.
    """
    def load():
        with open(path, 'r') as file:
            yaml = ruamel.yaml.YAML()
            return yaml.load(file)

    if assingle:
        if path in loaded_cfg:
            yaml_data = loaded_cfg[path]
        else:
            yaml_data = load()
            loaded_cfg[path] = yaml_data
    else:
        yaml_data = load()

    # if "models_dir" in yaml_data:
    #     yaml_data["models_dir"] = os.path.join(SCRIPT_DIR, yaml_data["models_dir"])
    return yaml_data

def dump_yaml(file_path, data):
    with open(file_path, 'w') as file:
        yaml = ruamel.yaml.YAML()
        yaml.dump(data, file)
