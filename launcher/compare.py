from . import LOGS_DIR, _get_sub_log_dir
from .Trainer import Trainer

import os
from utils.yaml import yaml_dump, yaml_load
import pandas as pd

def compare_train_log(sub_dirs):
    root_dir = _get_sub_log_dir(Trainer)
    yaml_names = ["setup.yaml", "config.yaml"]
    for yaml_name in yaml_names:
        compare_yaml_files(root_dir, sub_dirs, yaml_name)

def compare_yaml_files(root_dir, sub_dirs, yaml_name):
    data = {}
    all_keys = set()

    for subdirectory in sub_dirs:
        setup_path = os.path.join(root_dir, subdirectory, yaml_name)

        yaml_data = yaml_load(setup_path)

        data[subdirectory] = yaml_data
        all_keys.update(yaml_data.keys())

    diff_data = {}
    for key in all_keys:
        values = {}
        for subdir, subdir_data in data.items():
            value = subdir_data.get(key)
            values[subdir] = value
        value_list = list(values.values())
        try:
            set_ = set(value_list)
        except TypeError:
            set_ = set([str(x) for x in value_list])
        if len(set_) > 1:
            diff_data[key] = values

    diff_df = pd.DataFrame(diff_data)
    print(yaml_name)
    print(diff_df)
    print()

