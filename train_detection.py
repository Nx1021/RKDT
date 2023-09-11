from __init__ import CFG_DIR, SCRIPT_DIR, WEIGHTS_DIR
import launcher
from launcher.setup import setup
import torch
import platform
from utils.yaml import load_yaml, dump_yaml



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

if __name__ == '__main__':
    cfg_file = f"{CFG_DIR}/oldt_morrison_mix.yaml"
    # torch.cuda.set_device("cuda:0")
    setup_paras = load_yaml(cfg_file)["setup"]

    sys = platform.system()
    if sys == "Windows":
        batch_size = 2
        # model = torch.nn.DataParallel(model)
    elif sys == "Linux":
        batch_size = 32 # * torch.cuda.device_count()
        # model = torch.nn.DataParallel(model)
    setup_paras["ldt_branches"] = {}
    setup_paras["batch_size"] = batch_size
    setup_paras["sub_data_dir"] = f"morrison_mix/"
    setup("detection", **setup_paras)
        
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