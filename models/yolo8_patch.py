'''
修改Ultralytics库中部分函数的行为，
'''

import os
import platform

import torch

from ultralytics.yolo.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, RANK, __version__
from ultralytics.yolo.utils.checks import check_version
TORCH_2_0 = check_version(torch.__version__, minimum='2.0')
import shutil
import numpy as np
import ultralytics.yolo.engine.predictor as _predictor
import ultralytics.yolo.engine.trainer as _trainer
import ultralytics.yolo.engine.validator as _validator

from ultralytics.yolo.v8.detect.train import DetectionTrainer

def _select_device(device='', batch=0, newline=False, verbose=True):
    
    """Selects PyTorch Device. Options are device = None or 'cpu' or 0 or '0' or '0,1,2,3'."""
    s = f'Ultralytics YOLOv{__version__} 🚀 Python-{platform.python_version()} torch-{torch.__version__} '
    device = str(device).lower()
    for remove in 'cuda:', 'none', '(', ')', '[', ']', "'", ' ':
        device = device.replace(remove, '')  # to string, 'cuda:0' -> '0' and '(0, 1)' -> '0,1'
    cpu = device == 'cpu'
    mps = device == 'mps'  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        if device == 'cuda':
            device = '0'
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable - must be before assert is_available()
        if not (torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', ''))):
            LOGGER.info(s)
            install = 'See https://pytorch.org/get-started/locally/ for up-to-date torch install instructions if no ' \
                    'CUDA devices are seen by torch.\n' if torch.cuda.device_count() == 0 else ''
            raise ValueError(f"Invalid CUDA 'device={device}' requested."
                            f" Use 'device=cpu' or pass valid CUDA device(s) if available,"
                            f" i.e. 'device=0' or 'device=0,1,2,3' for Multi-GPU.\n"
                            f'\ntorch.cuda.is_available(): {torch.cuda.is_available()}'
                            f'\ntorch.cuda.device_count(): {torch.cuda.device_count()}'
                            f"\nos.environ['CUDA_VISIBLE_DEVICES']: {visible}\n"
                            f'{install}')

    if not cpu and not mps and torch.cuda.is_available():  # prefer GPU if available
        devices = device.split(',') if device else str(torch.cuda.current_device())  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch > 0 and batch % n != 0:  # check batch_size is divisible by device_count
            raise ValueError(f"'batch={batch}' must be a multiple of GPU count {n}. Try 'batch={batch // n * n}' or "
                            f"'batch={batch // n * n + n}', the nearest batch sizes evenly divisible by {n}.")
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
            arg = 'cuda:{}'.format(d)
    elif mps and getattr(torch, 'has_mps', False) and torch.backends.mps.is_available() and TORCH_2_0:
        # Prefer MPS if available
        s += 'MPS\n'
        arg = 'mps'
    else:  # revert to CPU
        s += 'CPU\n'
        arg = 'cpu'

    if verbose and RANK == -1:
        LOGGER.info(s if newline else s.rstrip())
    return torch.device(arg)

def get_MyTrainer(weights_copy_path, active_class_id = None):
    class MyBaseTrainer(DetectionTrainer):
        def final_eval(self):
            rlt = super().final_eval()
            shutil.copy(self.best, weights_copy_path)            
            return rlt
        
        def build_dataset(self, img_path, mode='train', batch=None):
            dataset = super().build_dataset(img_path, mode, batch)
            if active_class_id is None:
                return dataset
            else:
                assert isinstance(active_class_id, list)
                assert all([isinstance(x, int) for x in active_class_id])
                not_empty_indices = []
                for lb_i, lb in enumerate(dataset.labels):
                    org_cls:np.ndarray = lb["cls"]
                    org_cls_list:list = org_cls.reshape(-1).astype(np.int32).tolist()

                    keep_cls_id = np.array([x for x in org_cls_list if x in active_class_id], dtype=np.float32).reshape(-1, 1)
                    keep_cls_idx = [i for i, x in enumerate(org_cls_list) if x in active_class_id]

                    keep_bbox = np.array(lb["bboxes"][keep_cls_idx], dtype=np.float32)

                    lb["cls"] = keep_cls_id
                    lb["bboxes"] = keep_bbox

                    if len(keep_cls_id) > 0:
                        not_empty_indices.append(lb_i)
                
                dataset.im_files = [dataset.im_files[i] for i in not_empty_indices]
                dataset.im_hw = [dataset.im_hw[i] for i in not_empty_indices]
                dataset.im_hw0 = [dataset.im_hw0[i] for i in not_empty_indices]
                dataset.ims = [dataset.ims[i] for i in not_empty_indices]
                dataset.label_files = [dataset.label_files[i] for i in not_empty_indices]
                dataset.labels = [dataset.labels[i] for i in not_empty_indices]
                dataset.npy_files = [dataset.npy_files[i] for i in not_empty_indices]

                return dataset
    return MyBaseTrainer

def img2label_paths(img_paths):
    """Define label paths as a function of image paths."""
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]

print("_select_device", _select_device)
_predictor.select_device = _select_device
_trainer.select_device = _select_device
_validator.select_device = _select_device
