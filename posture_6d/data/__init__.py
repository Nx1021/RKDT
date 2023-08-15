from ..utils import JsonIO, JSONDecodeError, \
    extract_doc, modify_class_id, get_meta_dict
from ..posture import Posture
from ..intr import CameraIntr
from ..derive import calc_bbox_3d_proj, calc_landmarks_proj, calc_masks