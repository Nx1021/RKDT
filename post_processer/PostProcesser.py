from . import SCRIPT_DIR
from posture_6d.core.posture import Posture, Rotation
from posture_6d.derive import PnPSolver
from posture_6d.metric import MetricCalculator
from posture_6d.data.mesh_manager import MeshManager
import numpy as np
import numpy.typing as npt
import cv2
import matplotlib.pyplot as plt
import math
import torch
import os
import time
from torchvision.ops import generalized_box_iou

from models.results import LandmarkDetectionResult, ImagePosture, ObjPosture, denormalize_bbox, normalize_bbox
from models.utils import tensor_to_numpy, normalize_points, denormalize_points
from utils.yaml import load_yaml
from scipy.optimize import linear_sum_assignment

from MyLib.posture_6d.derive import draw_one_mask

from typing import TypedDict, Union, Callable, Optional, Iterable, TypeVar, overload, Any, Sequence, Literal

def create_model_manager(cfg_file) -> MeshManager:
    cfg_paras = load_yaml(cfg_file)
    model_manager = MeshManager(os.path.join(SCRIPT_DIR, cfg_paras["models_dir"]),
                                 cfg_paras["pcd_models"])
    return model_manager

def create_pnpsolver(cfg_file) -> PnPSolver:
    cfg_paras = load_yaml(cfg_file)
    default_K = os.path.join(SCRIPT_DIR, cfg_paras["models_dir"], cfg_paras["default_K"])
    return PnPSolver(default_K)

class TracePoint():
    STATE_UNKNOWN   = -1
    STATE_STILL     = 0
    STATE_MOVE      = 1
    STATE_MUTANT    = 2
    STATE_LOST      = 3

    LINE_V_SHRESHOLD = 8.0 #mm/s
    ANG_V_SHRESHOLD = 12 / 180 * np.pi #rad/s
    LDMK_THRESHOLD = 2 / 24
    BBOX_THRESHOLD = 3.0

    LDMKS = {}

    def __init__(self, timecode:np.uint32, trace:"ObjectTrace") -> None:
        self.trace = trace
        # rank 0
        self.__bbox:Union[None, np.ndarray] = None          # [6,]
        self.__posture:Union[Posture, None] = None

        # rank 1
        self.__v_calced = False
        self.__bbox_v:np.ndarray = np.zeros((6,), dtype=np.float32)
        self.__linv_v:np.ndarray = np.zeros((3,), dtype=np.float32)
        self.__ang_v:np.ndarray = np.zeros((3,), dtype=np.float32)

        # rank 2
        self.__a_calced = False
        self.__bbox_a:np.ndarray = np.zeros((6,), dtype=np.float32)
        self.__linv_a:np.ndarray = np.zeros((3,), dtype=np.float32)
        self.__ang_a:np.ndarray = np.zeros((3,), dtype=np.float32)

        self.__time_code:np.uint32 = timecode

        self.__state:int = TracePoint.STATE_UNKNOWN

    def set_x(self, bbox:Optional[np.ndarray], posture:Optional[Posture]):
        '''
        Set the bounding box and posture of the trace point.

        Args:
            bbox (Optional[np.ndarray]): The bounding box of the trace point.  [x1, y1, x2, y2]
            posture (Optional[Posture]): The posture of the trace point.
        '''
        #[x1, y1, x2, y2, w, h]
        if bbox is not None:
            self.__bbox = np.concatenate([bbox, bbox[2:] - bbox[:2]]) # type:ignore
        else:
            self.__bbox = None
        self.__posture = posture

        if bbox is None or posture is None:
            self.__state:int = TracePoint.STATE_LOST
        else:
            self.__state:int = TracePoint.STATE_UNKNOWN

    @property
    def lost(self):
        return self.__state == TracePoint.STATE_LOST
    
    @property
    def valid(self):
        return self.__state == TracePoint.STATE_STILL or self.__state == TracePoint.STATE_MOVE # or self.__state == TracePoint.STATE_UNKNOWN

    @property
    def calculable(self):
        return self.__state == TracePoint.STATE_UNKNOWN or self.__state == TracePoint.STATE_STILL or self.__state == TracePoint.STATE_MOVE

    @property
    def visible(self):
        return self.__state != TracePoint.STATE_LOST

    @property
    def v_calced(self):
        return self.__v_calced
    
    @property
    def a_calced(self):
        return self.__a_calced

    @property
    def timecode(self):
        return self.__time_code
    
    @property
    def state(self):
        return self.__state

    @state.setter
    def state(self, state:int):
        self.__state = state

    @property
    def has_data(self):
        return self.__bbox is not None and self.__posture is not None

    @property
    def bbox(self):
        return self.__bbox
    
    @property
    def posture(self):
        return self.__posture

    @property
    def line_v_value(self):
        return np.linalg.norm(self.__linv_v)
    
    @property
    def ang_v_value(self):
        return np.linalg.norm(self.__ang_v)

    @property
    def line_a_value(self):
        return np.linalg.norm(self.__linv_a)
    
    @property
    def ang_a_value(self):
        return np.linalg.norm(self.__ang_a)

    @property
    def bbox_v(self):
        return self.__bbox_v
    
    @property
    def bbox_a(self):
        return self.__bbox_a

    def get_x(self):
        return self.__bbox, self.__posture
    
    def get_v(self):
        return self.__bbox_v, self.__linv_v, self.__ang_v
    
    def get_a(self):
        return self.__bbox_a, self.__linv_a, self.__ang_a

    def set_v(self, bbox_v:np.ndarray, line_v:np.ndarray, ang_v:np.ndarray):
        self.__bbox_v[:] = bbox_v
        self.__linv_v[:] = line_v
        self.__ang_v[:] = ang_v

    def set_a(self, bbox_a:np.ndarray, linv_a:np.ndarray, ang_a:np.ndarray):
        self.__bbox_a[:] = bbox_a
        self.__linv_a[:] = linv_a
        self.__ang_a[:] = ang_a

    @overload
    def calc_v(self, other_trace_point:"TracePoint", del_time:float, forward:bool) -> bool: 
        ...
    
    @overload
    def calc_v(self, trace_point_last:"TracePoint", trace_point_next:"TracePoint", del_time:float) -> bool:
        ...

    def calc_v(self, arg1:Any, arg2:Any, arg3:Any): # type:ignore
        if not isinstance(arg2, TracePoint):
            other_trace_point:"TracePoint" = arg1
            del_time:float = arg2
            forward:bool = arg3
            # use 3 point difference
            assert isinstance(other_trace_point, TracePoint), "trace_point_2 must be TracePoint"
            if other_trace_point.lost or self.lost:
                return False
            del_time = abs(del_time) if forward else -abs(del_time)

            # calc v
            del_bbox:np.ndarray = other_trace_point.__bbox - self.__bbox # type:ignore
            del_posture:Posture = other_trace_point.__posture * self.__posture.inv() # type:ignore
            del_tvec = del_posture.tvec
            del_rvec = del_posture.rvec

            bbox_v = del_bbox / del_time
            line_v = del_tvec / del_time
            ang_v  = del_rvec / del_time

            self.set_v(bbox_v, line_v, ang_v)
            self.__v_calced = True
            self.__state = self.decide_state_by_v()
            return True
        else:
            trace_point_last:"TracePoint" = arg1
            trace_point_next:"TracePoint" = arg2
            if not (trace_point_last.has_data and trace_point_next.has_data):
                return False
            del_time:float = arg3
            # use 3 point difference
            assert isinstance(trace_point_next, TracePoint), "trace_point_0 must be TracePoint"
            assert isinstance(trace_point_last, TracePoint), "trace_point_2 must be TracePoint"
            if trace_point_last.lost or trace_point_next.lost:
                return False
            del_time = abs(del_time)

            # calc v
            del_bbox:np.ndarray = trace_point_next.__bbox - trace_point_last.__bbox # type:ignore
            del_rmat = trace_point_next.__posture.rmat.dot(np.linalg.inv(trace_point_last.__posture.rmat)) # type:ignore
            del_rvec = Posture(rmat = del_rmat).rvec
            del_tvec = trace_point_next.__posture.tvec - trace_point_last.__posture.tvec # type:ignore
            # del_posture:Posture = trace_point_next.__posture * trace_point_last.__posture.inv() # type:ignore
            # del_tvec = del_posture.tvec
            # del_rvec = del_posture.rvec

            bbox_v = del_bbox / del_time
            line_v = del_tvec / del_time
            ang_v  = del_rvec / del_time

            self.set_v(bbox_v, line_v, ang_v)
            self.__v_calced = True
            self.__state = self.decide_state_by_v()
            return True
    
    @overload
    def calc_a(self, other_trace_point:"TracePoint", del_time:float, forward:bool = True) -> bool: # type:ignore
        ...

    @overload
    def calc_a(self, trace_point_last:"TracePoint", trace_point_next:"TracePoint", del_time:float):
        ...

    def calc_a(self, arg1:Any, arg2:Any, arg3:Any): # type:ignore
        if not isinstance(arg2, TracePoint):
            other_trace_point:"TracePoint" = arg1
            del_time:float = arg2
            forward:bool = arg3
            # use forward / backward point difference
            assert isinstance(other_trace_point, TracePoint), "trace_point_2 must be TracePoint"
            if not (self.v_calced and other_trace_point.v_calced):
                return False
            del_time = abs(del_time) if forward else -abs(del_time)
            
            # calc a
            del_bbox_v = other_trace_point.__bbox_v - self.__bbox_v
            del_linv_v = other_trace_point.__linv_v - self.__linv_v
            del_ang_v  = other_trace_point.__ang_v  - self.__ang_v

            bbox_a = del_bbox_v / del_time
            linv_a = del_linv_v / del_time
            ang_a  = del_ang_v  / del_time

            self.set_a(bbox_a, linv_a, ang_a)
            self.__a_calced = True
            return True
        else:
            trace_point_last:"TracePoint" = arg1
            trace_point_next:"TracePoint" = arg2
            del_time:float = arg3
            # use 3 point difference
            assert isinstance(trace_point_last, TracePoint), "trace_point_2 must be TracePoint"
            assert isinstance(trace_point_next, TracePoint), "trace_point_0 must be TracePoint"
            if trace_point_last.v_calced or trace_point_next.v_calced:
                return False
            del_time = abs(del_time)

            # calc v
            del_bbox_v = trace_point_next.__bbox_v - trace_point_last.__bbox_v
            del_linv_v = trace_point_next.__linv_v - trace_point_last.__linv_v
            del_ang_v  = trace_point_next.__ang_v  - trace_point_last.__ang_v

            bbox_a = del_bbox_v / del_time
            linv_a = del_linv_v / del_time
            ang_a  = del_ang_v  / del_time

            self.set_a(bbox_a, linv_a, ang_a)
            self.__a_calced = True
            return True

    def decide_state_by_v(self) -> int:
        if self.v_calced == False:
            return TracePoint.STATE_UNKNOWN
        if self.posture is None or self.bbox is None:
            return TracePoint.STATE_LOST
        line_v = self.line_v_value
        ang_v = self.ang_v_value
        if line_v < TracePoint.LINE_V_SHRESHOLD and ang_v < TracePoint.ANG_V_SHRESHOLD:
            return TracePoint.STATE_STILL
        else:
            # 判据1：通过关键点的重投影是否在bbox之外来判断
            ldmk_3d = self.trace.ldmk_3d
            ldmk_2d = ObjectTraceManager.pnpsolver.calc_reproj(ldmk_3d, posture=self.posture) # [landmark_num, 2]
            ldmk_num = ldmk_2d.shape[0]
            lt = self.bbox[:2] - 2
            rb = self.bbox[2:4] + 2
            out = np.any(np.logical_or(ldmk_2d < lt, ldmk_2d > rb), axis=-1) # [landmark_num]
            rlt1 = np.sum(out) > (ldmk_num * self.LDMK_THRESHOLD)
            # if np.sum(out) > (ldmk_num * self.LDMK_THRESHOLD):
            #     return TracePoint.STATE_MUTANT
            # else:
            #     return TracePoint.STATE_MOVE
            # # 判据2：bbox的变化是否超过阈值来判断
            # bbox_v = self.bbox_v
            # bs = self.BBOX_THRESHOLD
            # x1, y1, x2, y2, w, h = bbox_v
            # if (((w < -2*bs) ^ (h < -2*bs)) and np.sum(np.abs(bbox_v[:4]) <= bs)==3) or \
            #     ((w < -2*bs and h < -2*bs)  and np.sum(np.abs(bbox_v[:4]) <= bs)==2):
            #     rlt2 = True
            # else:
            #     rlt2 = False
            rlt2 = line_v > 3 * TracePoint.LINE_V_SHRESHOLD and ang_v > 3 * TracePoint.ANG_V_SHRESHOLD

            if rlt1 and rlt2:
                return TracePoint.STATE_MUTANT
            else:
                return TracePoint.STATE_MOVE



    # def __sub__(self, other:'TracePoint'):
    #     assert isinstance(other, TracePoint), "other must be TracePoint"
    #     assert self.is_point and  other.is_point, "both must be point"
    #     if not self.visible or not other.visible:
    #         visible = False
    #         return TracePoint(None, None, None, False, False)
    #     else:
    #         # 求解两个轨迹点的差值
    #         del_center = self.center - other.center
    #         del_posture = self.posture * other.posture.inv()
    #         del_time = self.time - other.time
    #         return TracePoint(del_center, del_posture, del_time, True, False)
    
    # @staticmethod
    # def calcSpeed(p1:"TracePoint", p2:"TracePoint"):
    #     '''
    #     Returns:
    #     speed_line (float): The linear speed of the end effector. mm/s
    #     speed_angle (float): The angular speed of the end effector. rad/s
    #     '''
    #     assert isinstance(p1, TracePoint), "p1 must be TracePoint"
    #     assert isinstance(p2, TracePoint), "p2 must be TracePoint"
    #     if not p1.visible or not p2.visible:
    #         return None, None
    #     else:
    #         # 求解两个轨迹点的差值
    #         del_posture = p1.posture * p2.posture.inv()
    #         del_time = p1.time - p2.time

    #         speed_line = (np.linalg.norm(del_posture.tvec)) / del_time
    #         speed_angle = (np.linalg.norm(del_posture.rvec)) / del_time
    #         return speed_line, speed_angle

class ObjectTrace():

    STABLE_COUNT = 5

    LDMK_SEQ_LEN = 5
    VISIB_THRESHOLD = 2

    CALC_Derivative_Mode = 0 # 0: center difference, 1: front difference, 2: back difference

    def __init__(self, manager:"ObjectTraceManager", class_id:int) -> None:
        self.trace:list[TracePoint] = []
        self.cur_state:int = TracePoint.STATE_MOVE
        self.last_stable_timecode:np.uint32 = manager.this_timecode
        self.last_valid_timecode:np.uint32 = manager.this_timecode
        
        self.stable_count = 0

        self.visible_count = 0
        self.ldmk_sequence:list[tuple[np.ndarray, np.ndarray]] = []
        self.__ldmk_stable = False
        self.stable_point:Union[np.ndarray, None] = None
        self.stable_mask:Union[np.ndarray, None] = None
        self._TEST_last_mask = None

        self.class_id:int = class_id

        self.query_time:Callable[[np.uint32], float] = manager.query_time

    @property
    def stable(self):
        return self.cur_state == TracePoint.STATE_STILL
    
    def get_stable_score(self):
        length = 30
        return len([tp for tp in self.trace[-length:] if tp.state == TracePoint.STATE_STILL]) / length

    @property
    def move(self):
        return self.cur_state == TracePoint.STATE_MOVE
    
    @property
    def mutant(self):
        return self.cur_state == TracePoint.STATE_MUTANT
    
    @property
    def lost(self):
        return self.cur_state == TracePoint.STATE_LOST
    
    @property
    def valid(self):
        return self.cur_state == TracePoint.STATE_STILL or self.cur_state == TracePoint.STATE_MOVE
    
    @property
    def visible(self):
        return self.cur_state != TracePoint.STATE_LOST

    @property
    def ldmk_3d(self):
        return ObjectTraceManager.ldmk_3d[self.class_id]

    @overload
    def qurey_trace_point(self, timecode:Union[np.uint32, int]) -> TracePoint: # type:ignore
        timecode = np.uint32(timecode)
        first_timecode = self.trace[0].timecode
        idx = int(timecode - first_timecode) # overflow is ok
        return self.trace[idx]

    @overload
    def qurey_trace_point(self, timecode:None) -> None:
        return None

    def qurey_trace_point(self, timecode):
        if timecode is None:
            return None
        else:
            timecode = np.uint32(timecode)
            first_timecode = np.uint32(self.trace[0].timecode)
            idx = int(timecode - first_timecode) # overflow is ok
            if idx < 0 or idx >= len(self.trace):
                return None
            return self.trace[idx]

    def __search_not_empty_timecode(self, timecode, _off_list:Iterable[int], forward_direction:Optional[bool] = None):
        first_timecode = self.first_timecode
        def search_one(forward_direction:Optional[bool] = None):
            if _off == 0:
                assert forward_direction is not None
                forward_direction = bool(forward_direction)
                step_target = 1 if forward_direction else -1
            else:
                forward_direction = (_off > 0)
                step_target = _off

            step = 1 if forward_direction else -1

            cur_idx:int = timecode - first_timecode + (step if _off != 0 else 0) # from where to start
            ok_count = 0

            while cur_idx >= 0 and cur_idx < len(self.trace):
                trace_point = self.trace[cur_idx]
                if not trace_point.lost:
                    ok_count += (1 if _off >= 0 else -1)
                cur_idx += (1 if _off >= 0 else -1)
                if (ok_count != step_target):
                    yield np.uint32(first_timecode + cur_idx)
            yield None
        rlt_timecode_g = search_one(forward_direction)
        rlt:list[Union[np.uint32, None]] = []
        for _off in _off_list:
            try:
                rlt.append(next(rlt_timecode_g)) # type:ignore
            except StopIteration:
                rlt.append(None) # type:ignore
        return rlt

    @overload
    def search_not_empty_timecode(self, timecode:np.uint32, offset:Sequence[int], forward_direction:Optional[bool] = None) -> list[Union[np.uint32, None]]: # type:ignore
        ...
        
    @overload
    def search_not_empty_timecode(self, timecode:np.uint32, offset:int, forward_direction:Optional[bool] = None) -> Union[np.uint32, None]:
        ...
    
    def search_not_empty_timecode(self, timecode:np.uint32, offset, forward_direction:Optional[bool] = None):
        if isinstance(offset, int):
            return self.__search_not_empty_timecode(timecode, [offset], forward_direction)[0]
        elif isinstance(offset, Sequence):
            offset_list:npt.NDArray[np.int32] = np.unique(offset).astype(np.int32)
            offset_list.sort()
            minus_idx = [x for x in range(len(offset_list)) if offset_list[x] < 0]
            non_minus_idx = [x for x in range(len(offset_list)) if offset_list[x] >= 0]

            rlt_timecode:list[Union[np.uint32, None]] = []
            for indices in [minus_idx, non_minus_idx]:
                _off_list:Sequence[int] = offset_list[indices] # type:ignore
                rlt_timecode += self.__search_not_empty_timecode(timecode, _off_list, forward_direction)
            return rlt_timecode # type:ignore
    
    def _search_timecode(self, condition:Callable[[TracePoint], bool], timecode:np.uint32, offset:int, forward_direction:Optional[bool] = None):
        first_timecode = self.first_timecode
        if offset == 0:
            assert forward_direction is not None
            forward_direction = bool(forward_direction)
            step_target = 1 if forward_direction else -1
        else:
            forward_direction = (offset > 0)
            step_target = offset

        step = 1 if forward_direction else -1

        cur_idx:int = timecode - first_timecode + (step if offset != 0 else 0) # from where to start
        ok_count = 0

        while cur_idx >= 0 and cur_idx < len(self.trace):
            trace_point = self.trace[cur_idx]
            if condition(trace_point):
                ok_count += step
            if (ok_count == step_target):
                return np.uint32(first_timecode + cur_idx)
            cur_idx += step
        return None

    def search_valid_timecode(self, timecode:np.uint32, offset:int, forward_direction:Optional[bool] = None) -> Union[np.uint32, None]:
        return self._search_timecode(lambda x: x.valid, timecode, offset, forward_direction)
    
    def search_calculable_timecode(self, timecode:np.uint32, offset:int, forward_direction:Optional[bool] = None) -> Union[np.uint32, None]:
        return self._search_timecode(lambda x: x.calculable, timecode, offset, forward_direction)

    @property
    def first_timecode(self):
        return self.trace[0].timecode
    
    @property
    def last_timecode(self):
        return self.trace[-1].timecode

    @property
    def last_stable_trace_point(self):
        return self.qurey_trace_point(self.last_stable_timecode)
    
    @property
    def last_valid_trace_point(self):
        return self.qurey_trace_point(self.last_valid_timecode)

    @property
    def last_ref_trace_point(self):
        if self.valid:
            return self.last_valid_trace_point
        else:
            return self.last_stable_trace_point

    @property
    def last_1st_trace_point(self):
        return self.trace[-1]
    
    @property
    def last_2nd_trace_point(self):
        return self.trace[-2]

    @property
    def last_3rd_trace_point(self):
        return self.trace[-3]
    
    # @property
    # def last_valid_trace_point(self):
    #     tc = self.search_valid_timecode(self.last_timecode, 0, False)
    #     return self.qurey_trace_point( tc)
    
    @property
    def last_v_calced_trace_point(self):
        tc = self.search_calculable_timecode(self.last_timecode, 0, False)
        if tc is None:
            return None
        if self.CALC_Derivative_Mode == 0 or self.CALC_Derivative_Mode == 1:
            return self.qurey_trace_point(np.uint32(tc - 1))
        else:
            return self.qurey_trace_point( tc)

    @property
    def last_a_calced_trace_point(self):
        tc = self.search_calculable_timecode(self.last_timecode, 0, False)
        if tc is None:
            return None
        if self.CALC_Derivative_Mode == 0 or self.CALC_Derivative_Mode == 1:
            return self.qurey_trace_point(np.uint32(tc - 2))
        else:
            return self.qurey_trace_point( tc)

    def add_to_ldmk_sequence(self, ldmks:np.ndarray, mask:np.ndarray):
        if len(self.ldmk_sequence) == self.LDMK_SEQ_LEN:
            self.ldmk_sequence.pop(0)
        self.ldmk_sequence.append((ldmks, mask))

    def calc_ldmk_sequence(self):
        '''
        计算序列中的物体的平均姿态
        '''
        if len(self.ldmk_sequence) == 0:
            return None, None
        if len(self.ldmk_sequence) < 2:
            return self.ldmk_sequence[0]
        # 计算landmarks，对应位置计算去除离群值之后的均值
        ldmks_seq = np.stack([x[0] for x in self.ldmk_sequence], axis=0) # [N, landmark_num, 2]
        ldmks_mean = np.mean(ldmks_seq, axis=0) # [landmark_num, 2]
        ldmks_std = np.std(ldmks_seq, axis=0) # [landmark_num, 2]
        outliers = np.any(np.abs(ldmks_seq - ldmks_mean) > 2 * ldmks_std, axis = -1) # [N, landmark_num] # 如果某项距离均值过远，认为序列中的该项是离群值
        outliers = outliers + np.any(ldmks_std > 3.0, axis=-1) # [N, landmark_num] + [landmark_num] 如果标准差过大，排除该点的整个序列
        ldmks_seq_wo_outliers = ldmks_seq.copy()
        ldmks_seq_wo_outliers[outliers] = 0.0
        notoutliers_count = np.tile(np.expand_dims(np.sum(~outliers, axis=0), -1), (1,2)) # [landmark_num, 2]
        ldmks_mean_wo_outliers = np.sum(ldmks_seq_wo_outliers, axis=0) / notoutliers_count # [landmark_num, 2]
        
        mask_seq:np.ndarray = np.stack([x[1] for x in self.ldmk_sequence], axis=0) # [N, landmark_num]
        mask = np.sum(mask_seq, axis=0) == len(mask_seq) # [landmark_num]
        mask = mask * (~np.any(np.isnan(ldmks_mean_wo_outliers), axis =-1)) # [landmark_num] 如果均值中有nan，认为该点不可见
        
        if len(self.ldmk_sequence) == self.LDMK_SEQ_LEN:
            if not self.__ldmk_stable:
                stable = np.all(mask_seq == mask_seq[0], axis=0) # [landmark_num] 如果序列中的所有mask都相同，认为该点稳定
                if np.sum(stable) > 18:
                    self.__ldmk_stable = True
                    self.stable_point = stable
            else:
                stable = np.all(mask_seq == self.stable_mask, axis=0) # [landmark_num] 如果序列中的所有mask都相同，认为该点稳定
                if np.sum(stable) <= 18:
                    self.__ldmk_stable = False
                    self.stable_point = None
                else:
                    self.stable_point = stable * self.stable_point
        
        if self.__ldmk_stable and self.stable_point is not None:
            # 不稳定的点都变为false
            mask[~self.stable_point] = False
            self.stable_mask  = mask

        if self._TEST_last_mask is not None and np.any(mask != self._TEST_last_mask):
            pass # print("!")
        self._TEST_last_mask = mask
        # self.sequence[-1] = (self.sequence[0], mask)

        return ldmks_mean_wo_outliers, mask

    def if_use_ldmk_seq(self, bbox:np.ndarray):
        if len(self) > 10:
            last_bbox = self.last_2nd_trace_point.bbox
            if last_bbox is None:
                return False
            giou = generalized_box_iou(torch.Tensor(np.expand_dims(bbox, 0)), torch.Tensor(np.expand_dims(last_bbox[:4], 0))).numpy()[0, 0]
            return giou > 0.975

    def add_trace_point(self, timecode:np.uint32):
        # append x
        self.trace.append(TracePoint(timecode, self))

    def set_last_trace_point(self, bbox:Optional[np.ndarray], posture:Optional[Posture]):
        if isinstance(bbox, torch.Tensor):
            bbox = tensor_to_numpy(bbox)
        self.last_1st_trace_point.set_x(bbox, posture)
        self.update_state()

    def calc_trace_point_derivative(self, timecode:Union[np.uint32, TracePoint], mode:Literal['a', 'v'] = 'v', 
                                    _timecode_dst:Optional[np.uint32] = None, _timecode_src:Optional[np.uint32] = None) -> bool:
        if isinstance(timecode, np.integer):
            tracepoint = self.qurey_trace_point(timecode)
        else:
            tracepoint = timecode
            timecode = tracepoint.timecode

        if mode == 'v':
            calc_func = tracepoint.calc_v
        elif mode == 'a':
            calc_func = tracepoint.calc_a
        else:
            raise ValueError("mode must be 'a' or 'v'")
        
        if self.CALC_Derivative_Mode == 0:
            # center difference
            timecode_dst = self.search_calculable_timecode(timecode, 1) if _timecode_dst is None else _timecode_dst
            if self.mutant:
                timecode_src = self.last_stable_timecode
            else:
                timecode_src = self.search_calculable_timecode(timecode, -1) if _timecode_src is None else _timecode_src
            if timecode_dst is None or timecode_src is None:
                return False
            del_time:float = float((self.query_time(timecode_dst) - self.query_time(timecode_src)) / (timecode_dst - timecode_src))
            return calc_func(self.qurey_trace_point(timecode_src), self.qurey_trace_point(timecode_dst), del_time)
        elif self.CALC_Derivative_Mode == 1:
            # forward difference
            timecode_dst = self.search_calculable_timecode(timecode, 1) if _timecode_dst is None else _timecode_dst
            if timecode_dst is None:
                return False
            del_time:float = float(self.query_time(timecode_dst) - self.query_time(timecode))
            return calc_func(self.qurey_trace_point(timecode_dst), del_time, True)
        elif self.CALC_Derivative_Mode == 2:
            # back difference
            timecode_src = self.search_calculable_timecode(timecode, -1) if _timecode_src is None else _timecode_src
            if timecode_src is None:
                return False
            del_time:float = float(self.query_time(timecode) - self.query_time(timecode_src))
            return calc_func(self.qurey_trace_point(timecode_src), del_time, False)
        else:
            raise ValueError("CALC_Derivative_Mode must be 0, 1 or 2")

    def update_state(self):
        '''
            State-transition

            Still -> Move -> Mutant -> Lost\n
                <------        |        |\n
                       <--------------        |\n
                       <-----------------\n
        '''
        if self.last_1st_trace_point.lost:
            self.cur_state = TracePoint.STATE_LOST # -> LOST
            return 
        elif self.last_v_calced_trace_point is None:
            return
        else:
            last_v_calced_trace_point = self.last_v_calced_trace_point
            success = self.calc_trace_point_derivative(last_v_calced_trace_point, 'v', _timecode_dst=self.last_timecode)
            cvt_permitted = False
            if success:
                # -> MUTANT / MOVE / STILL
                new_state = last_v_calced_trace_point.state
                if self.lost:
                    cvt_permitted = new_state == TracePoint.STATE_STILL
                if self.mutant:
                    cvt_permitted = new_state == TracePoint.STATE_STILL 
                else:
                    cvt_permitted = True

                if new_state == TracePoint.STATE_STILL:
                    # only if continuous still state is long enough, it can be converted to still
                    if new_state == TracePoint.STATE_STILL:
                        self.stable_count += 1
                        self.stable_count = min(self.stable_count, self.STABLE_COUNT)
                    if self.stable_count == self.STABLE_COUNT:
                        cvt_permitted = cvt_permitted and True
                    else:
                        cvt_permitted = False
                else:
                    self.stable_count = 0

                if cvt_permitted:
                    self.cur_state = new_state
                    if self.cur_state == TracePoint.STATE_STILL:
                        self.last_stable_timecode = last_v_calced_trace_point.timecode
                        self.last_valid_timecode = last_v_calced_trace_point.timecode
                    elif self.cur_state == TracePoint.STATE_MOVE:
                        self.last_valid_timecode = last_v_calced_trace_point.timecode
            else:
                return # not cvt
            
    def __len__(self):
        return len(self.trace)

class ObjectTraceManager():
    TRACE_TIME_POINT_NUM = 10000 # seceonds
    ldmk_3d:dict[int, np.ndarray] = {}
    pnpsolver:PnPSolver = None # type:ignore

    INVALID_GIOU_THRESHOLD = 0.0

    def __init__(self) -> None:
        self.this_timecode:np.uint32 = np.uint32(-1)
        self.timecode_list:list[np.uint32] = []
        self.time_list:list[float] = []

        self.traces:dict[str, ObjectTrace] = {}
        self.trace_name_count = {}

        self.cur_match:dict[int, str] = {}

        self.update_time()

    def query_time(self, timecode:np.uint32) -> float:
        return self.time_list[timecode - self.timecode_list[0]]
    
    def query_trace(self, cur_idx):
        return self.traces[self.cur_match[cur_idx]]

    def match_to_trace(self, all_ldmk_rlt:list[LandmarkDetectionResult]):
        if len(all_ldmk_rlt) == 0:
            # if no obj detected, add empty tracepoint to all trace
            for trace in self.traces.values():
                trace.set_last_trace_point(None, None)
            self.cur_match.clear()
            return

        self.clear_lost_trace()

        all_class_ids = np.unique([x.class_id for x in all_ldmk_rlt])
        self.cur_match.clear()
        for class_id in all_class_ids:
            class_id:int #逐个class计算bbox之间的cost_matrix
            
            # traces = [x for x in self.traces.values() if x.class_id == class_id]
            same_class_detection_indices = [i for i, x in enumerate(all_ldmk_rlt) if x.class_id == class_id]
            same_class_trace_dict = {trace_name:trace for trace_name, trace in self.traces.items() if trace.class_id == class_id}

            ldmk_rlt = [x for x in all_ldmk_rlt if x.class_id == class_id]
            if len(same_class_trace_dict) == 0 and len(ldmk_rlt) == 0:
                continue
            elif len(same_class_trace_dict) == 0:
                unmatched_trace_indices = []
                unmatched_detection_indices = same_class_detection_indices
            elif len(ldmk_rlt) == 0:
                unmatched_trace_indices = np.arange(len(same_class_trace_dict)).tolist()
                unmatched_detection_indices = []
            else:
                # 计算bbox cost_matrix
                orig_bboxes = []
                for trace in same_class_trace_dict.values():
                    if trace.last_ref_trace_point.has_data:
                        orig_bboxes.append(trace.last_ref_trace_point.bbox[:4])
                    else:
                        orig_bboxes.append(np.array([0,0,0,0], np.float32))
                new_bboxes = [tensor_to_numpy(x.bbox) for x in ldmk_rlt if x.class_id == class_id]
                iou_cost_matrix:np.ndarray = generalized_box_iou(torch.Tensor(np.array(orig_bboxes)), torch.Tensor(np.array(new_bboxes))).numpy()

                to_pop_trace_idx:list[int] = []
                for ri, t in enumerate(same_class_trace_dict.values()):
                    if not t.valid and np.all(iou_cost_matrix[ri] < ObjectTraceManager.INVALID_GIOU_THRESHOLD):
                        to_pop_trace_idx.append(ri)
                reserved_trace_idx:list[int] = np.setdiff1d(np.arange(len(same_class_trace_dict)), to_pop_trace_idx).tolist()
                
                iou_cost_matrix = np.delete(iou_cost_matrix, to_pop_trace_idx, axis=0)

                # 计算时间间隔 cost_matrix
                timecode_cost_matrix = np.zeros((len(same_class_trace_dict) - len(to_pop_trace_idx), len(ldmk_rlt)), dtype=np.float32)
                if len(self.timecode_list) > 10:
                    for trace_i, trace in enumerate(same_class_trace_dict.values()):
                        if trace_i in to_pop_trace_idx:
                            continue
                        row_i = reserved_trace_idx.index(trace_i)
                        timecode_cost_matrix[row_i, :] = (1 - (((self.this_timecode - 1) - trace.last_valid_timecode) / len(self.timecode_list)) ** 2) * 0.5
                
                cost_matrix = iou_cost_matrix + timecode_cost_matrix
                mat_row_ind, mat_col_ind = linear_sum_assignment(cost_matrix, maximize=True)

                match_trace_indices = [reserved_trace_idx[m_row_i] for m_row_i in mat_row_ind]
                match_detection_indices = [same_class_detection_indices[m_col_i] for m_col_i in mat_col_ind]

                self.cur_match.update({detection_i:list(same_class_trace_dict.keys())[trace_i] 
                                       for trace_i, detection_i in zip(match_trace_indices, match_detection_indices)})

                unmatched_trace_indices:list[int] = np.setdiff1d(np.arange(len(same_class_trace_dict)), match_trace_indices).tolist() # lost objs
                # unmatched_trace_indices.extend(to_pop_trace_idx)
                unmatched_detection_indices:list[int] = np.setdiff1d(same_class_detection_indices, match_detection_indices).tolist() # new objs

            # add new trace for new obj
            for col_i in unmatched_detection_indices:
                name, _ = self.add_trace(class_id)
                # add to match
                self.cur_match[col_i] = name

            # add empty tracepoint
            for trace_i in unmatched_trace_indices:
                trace: ObjectTrace = list(same_class_trace_dict.values())[trace_i]
                trace.set_last_trace_point(None, None)

    def add_trace(self, class_id:int):
        trace = ObjectTrace(self, class_id)
        self.trace_name_count.setdefault(class_id, 0)
        name = f"obj_{str(class_id).rjust(2, '0')}.{str(self.trace_name_count[class_id]).rjust(3, '0')}"
        if name not in self.traces:
            self.traces[name] = trace
            trace.add_trace_point(self.this_timecode)
            self.trace_name_count[class_id] += 1
            self.trace_name_count[class_id] = self.trace_name_count[class_id] % 1000
            return name, trace
            
        raise ValueError("Too many objects")

    def update_time(self):
        np.seterr(over='ignore')
        self.this_timecode = np.uint32(1) + self.this_timecode
        np.seterr(over='warn')
        self.timecode_list.append(self.this_timecode)
        self.time_list.append(time.time())
        if len(self.time_list) > self.TRACE_TIME_POINT_NUM:
            self.timecode_list.pop(0)
            self.time_list.pop(0)

        for trace in self.traces.values():
            trace.add_trace_point(self.this_timecode)

    def clear_lost_trace(self):
        to_pop = []
        for name, trace in self.traces.items():
            if not trace.valid and (trace.query_time(trace.last_timecode) - trace.query_time(trace.last_stable_timecode)) > 2:
                # lost for 2 seconds
                to_pop.append(name)
        for name in to_pop:
            self.traces.pop(name)

    # def update_all_trace(self):
    #     for t in self.traces.values():
    #         t.update_state()

    def clear(self):
        self.traces.clear()
        self.cur_match.clear()
        self.time_list.clear()
        self.timecode_list.clear()
        self.this_timecode = np.uint32(0)

class ObjPredSequence():
    _sequence_length = 5
    visib_threshold = 2
    def __init__(self, class_id:int) -> None:
        self.class_id = class_id
        self.lasttime_center:np.ndarray = np.zeros((2,), dtype=np.float32)
        self.lasttime_posture:Posture = None
        self.lasttime_ldmks:np.ndarray = None
        self.lasttime_bbox:np.ndarray = None

        self.visible_count = 0
        self.sequence:list[tuple[np.ndarray, np.ndarray]] = []
        self.__ldmk_stable = False
        self.stable_point = None
        self.stable_mask = None

        self._TEST_last_mask = None

        self.__obj_stable = False
    
    @property
    def obj_stable(self):
        return self.__obj_stable

    def add_to_sequence(self, ldmks:np.ndarray, mask:np.ndarray):
        if len(self.sequence) == self._sequence_length:
            self.sequence.pop(0)
        self.sequence.append((ldmks, mask))

    def calc_sequence(self):
        '''
        计算序列中的物体的平均姿态
        '''
        if len(self.sequence) == 0:
            return None, None
        if len(self.sequence) < 2:
            return self.sequence[0]
        # 计算landmarks，对应位置计算去除离群值之后的均值
        ldmks_seq = np.stack([x[0] for x in self.sequence], axis=0) # [N, landmark_num, 2]
        ldmks_mean = np.mean(ldmks_seq, axis=0) # [landmark_num, 2]
        ldmks_std = np.std(ldmks_seq, axis=0) # [landmark_num, 2]
        outliers = np.any(np.abs(ldmks_seq - ldmks_mean) > 2 * ldmks_std, axis = -1) # [N, landmark_num] # 如果某项距离均值过远，认为序列中的该项是离群值
        outliers = outliers + np.any(ldmks_std > 3.0, axis=-1) # [N, landmark_num] + [landmark_num] 如果标准差过大，排除该点的整个序列
        ldmks_seq_wo_outliers = ldmks_seq.copy()
        ldmks_seq_wo_outliers[outliers] = 0.0
        notoutliers_count = np.tile(np.expand_dims(np.sum(~outliers, axis=0), -1), (1,2)) # [landmark_num, 2]
        ldmks_mean_wo_outliers = np.sum(ldmks_seq_wo_outliers, axis=0) / notoutliers_count # [landmark_num, 2]
        
        mask_seq = np.stack([x[1] for x in self.sequence], axis=0) # [N, landmark_num]
        mask = np.sum(mask_seq, axis=0) == len(mask_seq) # [landmark_num]
        mask = mask * (~np.any(np.isnan(ldmks_mean_wo_outliers), axis =-1)) # [landmark_num] 如果均值中有nan，认为该点不可见
        
        if len(self.sequence) == self._sequence_length:
            if not self.__ldmk_stable:
                stable = np.all(mask_seq == mask_seq[0], axis=0) # [landmark_num] 如果序列中的所有mask都相同，认为该点稳定
                if np.sum(stable) > 18:
                    self.__ldmk_stable = True
                    self.stable_point = stable
            else:
                stable = np.all(mask_seq == self.stable_mask, axis=0) # [landmark_num] 如果序列中的所有mask都相同，认为该点稳定
                if np.sum(stable) <= 18:
                    self.__ldmk_stable = False
                    self.stable_point = None
                else:
                    self.stable_point = stable * self.stable_point
        
        if self.__ldmk_stable:
            # 不稳定的点都变为false
            mask[~self.stable_point] = False
            self.stable_mask  = mask

        if self._TEST_last_mask is not None and np.any(mask != self._TEST_last_mask):
            print("!")
        self._TEST_last_mask = mask
        # self.sequence[-1] = (self.sequence[0], mask)

        return ldmks_mean_wo_outliers, mask
    
    def set_lasttime_datas(self, ldmks:np.ndarray, posture:Posture, bbox:np.ndarray):
        # decide whether the obj is stable
        if posture is None or self.lasttime_posture is None:
            self.__obj_stable = False
        else:
            d_tvec = np.linalg.norm(self.lasttime_posture.tvec - posture.tvec)

            new_rot = posture.rmat
            old_rot = self.lasttime_posture.rmat

            # angle error
            d_angle = np.arccos(np.clip((np.trace(np.dot(new_rot.T, old_rot)) - 1) / 2.0, -1, 1))
            
            if d_tvec > 8 or d_angle > np.pi / 180 * 10:
                self.__obj_stable = False
            else:
                self.__obj_stable = True

        self.lasttime_ldmks = ldmks
        self.lasttime_posture = posture
        self.lasttime_bbox = bbox

    def inc_visible(self):
        if self.visible_count < self.visib_threshold:
            self.visible_count += 1

    def dec_visible(self):
        if self.visible_count > -self.visib_threshold:
            self.visible_count -= 1

    @property
    def visible(self):
        return self.visible_count >= self.visib_threshold
    
    @property
    def invisible(self):
        return self.visible_count <= -self.visib_threshold


class PostProcesser():
    '''
    后处理

    所有的变量名，不带_batch的表示是单个关键点对应的数据，例如坐标、热图
    带有_batch的是一整组，形状为 [N, ..., kpnum, ...]
    '''
    def __init__(self, pnpsolver:PnPSolver, mesh_manager:MeshManager, out_bbox_threshold=0.2):
        '''
        parameters
        -----
        heatmap_field_batch:    np.ndarray 整组区域热图[N, HM_SIZE, HM_SIZE, KEYPOINTS_NUM]
        heatmap_coord_batch:    np.ndarray 整组偏移热图[N, HM_SIZE, HM_SIZE, KEYPOINTS_NUM, 4]
        class_ids_batch:        np.ndarray 整组偏移热图[N]
        bboxes_batch:           np.ndarray 整组包围框[N, 4]
        
        return
        -----
        Description of the return
        '''
        self.pnpsolver:PnPSolver        = pnpsolver
        self.mesh_manager:MeshManager   = mesh_manager
        self.ldmk_out_upper:float       = out_bbox_threshold

        self._sequence_mode = False # 
        self._sequence_more_accurate = False #
        self._camera_pose = None
        self._use_desktop_assumption = False
        self._use_bbox_area_assumption = True
        self._use_fix_z = True

        # self.objs:list[ObjPredSequence] = []
        self.trace_manager = ObjectTraceManager()
        ObjectTraceManager.ldmk_3d = mesh_manager.model_ldmk_3d
        ObjectTraceManager.pnpsolver = pnpsolver

        ### std posture ###
        self.std_posture:dict[int, Posture] = {}
        for class_id, mesh in self.mesh_manager.get_meta_dict().items():
            name = mesh.name
            path = os.path.join(self.mesh_manager.root, "std_posture", str(class_id).rjust(6, '0') + "_" + name + ".npy")
            if os.path.exists(path):
                self.std_posture[class_id] = Posture(homomat= np.load(path))
            else:
                self.std_posture[class_id] = Posture()

    def parse_exclusively(self, probs:torch.Tensor, coords:torch.Tensor):
        '''
        只选取得分最高的
        '''
        conf = 0.5
        probs = tensor_to_numpy(probs)[-1] #[tgt_num, landmark_num + 1]
        coords = tensor_to_numpy(coords)[-1] #[tgt_num, landmark_num + 1]
        landmark_num = probs.shape[-1] - 1
        # pred_class = np.argmax(probs, axis=-1) # [tgt_num]
        # 获取对应正例的tgt，用匈牙利算法匹配
        # positive = np.where(pred_class < landmark_num)[0]
        positive_probs = probs[:, :-1]
        row_ind, col_ind = linear_sum_assignment(positive_probs, maximize=True)
        # 过滤一次，最大小于conf的会被丢弃
        probs_filter = np.max(probs, axis = 0)[col_ind] > conf
        col_ind = col_ind[probs_filter]
        row_ind = row_ind[probs_filter]
        # 获取对应的坐标
        mask = np.zeros(landmark_num, dtype=np.bool_)
        ldmks = np.zeros((landmark_num, 2), dtype=np.float32)
        mask[col_ind] = True
        ldmks[col_ind] = coords[row_ind]
        return ldmks, mask

    def parse_by_voting(self, probs:torch.Tensor, coords:torch.Tensor):
        '''
        对于同一个landmark，选取多个tgt进行投票
        '''
        conf = 0.25
        probs = tensor_to_numpy(probs)[-1]    #[tgt_num, landmark_num + 1]
        coords = tensor_to_numpy(coords)[-1]         #[tgt_num, landmark_num + 1]
        landmark_num = probs.shape[-1] - 1

        mask = np.zeros(landmark_num, dtype=np.bool8)
        ldmks = np.zeros((landmark_num, 2), dtype=np.float32)
        for li in range(landmark_num):
            ok_i = np.where(probs[:, li] > conf)[0]
            if len(ok_i > 0):
                mask[li] = True
                ldmks[li, :] = np.sum(coords[ok_i].T * probs[ok_i, li], axis=-1) / np.sum(probs[ok_i, li])
        return ldmks, mask

    def obj_out_bbox(self, ldmks, bbox):
        in_bbox = ( (ldmks[..., 0] >= bbox[0]) &
                    (ldmks[..., 0] <= bbox[2]) &
                    (ldmks[..., 1] >= bbox[1]) &
                    (ldmks[..., 1] <= bbox[3]))
        return np.sum(in_bbox) < int((1 - self.ldmk_out_upper) * ldmks.shape[0])

    # 沿着物体中心与相机坐标系原点的连线移动物体
    @staticmethod
    def move_obj_by_optical_link(posture:Posture, move_dist) -> Posture:
        '''
        move_dist 为正数，表示沿着射线远离相机
        '''
        optical_link = posture.tvec
        optical_link = optical_link / np.linalg.norm(optical_link)# normalize
        new_t = posture.tvec + optical_link * move_dist
        return posture.__class__(rvec=posture.rvec, tvec=new_t)

    def desktop_assumption(self, O_in_C:Posture, C_in_B:Posture, points_in_O:np.ndarray):
        '''
        桌面假设，假设物体的最低点处于桌面上，重新计算物体的trans_vecs
        '''

        points_in_C = O_in_C * points_in_O
        points_in_B = C_in_B * points_in_C

        # 最低点的Z
        min_z = np.min(points_in_B[:, 2])

        cam_obj_link_in_C = O_in_C.tvec.copy()
        cam_obj_link_in_C = cam_obj_link_in_C / np.linalg.norm(cam_obj_link_in_C)
        cam_obj_link_in_B = C_in_B.rmat.dot(cam_obj_link_in_C)

        # 计算cam_obj_link_in_B和[0,0,-1]的夹角
        cos_theta = np.dot(cam_obj_link_in_B, np.array([0,0,-1]))
        move_dist = min_z / cos_theta # move_dist 与 min_z同号

        new_O_in_C = self.move_obj_by_optical_link(O_in_C, move_dist)

        return new_O_in_C
    
    def bbox_area_assumption(self, posture:Posture, class_id:int, bbox:np.ndarray):
        # 假设预测的物体的投影面积和真实的投影面积相同，重新计算物体的trans_vecs
        points = self.mesh_manager.get_model_pcd(class_id)
        reproj = self.pnpsolver.calc_reproj(points, posture = posture)#[N, 2]
        reproj_bbox = (np.min(reproj[:, 0]), np.min(reproj[:, 1]), np.max(reproj[:, 0]), np.max(reproj[:, 1]))
        reproj_bbox_area = (reproj_bbox[2] - reproj_bbox[0]) * (reproj_bbox[3] - reproj_bbox[1])
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        scale = np.sqrt(bbox_area / reproj_bbox_area)
        move_dist = -(1 - 1/scale) * posture.tvec[2]
        new_posture = self.move_obj_by_optical_link(posture, move_dist)
        return new_posture

    def fix_z_assumption(self, C_in_B:Posture, posture:Posture, class_id:int):
        Z_inS = np.array([0,0,1], np.float32)
        Z_inO = np.linalg.inv(self.std_posture[class_id].rmat).dot(Z_inS)
        R_C2O = posture.rmat

        dst = np.linalg.inv(C_in_B.rmat).dot(Z_inS)
        src = R_C2O.dot(Z_inO)
        transmit = Posture(rvec=Rotation.get_rvec_from_destination(dst, src))
        posture.set_rmat(transmit.rmat.dot(R_C2O))
        return posture

    def solve_posture(self, ldmks:np.ndarray, mask:np.ndarray, bbox:np.ndarray, intr_M:Optional[np.ndarray], class_id:int, depth:Optional[np.ndarray] = None):
        if np.sum(mask) < 8:
            return None
        elif self.obj_out_bbox(ldmks, bbox):
            return None
        else:
            points_3d = self.mesh_manager.get_ldmk_3d(class_id)
            posture:Posture = self.pnpsolver.solvepnp(ldmks, points_3d, mask, K=intr_M, return_posture=True) # type:ignore

            if depth is not None:
                posture = self.__refine_posture_by_depth(depth, bbox, class_id, posture)

            if self._use_bbox_area_assumption and depth is None:
                posture = self.bbox_area_assumption(posture, class_id, bbox)

            if self._use_fix_z and isinstance(self._camera_pose, Posture):
                posture = self.fix_z_assumption(self._camera_pose, posture, class_id)

            # 桌面假设
            if self._use_desktop_assumption and isinstance(self._camera_pose, Posture):
                ldmk_3d_inO = self.mesh_manager.get_ldmk_3d(class_id)
                posture = self.desktop_assumption(posture, self._camera_pose, ldmk_3d_inO)

            return posture

    def __refine_posture_by_depth(self, depth:np.ndarray, bbox:np.ndarray, class_id:int, raw_posture:Posture):
        self.depth_scale = 0.5 # mm

        points = self.mesh_manager.get_model_pcd(class_id)
        mesh_meta = self.mesh_manager.export_meta(class_id)
        points = points[np.linspace(0, len(points) - 1, 1000, dtype=np.int32)] # sample [N, 3]
        points_in_C = raw_posture * points # [N, 3]
        min_dist = np.min(points_in_C[:, 2])

        bbox = np.array([np.floor(bbox[0]), np.floor(bbox[1]), np.ceil(bbox[2]), np.ceil(bbox[3])]).astype(np.int32)
        ref_depth = depth[bbox[1]:bbox[3], bbox[0]:bbox[2]] * self.depth_scale # [h, w]

        self.pnpsolver.intr.cam_wid = depth.shape[1]
        self.pnpsolver.intr.cam_hgt = depth.shape[0]
        raw_depth, orig_proj = draw_one_mask(mesh_meta, raw_posture, self.pnpsolver.intr, tri_mode=False)
        raw_depth[raw_depth == self.pnpsolver.intr.max_depth] = 0
        raw_depth = raw_depth[bbox[1]:bbox[3], bbox[0]:bbox[2]] # [h, w]

        if raw_depth.size == 0:
            return raw_posture

        delta_depth = ref_depth - raw_depth
        valid_mask:np.ndarray = (ref_depth > 0) & (raw_depth > 0)
        valid_mask = cv2.morphologyEx(valid_mask.astype(np.uint8), cv2.MORPH_ERODE, np.ones((3,3), np.uint8)).astype(np.bool8)
        dists = delta_depth[valid_mask]

        # exclued outliers
        dists_mean = np.mean(dists)
        dists_std = np.std(dists)
        dists = dists[(dists > dists_mean - 2 * dists_std) & (dists < dists_mean + 2 * dists_std)]

        if len(dists) != 0:
            move_dist = np.mean(dists)
            new_posture = self.move_obj_by_optical_link(raw_posture, move_dist)
            return new_posture
        else:
            return raw_posture

    def process_one(self, pred_probs, pred_ldmks, class_id, bbox:torch.Tensor, mode = 'v', intr_M = None, *, obj_idx:Optional[int] = None, depth:Optional[np.ndarray] = None):
        if mode == "e":
            ldmks, mask = self.parse_exclusively(pred_probs, pred_ldmks) # 独占式
        elif mode == "v":
            ldmks, mask = self.parse_by_voting(pred_probs, pred_ldmks)   # 投票式
        else:
            raise ValueError     

        bbox:np.ndarray = tensor_to_numpy(bbox)

        if self._sequence_mode:
            # 查找trace的编号
            # 获取Trace对象
            assert obj_idx is not None
            trace = self.trace_manager.query_trace(obj_idx)
            trace.add_to_ldmk_sequence(ldmks, mask)
            if self._sequence_more_accurate:
                if trace.if_use_ldmk_seq(bbox): 
                    _ldmks, _mask = trace.calc_ldmk_sequence()
                    if np.sum(_mask) < 8:
                        pass
                    else:
                        ldmks, mask = _ldmks, _mask
            posture = self.solve_posture(ldmks, mask, bbox, intr_M, class_id, depth = depth) # type:ignore
            trace.set_last_trace_point(bbox, posture)
        else:
            posture = self.solve_posture(ldmks, mask, bbox, intr_M, class_id, depth = depth)
        # # 计算姿态
        # self.objs[obj_idx].add_to_sequence(ldmks, mask)
        # if self._sequence_mode and self._sequence_more_accurate and self.objs[obj_idx].obj_stable:
        #     ldmks, mask = self.objs[obj_idx].calc_sequence()
        return ldmks, posture
    
    def process(self, image_list:list[np.ndarray], ldmk_detection:list[list[LandmarkDetectionResult]], depths:Optional[list[np.ndarray]] = None, mode = "v"):
        image_posture_list:list[ImagePosture] = []
        depths:Union[list[np.ndarray], list[None]] = [None for _ in image_list] if depths is None else depths
        for bi, batch in enumerate(ldmk_detection):
            if self._sequence_mode:
                self.trace_manager.update_time()
                self.trace_manager.match_to_trace(batch)
            image_posture = ImagePosture(image_list[bi], depth = depths[bi])
            for pred_obj_idx, rlt in enumerate(batch):
                ldmks, posture = self.process_one(rlt.landmarks_probs, rlt.landmarks, rlt.class_id, rlt.bbox, mode, obj_idx = pred_obj_idx, depth = depths[bi])
                if not self._sequence_mode:
                    # 创建objp
                    objp = ObjPosture(ldmks, rlt.bbox_n, rlt.class_id, image_posture.image_size, posture)
                    image_posture.obj_list.append(objp) 

            if self._sequence_mode:
                for pred_obj_idx, trace_name in self.trace_manager.cur_match.items():
                    trace = self.trace_manager.traces[trace_name]
                    if trace.valid:
                        last_valid_trace_point = trace.last_valid_trace_point
                        if last_valid_trace_point is None: 
                            continue
                        objp = ObjPosture(batch[pred_obj_idx].landmarks, 
                                          batch[pred_obj_idx].bbox_n,
                                          batch[pred_obj_idx].class_id, 
                                          image_posture.image_size, 
                                          last_valid_trace_point.posture, 
                                          trace.stable,
                                          trace_name,
                                          stable_score = trace.get_stable_score()) # type:ignore
                        image_posture.obj_list.append(objp)

            image_posture_list.append(image_posture)
        return image_posture_list
    
    def set_sqeuence_mode(self, sequence_mode:bool, more_accurate:bool = False):
        self._sequence_mode = sequence_mode
        self._sequence_more_accurate = more_accurate
        if not sequence_mode:
            self.trace_manager.clear()
        if sequence_mode:
            self._use_bbox_area_assumption = False
    
    def set_camera_extrensic(self, camera_pose:Posture):
        self._camera_pose = camera_pose

    def set_use_desktop_assumption(self, use_desktop_assumption:bool = True):
        self._use_desktop_assumption = use_desktop_assumption



        