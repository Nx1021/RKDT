import numpy as np
from .utils import JsonIO

class CameraIntr():
    def __init__(self, intr_M, CAM_WID = 0, CAM_HGT= 0, DEPTH_SCALE = 0.0, EPS = 1.0e-6, MAX_DEPTH = 4000.0) -> None:
        if isinstance(intr_M, CameraIntr):
            self = intr_M
        elif isinstance(intr_M, np.ndarray):
            self.intr_M = intr_M  
            self.CAM_FX, self.CAM_FY, self.CAM_CX, self.CAM_CY = CameraIntr.parse_intr_matrix(intr_M)
            
            assert CAM_WID >= 0 and CAM_HGT >= 0, "CAM_WID or CAM_HGT is illegal"
            self.CAM_WID,   self.CAM_HGT =  CAM_WID, CAM_HGT # 重投影到的深度图尺寸

            assert DEPTH_SCALE >= 0, "DEPTH_SCALE is illegal"
            self.DEPTH_SCALE = DEPTH_SCALE

            assert EPS > 0, "EPS is illegal"
            self.EPS = EPS
            
            assert MAX_DEPTH > 0, "MAX_DEPTH is illegal"
            self.MAX_DEPTH = MAX_DEPTH
        else:
            raise ValueError("intr_M is illegal")

    @staticmethod
    def parse_intr_matrix(intr_M):
        '''
        return CAM_FX, CAM_FY, CAM_CX, CAM_C
        '''
        CAM_FX, CAM_FY, CAM_CX, CAM_CY = intr_M[0,0], intr_M[1,1], intr_M[0,2], intr_M[1,2]
        return CAM_FX, CAM_FY, CAM_CX, CAM_CY

    def __mul__(self, points:np.ndarray):
        '''
        points: [N, (x,y,z)]

        return
        ----
        pixels: [N, (x,y)]
        '''
        assert isinstance(points, np.ndarray), "points is not ndarray"
        assert len(points.shape) == 2, "points.shape must be [N, (x,y,z)]"
        assert points.shape[-1] == 3, "points.shape must be [N, (x,y,z)]"
        CAM_FX, CAM_FY, CAM_CX, CAM_CY = self.CAM_FX, self.CAM_FY, self.CAM_CX, self.CAM_CY
        z = points[:, 2]
        u = points[:, 0] * CAM_FX / z + CAM_CX
        v = points[:, 1] * CAM_FY / z + CAM_CY
        pixels = np.stack([u,v], axis=-1)
        valid = z > self.EPS
        return pixels[valid]
    
    def filter_in_view(self, pixels):
        '''
        pixels: ndarray [N, (x,y)]
        '''
        if self.CAM_WID == 0 or self.CAM_HGT == 0:
            raise ValueError("CAM_WID or CAM_HGT is not set")
        valid = np.bitwise_and(np.bitwise_and((pixels[:, 0] >= 0), (pixels[:, 0] < self.CAM_WID)),
                    np.bitwise_and((pixels[:, 1] >= 0), (pixels[:, 1] < self.CAM_HGT)))
        return pixels[valid]
    
    @staticmethod
    def from_json(path):
        dict_ = JsonIO.load_json(path)
        return CameraIntr(**dict_)

    def save_as_json(self, path):
        dict_ = {}
        self.dict_["intr_M"] = self.intr_M
        self.dict_["CAM_WID"] = self.CAM_WID
        self.dict_["CAM_HGT"] = self.CAM_HGT
        self.dict_["DEPTH_SCALE"] = self.DEPTH_SCALE
        self.dict_["EPS"] = self.EPS
        self.dict_["MAX_DEPTH"] = self.MAX_DEPTH
        JsonIO.dump_json(path, dict_)