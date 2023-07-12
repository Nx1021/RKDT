import numpy as np

class CameraIntr():
    def __init__(self, intr_M, CAM_WID, CAM_HGT, DEPTH_SCALE, EPS = 1.0e-6, MAX_DEPTH = 4.0) -> None:
        self.CAM_WID,   self.CAM_HGT =  CAM_WID, CAM_HGT # 重投影到的深度图尺寸
        self.CAM_FX, self.CAM_FY, self.CAM_CX, self.CAM_CY = CameraIntr.parse_intr_matrix(intr_M)
        self.intr_M = intr_M
        self.EPS = EPS
        self.MAX_DEPTH = MAX_DEPTH
        self.DEPTH_SCALE = DEPTH_SCALE

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
        valid = np.bitwise_and(np.bitwise_and((pixels[:, 0] >= 0), (pixels[:, 0] < self.CAM_WID)),
                    np.bitwise_and((pixels[:, 1] >= 0), (pixels[:, 1] < self.CAM_HGT)))
        return pixels[valid]
    