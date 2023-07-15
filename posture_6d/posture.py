import numpy as np
import cv2

class Posture:
    '''
    姿态类，接收各种类型的姿态的输入，并转换为矩阵
    '''
    POSTURE_VEC = 0
    POSTURE_MAT = 1
    POSTURE_HOMOMAT = 2
    POSTURE_EULARZYX = 3
    
    def __init__(self, *, rvec:np.ndarray = None, tvec:np.ndarray = None,
                                        rmat:np.ndarray = None,
                                        homomat:np.ndarray = None,
                                        EularZYX:np.ndarray = None
                                        ) -> None:
        self.trans_mat:np.ndarray  = np.eye(4)
        if rvec is not None:
            rvec = np.array(rvec, np.float32).squeeze()
            self.set_rvec(rvec)
        if rmat is not None:
            self.set_rmat(rmat)
        if tvec is not None:
            tvec = np.array(tvec, np.float32).squeeze()
            self.set_tvec(tvec)
        if homomat is not None:
            self.set_homomat(homomat)
        
    def __mul__(self, obj):
        if isinstance(obj, Posture):
            posture = Posture(homomat = self.trans_mat.dot(obj.trans_mat))
            return posture
        elif isinstance(obj, np.ndarray):
            # the shape of the array must be [N, 3] or [N, 4]
            if (len(obj.shape) == 1 and obj.size == 3 or obj.size == 4):
                pass
            elif (len(obj.shape) == 2 and obj.shape[1] == 3 or obj.shape[1] == 4):
                pass
            else:
                raise ValueError
            return (self.rmat.dot(obj[..., :3].T)).T + self.tvec
    
    def inv(self):
        inv_transmat = self.inv_transmat
        return Posture(homomat=inv_transmat)

    @property
    def inv_transmat(self) -> np.ndarray :
        return np.linalg.inv(self.trans_mat)
    
    @property
    def rvec(self) -> np.ndarray :
        return cv2.Rodrigues(self.trans_mat[:3,:3])[0][:,0]
    
    @property
    def tvec(self) -> np.ndarray :
        return self.trans_mat[:3,3].T
    
    @property
    def rmat(self) -> np.ndarray :
        return self.trans_mat[:3,:3]
    
    @property
    def eularZYX(self):
        pass
    
    def set_rvec(self, rvec):
        self.trans_mat[:3,:3] = cv2.Rodrigues(rvec)[0]

    def set_tvec(self, tvec):
        self.trans_mat[:3,3] = tvec

    def set_rmat(self, rmat):
        self.trans_mat[:3,:3] = rmat

    def set_homomat(self, homomat):
        self.trans_mat:np.ndarray = homomat.copy()

if __name__ == "__main__":
    Posture()