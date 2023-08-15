import os
import numpy as np
from .aruco_detector import ArucoDetector
from .capturing import Capturing, RsCamera
from .data_manager import DataRecorder, ModelManager
from .interact_icp import InteractIcp
from .pcd_creator import PcdCreator
from . import ARUCO_FLOOR, FRAMETYPE_DATA

class PipeLine():
    def __init__(self, dataset_name, sub_dir) -> None:
        self.dataset_name = dataset_name
        self.sub_dir = sub_dir
        self.directory = os.path.join(dataset_name, sub_dir)
        #
        self.model_manager = ModelManager(self.directory)
        #
        self.data_recorder = DataRecorder(self.directory)
        #
        if self.data_recorder.aruco_floor_json.all_exits:
            self.aruco_detector = ArucoDetector(self.data_recorder.aruco_floor_json.read()[0])
        elif self.data_recorder.aruco_floor_png.all_exits:
            image, long_side = self.data_recorder.aruco_floor_png.read()
            self.aruco_detector = ArucoDetector(image, long_side)
        else:
            raise ValueError("aruco_floor.json or (aruco_floor.png and aruco_floor_long_side.txt) must exist")
        #
        self.capturing = Capturing(self.data_recorder, self.aruco_detector)
        self.capturing.data_recorder.add_skip_seg(-1)
        #
        self.pcd_creator = PcdCreator(self.data_recorder, self.aruco_detector, self.model_manager)
        #
        # self.interact_icp = InteractIcp(self.data_recorder, self.model_manager)

    def capture_image(self):
        is_recording_model = True
        record_gate = True

        def callback_CALI(data_recorder:DataRecorder):
            if data_recorder.current_category_index != FRAMETYPE_DATA:
                return False
            else:
                return True

        def callback_DATA(data_recorder:DataRecorder):
            return False

        
        for mode, func in zip([RsCamera.MODE_CALI, RsCamera.MODE_DATA], [callback_CALI, callback_DATA]):
            self.capturing.rs_camera = RsCamera(mode, imu_calibration=self.data_recorder.imu_calibration.paths()[0])
            self.capturing.rs_camera.intr.save_as_json(os.path.join(self.directory, "intrinsics_" + str(mode) + ".json"))
            self.capturing.start(func)

    def register_pcd(self):
        pass

    def icp(self):
        pass    

    def plot_captured(self):
        pass

    def plot_dataset(self):
        pass