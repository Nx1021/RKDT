import numpy as np
import cv2
import platform
import time
import os

from posture_6d.core.utils import JsonIO

sys = platform.system()

def read_flag(path, trig=True):
    with open(path, "r") as f:
        flag = f.read()
    if trig == False:
        return not flag == "0"
    else:
        return flag == "1"

def write_flag(path, flag):
    with open(path, "w") as f:
        f.write(str(int(flag)))

os.makedirs("./save_load_speed_test", exist_ok=True)
reqed_flag_path = "./save_load_speed_test/saved_flag.txt"
answered_flag_path = "./save_load_speed_test/answered_flag.txt"
image_path = "./save_load_speed_test/image.jpg"
detection_dict_path = "./save_load_speed_test/detection_dict.npy"

if not os.path.exists(reqed_flag_path):
    write_flag(reqed_flag_path, False)
if not os.path.exists(answered_flag_path):
    write_flag(answered_flag_path, False)
image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)

if sys == "Windows":
    t1 = 0
    t2 = 0
    times = []
    while True:
        if read_flag(answered_flag_path, True): #只要不是1就不触发
            JsonIO.load_json(detection_dict_path)
            write_flag(answered_flag_path, False)        
            t1 = time.time()
        if not read_flag(reqed_flag_path, False): # 只要不是0就不触发
            cv2.imwrite(image_path, image)
            write_flag(reqed_flag_path, True)
            t2 = time.time()

            print('\r', "save time: ", t2 - t1, end="")
else:
    while True:
        if read_flag(reqed_flag_path, True):
            image = cv2.imread(image_path)
            detection_dict = {"0": np.random.randint(0, 255, (200, 2), dtype=np.uint16),
                              "1": 0.8,
                              "2": np.random.randint(0, 255, (100, 4), dtype=np.uint8)}
            
            JsonIO.dump_json(detection_dict_path, detection_dict)

            write_flag(answered_flag_path, True)
            write_flag(reqed_flag_path, False)