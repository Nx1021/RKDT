from . import LOGS_DIR, _get_sub_log_dir

import torch
import sys
import io
from typing import Union
import platform
import os
import shutil
import datetime
from utils.yaml import dump_yaml, load_yaml
import pandas as pd
import time

from typing import Callable
from functools import wraps 

class BaseLogger():
    def __init__(self, log_dir):
        # 初始化日志保存目录
        self.log_dir = log_dir
    
    def log(self, setup_info:Union[dict, str], filename:str = None):
        # 日志记录方法，根据输入的参数类型进行不同的操作
        if isinstance(setup_info, str):
            # 如果输入是字符串
            if filename is not None and filename.endswith(".txt"):
                # 如果指定了文件名并且文件名以.txt结尾，将字符串保存到文本文件
                save_path = os.path.join(self.log_dir, filename)
                with open(save_path, "w") as file:
                    # 将结果写入文件
                    file.write(setup_info)
            elif os.path.exists(setup_info):
                # 如果字符串是一个现有的文件路径，复制该文件到指定位置
                save_name: str = os.path.split(setup_info)[1] if filename is None else filename
                save_path = os.path.join(self.log_dir, save_name)
                shutil.copy(setup_info, save_path)
            else:
                # 抛出异常，表示输入的字符串既不是一个文件路径也不是需要保存的内容
                raise ValueError(f"{setup_info} does not exist")
        elif isinstance(setup_info, dict):
            # 如果输入是字典，将字典保存为YAML文件
            save_name: str = "setup.yaml" if filename is None else filename
            if not save_name.endswith(".yaml"):
                save_name = save_name + ".yaml"
            save_path = os.path.join(self.log_dir, save_name)
            dump_yaml(save_path, setup_info)
        else:
            # 抛出异常，表示输入的参数类型不支持
            raise ValueError
        
    def capture_output(self, save_name):
        # 开启输出捕获的上下文管理器
        self.__capture = PrintCapture(self, save_name)
        return self.__capture

class PrintCapture:
    def __init__(self, father_logger:BaseLogger, save_name:str):
        # 初始化输出捕获器
        self.output_buffer = io.StringIO()
        self.original_stdout = sys.stdout

        self.father_logger = father_logger
        # 检查文件名后缀，确保是.txt格式
        if not save_name.endswith(".txt"):
            save_name = save_name + ".txt"
        self.save_name = save_name

    def __enter__(self):
        # 进入上下文，将标准输出重定向到输出缓冲区
        sys.stdout = self.output_buffer
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # 退出上下文，将标准输出恢复为原始标准输出，并记录输出内容
        sys.stdout = self.original_stdout
        self.log_output()

    def log_output(self):
        # 将输出内容保存到日志，并清空缓冲区
        output = self.output_buffer.getvalue()
        print(output)
        self.father_logger.log(output, self.save_name)
        self.output_buffer.truncate(0)  # 清空缓冲区
        self.output_buffer.seek(0)  # 重置缓冲区的指针
        return output

def get_attr_by_class(obj, class_:type):
    process_timers = {}
    attributes = vars(obj)  # 获取对象的所有属性和值

    for attr_name, attr_value in attributes.items():
        if isinstance(attr_value, class_):
            process_timers[attr_name] = attr_value

    return process_timers

class _process_timer():
    def __init__(self, name:str) -> None:
        self.reset()
        self.name = name

    def update(self, time, bn = 1):
        self.time += time
        self.count += bn

    def reset(self):
        self.time = 0
        self.count = 0

    def print(self, intent=4):
        print(self.name)
        if self.count == 0:
            print(" " * intent + "not executed")
            return
        print(" " * intent + "total_frames: " + str(self.count))
        print(" " * intent + "total_time: " + str(self.time))
        rate = self.count / self.time
        if rate > 1:
            print(" " * intent + "frame_rate: " + str(rate))
        else:
            print(" " * intent + "average_time_per_frame: " + str(1 / rate))

class FrameTimer():
    def __init__(self) -> None:
        self.timers:dict[str, _process_timer] = {}

    def get(self, name):
        return self.timers.setdefault(name, _process_timer(name))

    def reset(self):
        for x in self.timers.values():
            x.reset()      

    def print(self):
        process_timers: dict[str, _process_timer] = self.timers

        total_time      = sum(process_timer.time  for process_timer in process_timers.values())
        total_frames    = max(process_timer.count for process_timer in process_timers.values())

        if total_frames == 0:
            print("No frames processed yet")
            return

        average_frame_rate = total_frames / total_time

        print("Total time:", total_time)
        print("Total num frames:", total_frames)
        print("Average frame rate:", average_frame_rate)

        for name, obj in process_timers.items():
            # 调用每个_process_timer对象的print函数
            obj.print(intent=4)
        print()


class Launcher():
    DONOT_COUNT_BATCH = 0
    COUNT_BY_INPUT_1 = 1
    COUNT_BY_RETURN_1 = -1

    def __init__(self, model, batch_size=32, log_remark = "") -> None:
        # 初始化 Launcher 类
        self.model:torch.Module = model
        self.batch_size: int = batch_size
        self.sys: str = platform.system()

        # 获取当前时间戳，并创建日志保存目录
        current_time = datetime.datetime.now()
        self.start_timestamp: str = current_time.strftime("%Y%m%d%H%M%S")
        self.log_root: str  = _get_sub_log_dir(self.__class__)
        self.log_dir: str   = self.log_root + self.start_timestamp + self.sys
        if log_remark != '':
            self.log_dir += '_' + log_remark # TensorBoard日志文件保存目录
        os.makedirs(self.log_dir, exist_ok=True)

        self.frame_timer = FrameTimer()

    @staticmethod
    def timing(count_batch_from = DONOT_COUNT_BATCH):
        '''
        time the function

        parameters
        ----
        * count_batch_from: int, default Launcher.DONOT_COUNT_BATCH.
        if count_batch_from == 0, the function will not count the batch number.
        if count_batch_from > 0, the function will count the batch number from the count_batch_from-th input.
        if count_batch_from < 0, the function will count the batch number from the count_batch_from-th return.
        '''
        assert isinstance(count_batch_from, int)
        def decorator(func:Callable):
            @wraps(func)
            def wrapper(obj: Launcher, *args, **kwargs):
                timer = obj.frame_timer.get(func.__name__)
                start = time.time()
                rlt = func(obj, *args, **kwargs)
                end = time.time()
                if count_batch_from == 0:
                    bn = 1
                elif count_batch_from > 0:
                    bn = len(args[count_batch_from - 1])
                else: 
                    if isinstance(rlt, tuple):
                        bn = len(rlt[-count_batch_from - 1])
                    else:
                        bn = len(rlt)
                timer.update((end - start), bn)
                return rlt
            return wrapper
        return decorator


