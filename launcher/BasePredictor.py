import torch
import numpy as np
import time
from torch.utils.data import Dataset, DataLoader
import abc

def get_attr_by_class(obj, class_:type):
    process_timers = {}
    attributes = vars(obj)  # 获取对象的所有属性和值

    for attr_name, attr_value in attributes.items():
        if isinstance(attr_value, class_):
            process_timers[attr_name] = attr_value

    return process_timers

class _process_timer():
    def __init__(self, batch_size:int) -> None:
        self.time = 0
        self.count = 0
        self.done = False
        self.batch_size = batch_size

    def update(self, time):
        self.time += time
        self.count += self.batch_size
        self.done = True

    def print(self, intent=4):
        if self.count == 0:
            print(" " * intent + "not executed")
            return
        print(" " * intent + "total_frames: " + str(self.count))
        print(" " * intent + "total_time: " + str(self.time))
        print(" " * intent + "average_time_per_frame: " + str(self.time / self.count))


class FrameTimer():
    def __init__(self, batch_size = 1) -> None:
        self.preprocess = _process_timer(batch_size)
        self.inference  = _process_timer(batch_size)
        self.postprocess= _process_timer(batch_size)
        self.calc_error = _process_timer(batch_size)

    def record(self, process_timer: _process_timer):
        def process(func):
            def wrapper(*arg, **kw):
                start = time.time()
                rlt = func(*arg, **kw)
                end = time.time()
                process_timer.update(end - start)
                return rlt
            return wrapper
        return process
    
    def is_all_done(self):
        process_timers:dict[str, _process_timer] = get_attr_by_class(self, _process_timer)
        return all([x.done for x in process_timers.values()])
    
    def reset_done(self):
        process_timers:dict[str, _process_timer] = get_attr_by_class(self, _process_timer)
        for x in process_timers.values():
            x.done = False

    def set_batch_size(self, batch_size):
        process_timers:dict[str, _process_timer] = get_attr_by_class(self, _process_timer)
        for x in process_timers.values():
            x.batch_size = batch_size        

    def print(self):
        process_timers: dict[str, _process_timer] = get_attr_by_class(self, _process_timer)

        total_time = sum(process_timer.time for process_timer in process_timers.values())
        total_frames = max( process_timer.count for process_timer in process_timers.values())

        if total_frames == 0:
            print("No frames processed yet")
            return

        average_frame_rate = total_frames / total_time

        print("Total time:", total_time)
        print("Total num frames:", total_frames)
        print("Average frame rate:", average_frame_rate)

        for name, obj in process_timers.items():
            print("{}".format(name))
            # 调用每个_process_timer对象的print函数
            obj.print(intent=4)

class BasePredictor:
    def __init__(self, model, batch_size=32):
        self.model = model
        self.batch_size = batch_size
        self.frametimer = FrameTimer()

    @abc.abstractmethod
    def _preprocess(self, image):
        pass

    def __inference(self, inputs):
        results = self.model(inputs)
        return results

    @abc.abstractmethod
    def _postprocess(self, predictions):
        pass

    def _calc_error(self, gt, pred):
        return

    def preprocess(self, image):
        # Preprocessing code here
        return self.frametimer.record(self.frametimer.preprocess)(self._preprocess)(image)

    def inference(self, inputs):
        # Model inference code here
        return self.frametimer.record(self.frametimer.inference)(self.__inference)(inputs)

    def postprocess(self, predictions):
        # Postprocessing code here
        return self.frametimer.record(self.frametimer.postprocess)(self._postprocess)(predictions)

    def calc_error(self, gt:list, postprocessed:list):
        return self.frametimer.record(self.frametimer.calc_error)(self._calc_error)(gt, postprocessed)

    # def predict_from_dataset(self, dataset, collate_fn = None):
    #     data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)

    #     # Enable GPU if available
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     self.model = self.model.to(device)
    #     self.model.eval()

    #     with torch.no_grad():
    #         for batch in data_loader:
    #             # Preprocessing
    #             inputs = self.preprocess(batch)

    #             # Model inference
    #             predictions = self.inference(inputs.to(device))

    #             # Postprocessing
    #             processed_predictions = self.postprocess(predictions)

    # def predict_single_image(self, image):
    #     # Enable GPU if available
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     self.model = self.model.to(device)
    #     self.model.eval()

    #     with torch.no_grad():
    #         preprocessed_image = self.preprocess(image)
    #         inputs = torch.from_numpy(np.expand_dims(preprocessed_image, axis=0)).to(device)
    #         predictions = self.inference(inputs)
    #         processed_predictions = self.postprocess(predictions)


    