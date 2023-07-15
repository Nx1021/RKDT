from . import LOGS_DIR

from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter  
import os
import shutil

root = f"{LOGS_DIR}/Trainer_logs/"

for r,d,f in os.walk(root):
    for file in f:
        input_path = os.path.join(r, file)  # 输入需要指定event文件
        if "DESKTOP-SND9T9D" in file:
            os.remove(input_path)
            continue
        if "tfevent" in file:
            # 读取需要修改的event文件
            ea = event_accumulator.EventAccumulator(input_path)
            ea.Reload()
            tags = ea.scalars.Keys()  # 获取所有scalar中的keys
            
            # 写入新的文件
            if len(tags) == 0:
                os.remove(input_path)
                continue
            writer = SummaryWriter(r + "_")  # 创建一个SummaryWriter对象
            for tag in tags:
                scalar_list = ea.scalars.Items(tag)
            
                if tag == 'Train DecoderLoss':  # 修改一下对应的tag即可
                    tag = 'Train Last Decoder Loss'
                elif tag == 'Val DecoderLoss':  # 修改一下对应的tag即可
                    tag = 'Val Last Decoder Loss'
                elif tag == 'Train Dist Loss':  # 修改一下对应的tag即可
                    tag = 'Train Distance Loss'
                elif tag == 'Val Dist Loss':  # 修改一下对应的tag即可
                    tag = 'Val Distance Loss'
                elif tag == 'Train Obj Loss':  # 修改一下对应的tag即可
                    tag = 'Train PN Loss'
                elif tag == 'Val Obj Loss':  # 修改一下对应的tag即可
                    tag = 'Val PN Loss'
                for scalar in scalar_list:
                    writer.add_scalar(tag, scalar.value, scalar.step, scalar.wall_time)  # 添加修改后的值到新的event文件中
            writer.close()  # 关闭SummaryWriter对象

            new_name = os.path.join(r + "_", os.listdir(r + "_")[0])
            shutil.copy(new_name, input_path)

            shutil.rmtree(r + "_")