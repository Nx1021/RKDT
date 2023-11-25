import torch
import multiprocessing

import torch.multiprocessing as mp

def load_weights_and_run_inference(model, input_queue, output_queue):
    while True:
        input_data = input_queue.get()  # 等待主进程传入输入数据
        if input_data is None:
            break
        
        # 在这里进行模型推理或其他任务
        result = model(input_data)
        
        # 将结果放入输出队列
        output_queue.put(result)

if __name__ == '__main__':
    torch.cuda.init()
    mp.set_start_method("spawn")
    # 创建输入和输出队列
    input_queue = multiprocessing.Queue()
    output_queue = multiprocessing.Queue()

    model1 = torch.nn.Linear(10, 10).cuda()
    model2 = torch.nn.Linear(10, 10).to("cuda:0")
    model3 = torch.nn.Linear(10, 10).to("cuda:0")

    # 创建多个进程来加载权重和运行任务
    process1 = mp.Process(target=load_weights_and_run_inference, args=(model1, input_queue, output_queue))
    process1.start()
    process2 = multiprocessing.Process(target=load_weights_and_run_inference, args=(model2, input_queue, output_queue))
    process3 = multiprocessing.Process(target=load_weights_and_run_inference, args=(model3, input_queue, output_queue))

    # 启动进程
    process1.start()
    process2.start()
    process3.start()

    # 传入输入数据到子进程
    input_queue.put(input_data1)  # 替换为实际的输入数据
    input_queue.put(input_data2)
    input_queue.put(input_data3)
    
    # 发送结束信号
    input_queue.put(None)
    input_queue.put(None)
    input_queue.put(None)

    # 等待子进程完成
    process1.join()
    process2.join()
    process3.join()

    # 从输出队列中获取结果
    result1 = output_queue.get()
    result2 = output_queue.get()
    result3 = output_queue.get()
    
    # 现在，你可以使用模型的结果进行后续处理
