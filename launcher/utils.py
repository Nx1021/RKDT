import subprocess

def get_gpu_with_lowest_memory_usage():
    command = "nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,nounits,noheader"
    output = subprocess.check_output(command, shell=True).decode()
    lines = output.strip().split("\n")
    
    lowest_memory_usage = float('inf')
    gpu_with_lowest_memory = None
    
    for line in lines:
        index, name, used_memory, total_memory = line.split(",")
        used_memory = int(used_memory)
        total_memory = int(total_memory)
        
        memory_info = f"{used_memory}/{total_memory} MB"
        
        gpu_info = f"GPU {index}: {name}, Memory Usage: {memory_info}"
        print(gpu_info)
        
        if used_memory < lowest_memory_usage:
            lowest_memory_usage = used_memory
            gpu_with_lowest_memory = int(index)
    
    return gpu_with_lowest_memory

