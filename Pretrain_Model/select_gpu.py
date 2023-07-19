import subprocess
import re
import torch


def get_lowest_usage_gpu_index(print_usage=False):
    # Execute nvidia-smi command to get GPU usage information
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
                            stdout=subprocess.PIPE, encoding='utf-8')
    gpu_info = result.stdout.strip().split('\n')
    gpu_usage = []
    for info in gpu_info:
        used_memory, total_memory = info.split(',')
        gpu_usage.append(int(used_memory) / int(total_memory))

    # Print GPU usage
    if print_usage:
        print("GPU Usage:")
        for i, usage in enumerate(gpu_usage):
            print("GPU {}: {:.2%}".format(i, usage))

    # Select the GPU with the lowest usage
    lowest_usage = float('inf')
    selected_gpu = -1
    for i in range(torch.cuda.device_count()):
        if gpu_usage[i] < lowest_usage:
            lowest_usage = gpu_usage[i]
            selected_gpu = i

    if selected_gpu != -1:
        # selected_gpu = 2
        print("\nSelected GPU: {}".format(selected_gpu))
    else:
        print("\nNo GPU available or couldn't find the GPU with the lowest usage.")

    return selected_gpu


def get_gpu_with_most_free_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    output = subprocess.check_output(command, shell=True).decode("utf-8")

    gpu_memory_free = re.findall(r"(\d+) MiB", output)  # 提取所有GPU的空闲内存数量
    gpu_memory_free = [memory for memory in gpu_memory_free]  # 转换为整数
    print(gpu_memory_free)
    gpu_with_most_free_memory = gpu_memory_free.index(max(gpu_memory_free))  # 获取空闲内存最多的GPU索引

    return gpu_with_most_free_memory

# selected_gpu = get_lowest_usage_gpu_index()
# torch.cuda.set_device(selected_gpu)
# device = torch.device("cuda:{}".format(selected_gpu))
