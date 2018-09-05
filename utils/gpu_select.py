import numpy as np
import commands
import time

THRESHOLD = {'memory': 2500, 'usage': 20}

def get_gpu_info():
    # Parsing nvidia-smi information
    gpu_info_dict = {}
    # Memory
    _, output = commands.getstatusoutput('nvidia-smi -q -d Memory | grep -A4 GPU | grep Used')
    gpu_info_dict['memory'] = np.array([])
    for output_str in output.split('\n'):
        gpu_info_dict['memory'] = np.append(gpu_info_dict['memory'], int(filter(str.isdigit, output_str)))
    # Usage
    _, output = commands.getstatusoutput('nvidia-smi -q -d Utilization | grep -A4 GPU | grep Gpu')
    gpu_info_dict['usage'] = np.array([])
    for output_str in output.split('\n'):
        gpu_info_dict['usage'] = np.append(gpu_info_dict['usage'], int(filter(str.isdigit, output_str)))

    memory_mask = np.less_equal(gpu_info_dict['memory'], THRESHOLD['memory'])
    usage_mask = np.less_equal(gpu_info_dict['usage'], THRESHOLD['usage'])
    total_mask = np.logical_and(memory_mask, usage_mask)

    return np.arange(len(total_mask))[total_mask]

def gpu_select(num_gpus, wait_hour=0, wait_for_long=False):
    ready_flag = False
    avail_gpu_idx = get_gpu_info()
    if len(avail_gpu_idx) < num_gpus:
        if wait_hour > 0:
            print('No free GPUs can be used for tasks now, and wait for %.1f hours'%wait_hour)
            time.sleep(wait_hour*3600)
            avail_gpu_idx = get_gpu_info()
            while len(avail_gpu_idx) < num_gpus:
                if wait_for_long:
                    print('No free GPUs can be used for tasks now, and wait for %.1f hours'%wait_hour)
                    time.sleep(wait_hour*3600)
                    avail_gpu_idx = get_gpu_info()
                else:
                    raise MemoryError('No free GPUs can be used for tasks after waiting.')
        else:
            raise MemoryError('No free GPUs can be used for tasks now, but no wait request.')

    gpu_list = ','.join([str(idx) for idx in avail_gpu_idx[:num_gpus]])
    print('Use GPU: %s' % gpu_list)
    
    return gpu_list
