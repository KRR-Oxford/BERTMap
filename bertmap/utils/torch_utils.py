import torch
import random
import numpy as np

def get_device(device_num=0):
    # If there's a GPU available...
    if torch.cuda.is_available():    
        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(device_num))
    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device

def set_seed(seed_val=888):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)