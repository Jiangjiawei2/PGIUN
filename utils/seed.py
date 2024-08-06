import numpy as np
import torch
import random


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU
    random.seed(seed)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.manual_seed(seed)  # GPU
