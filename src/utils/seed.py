import random
import os
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    
    Args:
        seed: The integer seed value.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Force deterministic algorithms in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
