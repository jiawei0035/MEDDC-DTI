import os
import random
import numpy as np
import scipy.sparse as sp
import torch


def set_global_seed(seed, deterministic=1, warn_only=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    if int(deterministic):
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=bool(int(warn_only)))


def normalize_features(matrix):
    row_sum = np.array(matrix.sum(1))
    inv = np.power(row_sum, -1).flatten()
    inv[np.isinf(inv)] = 0.0
    return sp.diags(inv).dot(matrix)


def ensure_reproducible_environment():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
