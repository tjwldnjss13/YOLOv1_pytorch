import torch
import numpy as np

from utils import time_calculator


def log_loss(predict, target):
    return -(target * np.log2(predict + 1e-15) + (1 - target) * np.log2(1 - predict))

h, m, s = time_calculator(63)
print('{:02d} {:.2f}'.format(m, s))