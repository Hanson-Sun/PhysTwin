from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def distCUDA2(points):
    return _C.distCUDA2(points)
