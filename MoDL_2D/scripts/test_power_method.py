import os, sys
import glob
import h5py
import argparse
import numpy as np
from matplotlib import pyplot as plt

# add PyTorch modules
import torch
from torch import nn
from utils import complex_utils as cplx
from utils import transforms as T

# add BART modules
toolbox_path = "/home/sandino/bart/bart-0.5.00/"
os.environ["TOOLBOX_PATH"] = toolbox_path
sys.path.append(os.path.join(toolbox_path, 'python'))
import bart, cfl

np.random.seed(0)

# blocking params
blk_size = 16
overlapping = False # overlapping vs. non-overlapping blocks
num_iter = 10 # number of power iterations

# data dimensions
npp = 10
ne = 2
nt = 20
nb = 8 # number of basis functions
scale = 1e2

# Construct random matrices
L_shape = (npp, ne*blk_size**2, nb)
L = torch.stack((torch.rand(L_shape), torch.rand(L_shape)), dim=-1)
R_shape = (npp, nt, nb)
R = torch.stack((torch.rand(R_shape), torch.rand(R_shape)), dim=-1)

# Scale
L *= scale
R *= scale

# Compute L step size (SVD method)
_, S, _ = cplx.svd(L, compute_uv=True)
step_size_L1 = -1.0 / S.max()**2

# Compute L step size (power method)
eigenvals = cplx.power_method(L, num_iter)
step_size_L2 = -1.0 / eigenvals.max()

# Compute R step size (SVD method)
_, S, _ = cplx.svd(R, compute_uv=True)
step_size_R1 = -1.0 / S.max()**2

# Compute R step size (power method)
eigenvals = cplx.power_method(R, num_iter)
step_size_R2 = -1.0 / eigenvals.max()


print(step_size_L1)
print(step_size_L2)
print(step_size_R1)
print(step_size_R2)




