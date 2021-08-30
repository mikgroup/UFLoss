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

# blocking params
blk_size = 16
nb = 4 # number of basis functions
overlapping = True # overlapping vs. non-overlapping blocks

# test dataset
filename = '/data/sandino/Cine/validate/Exam2200_Series5_Phases20.h5'

slice = 0 # pick slice
with h5py.File(filename, 'r') as data:
	orig_images = data['target'][slice] 

 # Convert numpy array to tensor
images = cplx.to_tensor(orig_images).unsqueeze(0)
_, nx, ny, nt, ne, _ = images.shape

# Initialize blocking operator
block_op = T.ArrayToBlocks(blk_size, images.shape, overlapping)

# Extract spatial patches across images
patches = block_op(images)
np = patches.shape[0]

# Reshape into batch of 2D matrices
patches = patches.permute(0,1,2,4,3,5)
patches = patches.reshape((np, ne*blk_size**2, nt, 2))

# Perform SVD to get left and right singular vectors
U, S, V = cplx.svd(patches, compute_uv=True)

# Truncate singular values and corresponding singular vectors
U = U[:, :, :nb, :] # [N, Px*Py*E, B, 2]
S = S[:, :nb]       # [N, B]
V = V[:, :, :nb, :] # [N, T, B, 2]

# Combine and reshape matrices
S_sqrt = S.reshape((np, 1, 1, 1, 1, nb, 1)).sqrt()
L = U.reshape((np, blk_size, blk_size,  1, ne, nb, 2)) * S_sqrt
R = V.reshape((np,   1,   1, nt,  1, nb, 2)) * S_sqrt
blocks = torch.sum(cplx.mul(L, cplx.conj(R)), dim=-2)

images = block_op(blocks, adjoint=True)

# Write out images
images = cplx.to_numpy(images.squeeze(0))
cfl.writecfl('svdinit_input',  orig_images)
cfl.writecfl('svdinit_output', images)
cfl.writecfl('svdinit_error',  orig_images-images)

