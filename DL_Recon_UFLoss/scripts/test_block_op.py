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
block_size = 16
block_stride = 16

# test dataset
filename = '/data/sandino/Cine/validate/Exam2200_Series5_Phases20.h5'

slice = 0 # pick slice
with h5py.File(filename, 'r') as data:
	orig_images = data['target'][slice] 

 # Convert numpy array to tensor
images = cplx.to_tensor(orig_images).unsqueeze(0)
_, nx, ny, nt, nmaps, _ = images.shape

# Initialize blocking operator
block_op = T.ArrayToBlocks(block_size, images.shape, overlapping=True)

blocks = block_op(images)
images = block_op(blocks, adjoint=True)
images = images.squeeze(0)

# Write out images
images = cplx.to_numpy(images)
cfl.writecfl('block_input',  orig_images)
cfl.writecfl('block_output', images)
cfl.writecfl('block_error',  orig_images-images)