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

def glr_compress(images, nb):
	_, nx, ny, nt, ne, _ = images.shape

	images = images.permute(0,1,2,4,3,5)
	images = images.reshape((1, nx*ny*ne, nt, 2))

	# Perform SVD to get left and right singular vectors
	U, S, V = cplx.svd(images, compute_uv=True)

	# Truncate singular values and corresponding singular vectors
	U = U[:, :, :nb, :] # [1, Nx*Ny*E, B, 2]
	S = S[:, :nb]       # [1, B]
	V = V[:, :, :nb, :] # [1, T, B, 2]

	# Combine and reshape matrices
	S_sqrt = S.reshape((1, 1, 1, 1, 1, nb, 1)).sqrt()
	L = U.reshape((1, nx, ny,  1, ne, nb, 2)) * S_sqrt
	R = V.reshape((1,  1,  1, nt,  1, nb, 2)) * S_sqrt
	images = torch.sum(cplx.mul(L, cplx.conj(R)), dim=-2)

	return images

def llr_compress(images, nb, block_size, overlapping):
	# Initialize blocking operator
	block_op = T.ArrayToBlocks(block_size, images.shape, overlapping)

	# Extract spatial patches across images
	patches = block_op(images)
	np = patches.shape[0]

	# Reshape into batch of 2D matrices
	patches = patches.permute(0,1,2,4,3,5)
	patches = patches.reshape((np, ne*block_size**2, nt, 2))

	# Perform SVD to get left and right singular vectors
	U, S, V = cplx.svd(patches, compute_uv=True)

	# Truncate singular values and corresponding singular vectors
	U = U[:, :, :nb, :] # [N, Px*Py*E, B, 2]
	S = S[:, :nb]       # [N, B]
	V = V[:, :, :nb, :] # [N, T, B, 2]

	# Combine and reshape matrices
	S_sqrt = S.reshape((np, 1, 1, 1, 1, nb, 1)).sqrt()
	L = U.reshape((np, block_size, block_size,  1, ne, nb, 2)) * S_sqrt
	R = V.reshape((np,   1,   1, nt,  1, nb, 2)) * S_sqrt
	blocks = torch.sum(cplx.mul(L, cplx.conj(R)), dim=-2)

	images = block_op(blocks, adjoint=True)

	return images

# blocking params
blk_size = 16
num_basis = [16, 8, 4, 2] # number of basis functions
overlapping = True # overlapping vs. non-overlapping blocks

# test dataset
filename = '/data/sandino/Cine/validate/Exam2200_Series5_Phases20.h5'

slice = 0 # pick slice
with h5py.File(filename, 'r') as data:
	orig_images = data['target'][slice] 

 # Convert numpy array to tensor
images = cplx.to_tensor(orig_images).unsqueeze(0)
_, nx, ny, nt, ne, _ = images.shape

# Initialize lists
glr_images = [None] * len(num_basis)
glr_error = [None] * len(num_basis)
llr_images = [None] * len(num_basis)
llr_error = [None] * len(num_basis)

for i in range(len(num_basis)):
	# Use globally low-rank model to compress images
	glr_images[i] = glr_compress(images, num_basis[i])
	glr_error[i] = images - glr_images[i]

	# Use locally low-rank model to compress images
	llr_images[i] = llr_compress(images, num_basis[i], blk_size, overlapping)
	llr_error[i] = images - llr_images[i]

glr_images = torch.cat(glr_images, axis=2).squeeze(0)
glr_error = torch.cat(glr_error, axis=2).squeeze(0)
llr_images = torch.cat(llr_images, axis=2).squeeze(0)
llr_error = torch.cat(llr_error, axis=2).squeeze(0)

# Write out images
cfl.writecfl('svd_glr_images', cplx.to_numpy(glr_images).swapaxes(0,1))
cfl.writecfl('svd_glr_error', cplx.to_numpy(glr_error).swapaxes(0,1))
cfl.writecfl('svd_llr_images', cplx.to_numpy(llr_images).swapaxes(0,1))
cfl.writecfl('svd_llr_error', cplx.to_numpy(llr_error).swapaxes(0,1))
