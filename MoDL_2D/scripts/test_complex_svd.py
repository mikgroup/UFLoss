import os, sys
import glob
import h5py
import time
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

# simulate complex-valued 2D matrix
batch_size = 700
m = 16 * 16 * 2
n = 20
X = np.random.randn(batch_size, m, n) + 1j*np.random.randn(batch_size, m, n)
X = X.astype(np.complex64)

# perform SVD (numpy)
numpy_start_time = time.time()
U, S, Vh = np.linalg.svd(X, full_matrices=False)
numpy_end_time = time.time()
print('Numpy time: %f' % (numpy_end_time - numpy_start_time))

# convert matrix into a pytorch tensor
X_torch = cplx.to_tensor(X)

# perform SVD (pytorch)
torch_start_time = time.time()
U_torch, S_torch, V_torch = cplx.svd2(X_torch, compute_uv=True)
torch_end_time = time.time()
print('PyTorch (CPU) time: %f' % (torch_end_time - torch_start_time))

X_torch = X_torch.cuda()
gpu_start_time = time.time()
U_gtorch, S_gtorch, V_gtorch = cplx.svd2(X_torch, compute_uv=True)
gpu_end_time = time.time()
print('PyTorch (GPU) time: %f' % (gpu_end_time - gpu_start_time))

#print('CPU time: %f' % (cpu_end_time - cpu_start_time))
#print('GPU time: %f' % (gpu_end_time - gpu_start_time))

# convert to numpy arrays
U_torch = cplx.to_numpy(U_torch)
S_torch = S_torch.numpy() # real-valued
V_torch = cplx.to_numpy(V_torch)

if 1:
	print('numpy')
	print(U.shape)
	print(S.shape)
	print(Vh.shape)
	print('\npytorch')
	print(U_torch.shape)
	print(S_torch.shape)
	print(V_torch.shape)

# re-compose
for i in range(batch_size):
	X1 = (U[i] * S[i, None, :]) @ Vh[i] 
	X2 = (U_torch[i] * S_torch[i, None, :]) @ np.conj(V_torch[i].T)
	#print(np.abs(X2-X))
	#print(np.sum(np.abs(X1 - X[i])))
	#print(np.sum(np.abs(X2 - X[i])))
	#print(np.amax(np.abs(X1 - X)))
	#print(np.amax(np.abs(X2 - X)))
	#print(np.abs(X1-X))


