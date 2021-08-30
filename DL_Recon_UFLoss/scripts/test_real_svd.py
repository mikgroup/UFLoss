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

np.random.seed(2)

# simulate complex-valued 2D matrix
batch_size = 1
m = 10
n = 10
X = np.random.randn(batch_size, m, n)
X = X.astype(np.float32)

# compute singular values using numpy
U, S, V = np.linalg.svd(X)

# convert matrix into a pytorch tensor
X_torch = torch.from_numpy(X)

# perform SVD
U_torch, S_torch, V_torch = torch.svd(X_torch)

# convert to numpy arrays
U_torch = U_torch.numpy()
S_torch = S_torch.numpy()
V_torch = V_torch.numpy()

print('numpy')
print(U.shape)
print(S.shape)
print(V.shape)

print('\npytorch')
print(U_torch.shape)
print(S_torch.shape)
print(V_torch.shape)

# re-compose
for i in range(batch_size):
	X1 = np.matmul(U[i], np.matmul(np.diag(S[i]), V[i]))
	X2 = np.matmul(U_torch[i], np.matmul(np.diag(S_torch[i]), V_torch[i].T))

	print(np.sum(np.abs(X1 - X)))
	print(np.sum(np.abs(X2 - X)))

