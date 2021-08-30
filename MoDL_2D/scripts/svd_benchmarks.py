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
num_trials = 16
batch_sizes = [1, 25, 50, 100, 200, 400, 600, 800]
m = 16 * 16 * 2
n = 20

# make lists
numpy_times = np.zeros((len(batch_sizes), num_trials), dtype=np.float32)
torch_times = np.zeros((len(batch_sizes), num_trials), dtype=np.float32)
torch_gpu_times = np.zeros((len(batch_sizes), num_trials), dtype=np.float32)

# loop over batch sizes
for i in range(len(batch_sizes)):
	batch_size = batch_sizes[i]
	print('Batch size: %d' % batch_size)

	for trial in range(num_trials):
		X = np.random.randn(batch_size, m, n) + 1j*np.random.randn(batch_size, m, n)
		X = X.astype(np.complex64)

		# perform batch-SVD (numpy)
		numpy_start_time = time.time()
		U, S, Vh = np.linalg.svd(X, full_matrices=False)
		numpy_end_time = time.time()
		numpy_times[i, trial] = numpy_end_time - numpy_start_time

		# convert matrix into a pytorch tensor
		X_torch = cplx.to_tensor(X)

		# perform SVD (pytorch, cpu)
		torch_start_time = time.time()
		U_torch, S_torch, V_torch = cplx.svd2(X_torch, compute_uv=True)
		torch_end_time = time.time()
		torch_times[i, trial] = torch_end_time - torch_start_time

		# perform batch-SVD (pytorch, gpu)
		X_torch = X_torch.cuda()
		gpu_start_time = time.time()
		U_gtorch, S_gtorch, V_gtorch = cplx.svd2(X_torch, compute_uv=True)
		gpu_end_time = time.time()
		torch_gpu_times[i, trial] = gpu_end_time - gpu_start_time

# average over trials
numpy_mean_times = np.mean(numpy_times, axis=1)
numpy_std_times = np.std(numpy_times, axis=1)
torch_mean_times = np.mean(torch_times, axis=1)
torch_std_times = np.std(torch_times, axis=1)
torch_gpu_mean_times = np.mean(torch_gpu_times, axis=1)
torch_gpu_std_times = np.std(torch_gpu_times, axis=1)

# plot results
plt.errorbar(batch_sizes, np.log10(numpy_mean_times), yerr=numpy_std_times, label='NumPy')
plt.errorbar(batch_sizes, np.log10(torch_mean_times), yerr=torch_std_times, label='PyTorch (CPU)')
plt.errorbar(batch_sizes, np.log10(torch_gpu_mean_times), yerr=torch_gpu_std_times, label='PyTorch (GPU)')
plt.xlabel('Batch size')
plt.ylabel('log(Runtime) (seconds)')
plt.title('Speed of complex-valued batch SVD implementations')
plt.legend()
plt.show()


