"""
Adds sensitivity maps to fastMRI dataset
"""

import os, sys
import glob
import h5py
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

def main():
	# ARGS
	input_data_path = '/mnt/dense/data_public/fastMRI/multicoil_val'
	output_data_path = '/mnt/raid3/sandino/fastMRI/validate_full'
	center_fraction = 0.04 # number of k-space lines used to do ESPIRiT calib
	num_emaps = 1
	dbwrite = False

	input_files = glob.glob(os.path.join(input_data_path, '*.h5'))
	
	for file in input_files:
		# Load HDF5 file
		hf = h5py.File(file, 'r')
		# existing keys: ['ismrmrd_header', 'kspace', 'reconstruction_rss']

		# load k-space and image data from HDF5 file
		kspace_orig = hf['kspace'][()] 
		im_rss = hf['reconstruction_rss'][()] # (33, 320, 320)

		# get data dimensions
		num_slices, num_coils, num_kx, num_ky = kspace_orig.shape
		xres, yres = im_rss.shape[1:3] # matrix size
		num_low_freqs = int(round(center_fraction * yres))

		# allocate memory for new arrays
		im_shape = (xres, yres)
		kspace = np.zeros((num_slices, xres, yres, num_coils), dtype=np.complex64)
		maps = np.zeros((num_slices, xres, yres, num_coils, num_emaps), dtype=np.complex64)
		im_truth = np.zeros((num_slices, xres, yres, num_emaps), dtype=np.complex64)

		for sl in range(num_slices):
			kspace_slice = np.transpose(kspace_orig[sl], axes=[1,2,0])
			kspace_slice = kspace_slice[:,:,None,:]

			# Data dimensions for BART:
			#  kspace - (kx, ky, 1, coils) 
            #  maps - (kx, ky, 1, coils, emaps)
            # Data dimensions for PyTorch:
            #  kspace - (1, kx, ky, coils, real/imag)
            #  maps   - (1, kx, ky, coils, emaps, real/imag)

			# Pre-process k-space data (PyTorch)
			kspace_tensor = cplx.to_tensor(np.transpose(kspace_slice, axes=[2,0,1,3])) # (1, 640, 372, 15, 2)
			image_tensor = T.ifft2(kspace_tensor)
			print(image_tensor.size())
			image_tensor = cplx.center_crop(image_tensor, im_shape)
			kspace_tensor = T.fft2(image_tensor)
			kspace_slice = np.transpose(cplx.to_numpy(kspace_tensor), axes=[1,2,0,3])

			# Compute sensitivity maps (BART)
			maps_slice = bart.bart(1, f'ecalib -d 0 -m {num_emaps} -c 0.1 -r {num_low_freqs}', kspace_slice)
			maps_slice = np.reshape(maps_slice, (xres, yres, 1, num_coils, num_emaps))
			maps_tensor = cplx.to_tensor(np.transpose(maps_slice, axes=[2,0,1,3,4]))

			# Do coil combination using sensitivity maps (PyTorch)
			A = T.SenseModel(maps_tensor)
			im_tensor = A(kspace_tensor, adjoint=True)

			# Convert image tensor to numpy array
			im_slice = cplx.to_numpy(im_tensor)

			# Re-shape and save everything
			kspace[sl] = np.reshape(kspace_slice, (xres, yres, num_coils))
			maps[sl]   = np.reshape(maps_slice, (xres, yres, num_coils, num_emaps))
			im_truth[sl] = np.reshape(im_slice, (xres, yres, num_emaps))

		# write out new hdf5
		file_new = os.path.join(output_data_path, os.path.split(file)[-1])
		with h5py.File(file_new, 'w') as hf_new:
			# create datasets within HDF5
			hf_new.create_dataset('kspace', data=kspace)
			hf_new.create_dataset('maps', data=maps)
			hf_new.create_dataset('reconstruction_espirit', data=im_truth)
			hf_new.create_dataset('reconstruction_rss', data=im_rss) # provided by fastMRI
			hf_new.create_dataset('ismrmrd_header', data=hf['ismrmrd_header'])

			# create attributes (metadata)
			for key in hf.attrs.keys():
				hf_new.attrs[key] = hf.attrs[key]

		if dbwrite: 
			hf_new = h5py.File(file_new, 'r')
			print('Keys:', list(hf_new.keys()))
			print('Attrs:', dict(hf_new.attrs))
			cfl.writecfl('/home/sandino/maps', hf_new['maps'][()])
			cfl.writecfl('/home/sandino/kspace', hf_new['kspace'][()])
			cfl.writecfl('/home/sandino/im_truth', hf_new['reconstruction_rss'][()])
			cfl.writecfl('/home/sandino/im_recon',  hf_new['reconstruction_espirit'][()])


if __name__ == '__main__':
    main()