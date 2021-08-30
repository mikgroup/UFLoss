"""
Creates HDF5 files for training DL cine network.
"""

import os, sys
import glob
import h5py
import argparse
import numpy as np

# import FFT libraries
try:
    import pyfftw.interfaces.numpy_fft as fft
except:
    from numpy import fft

import torch
from torch import nn
from utils import complex_utils as cplx
from utils import transforms as T
from utils import fftc

# add BART modules
toolbox_path = "/home/sandino/bart/bart-0.5.00/"
os.environ["TOOLBOX_PATH"] = toolbox_path
sys.path.append(os.path.join(toolbox_path, 'python'))
import bart, cfl


def time_average(kspace, axis=0):
	"""
	Computes time average across a specified axis.
	"""
	eps =  np.finfo(kspace.dtype).eps
	kspace_avg = np.sum(kspace, axis=axis)
	kspace_avg /= (np.sum(kspace!=0, axis=axis) + eps)

	return kspace_avg


def center_crop(data, out_shape):
	ndims = len(data.shape)
	for i in range(ndims):
		if data.shape[i] == out_shape[i]:
			continue
		idx_from = (data.shape[i] - out_shape[i]) // 2
		idx_to = idx_from + out_shape[i]
		data = np.take(data, np.arange(idx_from, idx_to), axis=i)
	return data


def reduce_fov(kspace, factor=0.15):
	"""
	Simulates a reduced field of view across the phase encoding direction.
	"""
	return kspace


def process_slice(kspace, args):
	nkx, nky, nphases, ncoils = kspace.shape

	if 0 < args.crop_size < nkx:
		# crop along readout dimension
		images = fftc.ifftc(kspace, axis=0)
		images = center_crop(images, [args.crop_size, nky, nphases, ncoils])
		kspace = fftc.fftc(images, axis=0).astype(np.complex64)
		nkx = args.crop_size

	# simulate reduced FOV
	#kspace = reduce_fov(kspace, ...)

	# compute time-average for ESPIRiT calibration
	kspace_avg = time_average(kspace, axis=-2)

	# ESPIRiT - compute sensitivity maps
	cmd = f'ecalib -d 0 -S -m {args.nmaps} -c {args.crop_value} -r {args.calib_size}'
	maps = bart.bart(1, cmd, kspace_avg[:,:,None,:])
	maps = np.reshape(maps, (nkx, nky, 1, ncoils, args.nmaps))

	# Convert everything to tensors
	kspace_tensor = cplx.to_tensor(kspace).unsqueeze(0)
	maps_tensor = cplx.to_tensor(maps).unsqueeze(0)

	# Do coil combination using sensitivity maps (PyTorch)
	A = T.SenseModel(maps_tensor)
	im_tensor = A(kspace_tensor, adjoint=True)

	# Convert tensor back to numpy array
	target = cplx.to_numpy(im_tensor.squeeze(0))

	return kspace, maps, target


def main(args):

	# create data directories if they don't already exist
	if not os.path.exists(args.output_path):
		os.makedirs(args.output_path)
	if not os.path.exists(os.path.join(args.output_path, 'train')):
		os.makedirs(os.path.join(args.output_path, 'train'))
	if not os.path.exists(os.path.join(args.output_path, 'validate')):
		os.makedirs(os.path.join(args.output_path, 'validate'))
	if not os.path.exists(os.path.join(args.output_path, 'test')):
		os.makedirs(os.path.join(args.output_path, 'test'))

	# determine splits manually for now...
	train_exams = ['Exam2323', 'Exam3330', 'Exam3331', 
				   'Exam3332', 'Exam3410', 'Exam3411', 
				   'Exam3412', 'Exam4873', 'Exam4874', 
				   'Exam4905', 'Exam5003', 'Exam2406']
	val_exams   = ['Exam2200', 'Exam5050']
	test_exams  = []
	all_exams = train_exams + val_exams + test_exams

	# figure out data splits
	num_train = len(train_exams)
	num_validate = len(val_exams)
	num_test = len(test_exams)
	num_cases = num_train + num_validate + num_test

	# how many cardiac phases to simulate
	num_phases_list = [20]

	for exam_name in all_exams:
		exam_path = os.path.join(args.input_path, exam_name)
		series_list = os.listdir(exam_path)

		if args.verbose:
			print("Processing %s..." % exam_name)

		for num_phases in num_phases_list:

			for series_name in series_list:
				series_path = os.path.join(exam_path, series_name, 'Phases%d' % num_phases)

				num_slices = len(glob.glob('%s/ks_*.cfl' % series_path))
				kspace = [None] * num_slices
				maps   = [None] * num_slices
				target = [None] * num_slices

				if args.verbose:
					print("  %s (%d slices)..." % (series_name, num_slices))

				for sl in range(num_slices):
					# loading k-space data
					file_ks = "ks_%02d" % sl
					ks_slice = cfl.readcfl(os.path.join(series_path, file_ks))
					ks_slice = np.transpose(np.squeeze(ks_slice), [0,1,3,2])

					# process slice to get images and sensitivity maps
					kspace[sl], maps[sl], target[sl] = process_slice(ks_slice, args)

				# Stack volume
				kspace = np.stack(kspace, axis=0)
				maps = np.stack(maps, axis=0)
				target = np.stack(target, axis=0)

				# Determine path to new hdf5 file
				if exam_name in train_exams:
					folder = 'train'
				elif exam_name in val_exams:
					folder = 'validate'
				else:
					folder = 'test'

				# write out HDF5 file for entire volume
				h5_name = "%s_%s_Phases%02d.h5" % (exam_name, series_name, num_phases)
				filename = os.path.join(args.output_path, folder, h5_name)
				with h5py.File(filename, 'w') as hf:
					hf.create_dataset('kspace', data=kspace)
					hf.create_dataset('maps',   data=maps)
					hf.create_dataset('target', data=target)

				if args.dbwrite:
					print('Writing out files to home folder!')
					cfl.writecfl('/home/sandino/kspace', kspace)
					cfl.writecfl('/home/sandino/maps', maps)
					cfl.writecfl('/home/sandino/images', target)

	return
		

def create_arg_parser():
	parser = argparse.ArgumentParser(description="Training script for unrolled MRI recon.")
	parser.add_argument('--input-path', type=str, 
						default='/mnt/dense/sandino/Studies_2DFiesta/CFL-Phases', help='Path to input data.')
	parser.add_argument('--output-path', type=str,  
						default='/data/sandino/Cine-Large', help='Path to output data.')
	# Data parameters
	parser.add_argument('--crop-size', type=int, default=-1, help='Number of readout points to keep.')
	# ESPIRiT parameters
	parser.add_argument('--nmaps', type=int, default=2, help='Number of ESPIRiT maps.')
	parser.add_argument('--calib-size', type=int, default=20, help='Calibration region size.')
	parser.add_argument('--crop-value', type=float, default=0.1, help='Crop value for ESPIRiT maps.')
	# Debug parameters
	parser.add_argument('--dbwrite', action='store_true', help='Write out files for debugging.')
	parser.add_argument('--verbose', action='store_true')
	return parser


if __name__ == '__main__':
	args = create_arg_parser().parse_args()
	main(args)