"""
Adds sensitivity maps to fastMRI dataset
"""

import os, sys
import glob
import h5py
import argparse
import numpy as np

# add SigPy modules
import sigpy as sp
from sigpy.mri import app

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

# add PyTorch modules
from utils import cfl
from utils import complex_utils as cplx
from utils import transforms as T

# add BART modules
toolbox_path = "/home/sandino/bart/bart-0.5.00/"
os.environ["TOOLBOX_PATH"] = toolbox_path
sys.path.append(os.path.join(toolbox_path, 'python'))
import bart, cfl


def generate_masks(N, shape, acceleration, calib_size, seed):
	masks = np.zeros((N, shape[0], shape[1]), dtype=np.float32)
	for i in range(N):
		masks[i] = sigpy.mri.poisson(shape, acceleration,
				   calib=calib_size,
				   dtype=np.complex64,
				   seed=np.random.seed(seed))
	return masks

def process_slice(kspace, args, calib_method='jsense'):
	# get data dimensions
	nky, nkz, nechoes, ncoils = kspace.shape

	# ESPIRiT parameters
	nmaps = args.num_emaps
	calib_size = args.ncalib
	crop_value = args.crop_value

	if args.device is -1:
		device = sp.cpu_device
	else:
		device = sp.Device(args.device)

	# compute sensitivity maps (BART)
	#cmd = f'ecalib -d 0 -S -m {nmaps} -c {crop_value} -r {calib_size}'
	#maps = bart.bart(1, cmd, kspace[:,:,0,None,:])
	#maps = np.reshape(maps, (nky, nkz, 1, ncoils, nmaps))

	# compute sensitivity maps (SigPy)
	ksp = np.transpose(kspace[:,:,0,:], [2,1,0])
	if calib_method is 'espirit':
		maps = app.EspiritCalib(ksp, calib_width=calib_size, crop=crop_value, device=device, show_pbar=False).run()
	elif calib_method is 'jsense':
		maps = app.JsenseRecon(ksp, mps_ker_width=6, ksp_calib_width=calib_size, device=device, show_pbar=False).run()
	else:
		raise ValueError('%s calibration method not implemented...' % calib_method)
	maps = np.reshape(np.transpose(maps, [2,1,0]), (nky, nkz, 1, ncoils, nmaps))

	# Convert everything to tensors
	kspace_tensor = cplx.to_tensor(kspace).unsqueeze(0)
	maps_tensor = cplx.to_tensor(maps).unsqueeze(0)

	# Do coil combination using sensitivity maps (PyTorch)
	A = T.SenseModel(maps_tensor)
	im_tensor = A(kspace_tensor, adjoint=True)

	# Convert tensor back to numpy array
	image = cplx.to_numpy(im_tensor.squeeze(0))

	return image, maps


def main(args):
	# Get list of all files
	input_files = glob.glob(os.path.join(args.input_path, '*.kspace'))
	num_files = len(input_files)

	# Akshay's selected test cases
	selected_files = ['21895_122887.kspace', '21927_204807.kspace', '21929_312327.kspace',
                      '21944_092167.kspace', '21998_757767.kspace', '22038_235527.kspace', 
                      '22065_317447.kspace', '22068_604167.kspace', '22113_942087.kspace', 
                      '22242_721927.kspace', '22320_358407.kspace', '22359_153607.kspace', 
                      '22453_046087.kspace', '22546_194567.kspace', '22563_225287.kspace', 
                      '22576_296967.kspace', '22597_296967.kspace', '22671_358407.kspace', 
                      '22705_691207.kspace', '22723_128007.kspace', '22863_071687.kspace', 
                      '23080_215047.kspace', '23097_706567.kspace', '23110_696327.kspace', 
                      '23133_230407.kspace', '23172_153607.kspace', '23226_158727.kspace', 
                      '23294_153607.kspace', '23452_286727.kspace', '23454_051207.kspace', 
                      '23515_706567.kspace', '23536_071687.kspace', '23545_481287.kspace', 
                      '23601_614407.kspace', '23632_865287.kspace', '23641_537607.kspace', 
                      '23679_906247.kspace', '23862_138247.kspace', '23902_153607.kspace', 
                      '24051_686087.kspace', '24052_727047.kspace', '24079_046087.kspace', 
                      '24159_645127.kspace', '24326_583687.kspace', '24415_327687.kspace', 
                      '24478_158727.kspace', '24555_384007.kspace', '24705_501767.kspace', 
                      '24789_665607.kspace', '24875_353287.kspace', '24892_578567.kspace']

	# Sort test cases into a list
	test_files = []
	for file in input_files:
		if os.path.split(file)[-1] in selected_files:
			test_files.append(file)
			input_files.remove(file)

	# Figure out data split (hard-code this to 65/10/25 for now)
	num_test = len(test_files)
	num_validate = int(round(0.1 * num_files))
	num_train = num_files - num_validate - num_test

	# Sort remaining cases into training and validation lists
	train_files = input_files[:num_train]
	validate_files = input_files[num_train:]

	# Print out data split summary
	print('Total datasets: %d' % num_files)
	print('  Training datasets: %d' % num_train)
	print('  Validation datasets: %d' % num_validate)
	print('  Test datasets: %d' % num_test)

	# create data directories if they don't already exist
	if not os.path.exists(args.output_path):
		os.makedirs(args.output_path)
	if not os.path.exists(os.path.join(args.output_path, 'train')):
		os.makedirs(os.path.join(args.output_path, 'train'))
	if not os.path.exists(os.path.join(args.output_path, 'validate')):
		os.makedirs(os.path.join(args.output_path, 'validate'))
	if not os.path.exists(os.path.join(args.output_path, 'test')):
		os.makedirs(os.path.join(args.output_path, 'test'))
	
	for file in input_files:
		print('Processing %s...' % os.path.split(file)[-1])

		# Load HDF5 file, read k-space and image data
		hf = h5py.File(file, 'r')
		kspace = hf['kspace_real'][()] + 1j*hf['kspace_imag'][()]
		# mask = hf['mask'][()]

		# get data dimensions
		kspace = np.transpose(kspace, axes=[1,2,0,3,4]) # remove this line later
		xres, yres, num_slices, num_echoes, num_coils = kspace.shape

		# pre-process k-space data (BART)# 
		kspace = bart.bart(1, 'fftmod 4', kspace) # de-modulate across slice
		kspace = bart.bart(1, 'fft -i 1', kspace) # inverse FFT across readout

		# crop readout (to remove edge slices with overlap)
		if args.crop_readout < 1.0:
			xres_old = xres
			xres = int(round(args.crop_readout * xres_old))
			x_from = (xres_old - xres) // 2
			x_to = x_from + xres
			kspace = kspace[x_from:x_to]

		# declare arrays
		maps   = np.zeros((xres, yres, num_slices, 1, num_coils, args.num_emaps), dtype=np.complex64)
		images = np.zeros((xres, yres, num_slices, num_echoes, args.num_emaps), dtype=np.complex64)

		for x in range(xres):
			# Process data readout pt by readout pt
			im_slice, maps_slice = process_slice(kspace[x], args)

			# Save everything into arrays
			maps[x] = maps_slice
			images[x] = im_slice

		# Determine path to new hdf5 file
		if file in train_files:
			folder = 'train'
		elif file in validate_files:
			folder = 'validate'
		else:
			folder = 'test'

		# write out new hdf5 for each echo
		for echo in range(num_echoes):
			file_new = os.path.join(args.output_path, folder, os.path.split(file)[-1])
			file_new += '.echo%d' % echo
			print(file_new)
			with h5py.File(file_new, 'w') as hf_new:
				# create datasets within HDF5
				hf_new.create_dataset('kspace', data=kspace[:,:,:,echo,:])
				hf_new.create_dataset('maps',   data=maps[:,:,:,0,:,:])
				hf_new.create_dataset('target', data=images[:,:,:,echo,:])

			if args.dbwrite:
				print('Writing out files to home folder!')
				cfl.writecfl('~/kspace', kspace)
				cfl.writecfl('~/maps', maps)
				cfl.writecfl('~/images', images)


def create_arg_parser():
	parser = argparse.ArgumentParser(description="Training script for unrolled MRI recon.")
	parser.add_argument('--input-path', type=str, required=True, help='Path to input data.')
	parser.add_argument('--output-path', type=str, required=True, help='Path to output data.')
	# Data parameters
	parser.add_argument('--data-split', nargs='+', default=[0.6, 0.2, 0.2], type=float,
                        help='Training / validation / testing splits')
	parser.add_argument('--crop-readout', type=float, default=0.8, help='Fraction of readout points to keep.')
	# ESPIRiT parameters
	parser.add_argument('--device', type=int, default=-1, help='Device on which to run ESPIRiT calibration step.')
	parser.add_argument('--num-emaps', type=int, default=1, help='Number of ESPIRiT maps.')
	parser.add_argument('--ncalib', type=int, default=16, help='Calibration region sisze.')
	parser.add_argument('--crop-value', type=float, default=0.1, help='Crop value for ESPIRiT maps.')
	# Debug parameters
	parser.add_argument('--dbwrite', action='store_true', help='Write out files for debugging.')
	return parser


if __name__ == '__main__':
	args = create_arg_parser().parse_args()
	main(args)