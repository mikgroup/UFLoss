"""
Reformat fastMRI single coil dataset
"""

import os, sys
import glob
import h5py
import numpy as np


def main():
    # ARGS
    input_data_path = '/mikQNAP/NYU_knee_data/singlecoil_val/'
    output_data_path = '/mikQNAP/frank/dl-cs-torch/fastMRI-singlecoil/validate'

    input_files = glob.glob(os.path.join(input_data_path, '*.h5'))

    for file in input_files:
        # Load HDF5 file
        hf = h5py.File(file, 'r')
        # existing keys: ['ismrmrd_header', 'kspace', 'reconstruction_esc']

        # load image data from HDF5 file
        im_esc = hf['reconstruction_esc']
        kspace = np.fft.ifftshift(im_esc, axes=(-1, -2))
        kspace = np.fft.fftn(kspace, axes=(-1, -2))
        kspace = np.fft.fftshift(kspace, axes=(-1, -2))
        kspace = np.expand_dims(kspace, -1)  # [33, 320, 320, 1]
        kspace = kspace.astype(np.complex64)
        maps = np.ones(kspace.shape + (1, ), dtype=np.complex64)  # [33, 320, 320, 1, 1]
        im_truth = np.expand_dims(im_esc, -1)

        # write out new hdf5
        file_new = os.path.join(output_data_path, os.path.split(file)[-1])
        with h5py.File(file_new, 'w') as hf_new:
            # create datasets within HDF5
            hf_new.create_dataset('kspace', data=kspace)
            hf_new.create_dataset('maps', data=maps)
            hf_new.create_dataset('reconstruction_espirit', data=im_truth)
            hf_new.create_dataset('ismrmrd_header', data=hf['ismrmrd_header'])

            # create attributes (metadata)
            for key in hf.attrs.keys():
                hf_new.attrs[key] = hf.attrs[key]


if __name__ == '__main__':
    main()
