from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset


class UFData(Dataset):

    def __init__(self, data_directory, max_offset=None, magnitude=False, device=torch.device('cpu'),
                 fastmri=False, complex=False):
        """

        Parameters
        ----------
        data_directory : str
            The directory containing the training npy data.
        max_offset : tuple
            The maximum offset to crop an image to.
        magnitude : bool
            If True, train using magnitude image as input. Otherwise, use real and imaginary image in separate channels.
        device : torch.device
            The device to load the data to.
        complex : bool
            If True, return images as complex data. Otherwise check for magnitude return or for real and imaginary
            channels. This is needed when training, since post processing is done in the model (adds phase augmentation
            and converts to magnitude or channels). Magnitude and channels are implemented for evaluation.
        """
        if max_offset is None:
            if fastmri:
                max_offset = (434, 50)  # FastMRI dataset to make same size of 3D dataset slices
                # max_offset = (200, 40)  # FastMRI dataset (biggest is around 386 and smallest is 320)
            else:
                max_offset = (50, 50)  # 3D Dataset
                # max_offset = (216, 280)  # 40 by 40 patch in original dataset of 256x320

        self.image_paths = glob(f"{data_directory}/*.npy")
        print(f"Using data from: {data_directory}\nFound {len(self.image_paths)} image paths.")
        self.device = device
        self.magnitude = magnitude
        self.complex = complex

        if fastmri:  # Fast MRI Dataset:
            self.cropped_image_size = np.array([640, 320]) - max_offset
        else:  # Original mri.org Dataset:
            self.cropped_image_size = np.array(np.load(self.image_paths[0]).shape[-2:]) - max_offset
        # self.cropped_image_size = np.array([47, 47])  # Just the image size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        """Get image at the specified index. Two cropped versions of the images will be returned: one with the previous
        cropping offset (and possibly other augmentation settings), and a new one with a new cropping offset. The new
        cropping offset will be stored for the next time this image is accessed.

        Parameters
        ----------
        index : int
            The image index.

        Returns
        -------
        previous_image : torch.Tensor
            Image cropped (augmented) at previous settings.
        new_image : torch.Tensor
            Image cropped (augmented) at new settings.
        """
        original_image = np.load(self.image_paths[index])[None]

        # +1 since random doesn't include max. i.e [a, b).
        offset = np.random.randint(0, original_image.shape[1:] - self.cropped_image_size + 1)
        original_image = self.crop(original_image, offset)

        if self.complex:
            return original_image
        elif self.magnitude:
            return torch.tensor(np.abs(original_image))
        else:
            return torch.tensor(np.concatenate((original_image.real, original_image.imag), axis=0))

    def crop(self, image, offset):
        """Crop image(s) to `self.cropped_image_size` starting at the specified offset.

        Parameters
        ----------
        image : torch.Tensor
            The image to crop of shape (C, H, W)
        offset : np.array
            The offset to add of shape (2,)

        Returns
        -------
        torch.Tensor
            The cropped image of size `self.cropped_image_size`. Shape (C, H, W).
        """
        stop = offset + self.cropped_image_size
        return image[:, offset[0]:stop[0], offset[1]:stop[1]]
