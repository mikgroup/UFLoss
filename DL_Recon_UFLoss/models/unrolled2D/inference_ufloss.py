"""
Script for training Unrolled 3D model.
"""

import os, sys
import logging
import random
import shutil
import time
import argparse
import numpy as np
import sigpy.plot as pl
import torch
import sigpy as sp
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt
from networks.clean_ufloss.model import Model
from tqdm import tqdm

matplotlib.use("TkAgg")
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

# import custom libraries
from utils import transforms as T
from utils import complex_utils as cplx
import networks.resnet as resnet
from networks.momentum.momentum_model import MomentumModel
from networks.momentum.network import SimpleNet

from utils.flare_utils import roll
from utils.datasets import SliceData
from unrolled2D_MoDL import UnrolledModel
from subsample_fastmri import MaskFunc


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataTransform:
    """
    Data Transformer for training unrolled reconstruction models.
    """

    def __init__(self, mask_func, args, use_seed=False):
        """
        Args:
            mask_func (utils.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
        """
        self.mask_func = mask_func
        if args.ge_mask is not None:
            self.ge_mask = args.ge_mask
        else:
            self.ge_mask = None

    def __call__(self, kspace, maps, target):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (num_coils, rows, cols) for multi-coil
                data or (rows, cols, 2) for single coil data.
            target (numpy.array): Target image.
            maps (numpy.array): Sensitivity maps for multi-channel MRI.
        Returns:
            (tuple): tuple containing:
                kspace_torch (torch.Tensor): Undersampled k-space data.
                maps (torch.Tensor): maps converted to a torch Tensor.
                target (torch.Tensor): target image to a torch Tensor.
                mask (torch.Tensor): Mask to a torch Tensor.
        """

        kspace_torch = torch.tensor(kspace).unsqueeze(0)
        maps_torch = torch.tensor(maps).unsqueeze(0)
        target_torch = torch.tensor(target).unsqueeze(0)
        if self.ge_mask is None:

            mask_slice = np.ones((640, 372))

            mk1 = self.mask_func((1, 1, 372, 2))[0, 0, :, 0]
            knee_masks = mask_slice * mk1

        else:
            mask_list = np.array(os.listdir(self.ge_mask))
            mk = np.random.choice(mask_list, 1)[0]
            knee_masks = np.load(self.ge_mask + mk)
            # print(mk, knee_masks.shape)
            # sys.exit()
        mask_torch = torch.tensor(knee_masks[None, None, ...]).float()
        kspace_torch = kspace_torch * mask_torch

        return (
            kspace_torch.squeeze(0),
            maps_torch.squeeze(0),
            target_torch.squeeze(0),
            mask_torch.squeeze(0),
        )


def create_datasets(args):
    """
    Creat pytorch Dataset for DataLoader (Train and Validate)
    """
    # Generate k-t undersampling masks
    dev_mask = MaskFunc([0.08], [5])
    dev_data = SliceData(
        root=os.path.join(str(args.data_path), "validate"),
        transform=DataTransform(dev_mask, args),
        sample_rate=1,
    )
    return dev_data


def create_data_loaders(args):
    """
    Create Data loaders
    """

    dev_data = create_datasets(args)

    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=1,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=4,
    )
    return dev_loader


def compute_metrics(args, model, data, fname):
    """
    Args:
        Reconstruction model
        data (list): contains input k-space, maps, target, and mask
        model_ufloss: model to compute ufloss (None for per-pixel loss)
    Returns:
        loss metrics.
    """
    input, maps, target, mask = data
    input = input.to(args.device)

    maps = maps.to(args.device)
    target = target.numpy()[0, ...]
    mask = mask.numpy()[0, 0, ...]

    # Forward pass through network
    model.initiate(input, maps)
    output = model.forward()
    output = output.detach().cpu().numpy()
    output = output[0, 0, ...] + 1j * output[0, 1, ...]
    output = sp.resize(output, (320, 320))
    target = sp.resize(target, (320, 320))
    mask = sp.resize(mask, (320, 368))
    im_all = np.concatenate((mask, output, target), 1)
    plt.ioff()
    plt.imshow(abs(im_all)[::-1, :], cmap="gray")
    plt.savefig(
        args.exp_dir + fname[0].split("/")[-1].split(".")[0] + ".png",
        transparent=True,
        format="png",
        bbox_inches="tight",
        pad_inches=0,
        dpi=200,
        quality=95,
    )
    plt.close()
    return None


def evaluate(args, model, data_loader):
    model.eval()

    with torch.no_grad():
        for iter, (data, fname) in tqdm(enumerate(data_loader)):
            # Compute image quality metrics
            compute_metrics(args, model, data, fname)


def build_model(args):
    print("Using MoDL for training")
    model = UnrolledModel(args).to(args.device)
    return model


def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint["args"]
    model = build_model(args)
    model.load_state_dict(checkpoint["model"])

    return checkpoint, model


def main(args):
    # Create model directory if it doesn't exist
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    # writer = SummaryWriter(log_dir=args.exp_dir)

    if int(args.device_num) > -1:
        logger.info(f"Using GPU device {args.device_num}...")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_num
        args.device = "cuda"
    else:
        logger.info("Using CPU...")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        args.device = "cpu"

    _, model = load_model(args.checkpoint)
    # args_training = checkpoint["args"]
    # del checkpoint

    # logging.info(args)
    # logging.info(model)

    torch.cuda._lazy_init()

    dev_loader = create_data_loaders(args)

    evaluate(
        args, model, dev_loader,
    )
    # writer.close()


def create_arg_parser():
    parser = argparse.ArgumentParser(
        description="Training script for unrolled MRI recon."
    )
    # Network parameters
    parser.add_argument(
        "--ge-mask", type=str, required=False, default=None, help="Path to the GE mask"
    )

    # Data parameters
    parser.add_argument(
        "--data-path", type=str, required=True, help="Path to the dataset"
    )

    parser.add_argument(
        "--device-num", type=str, default="0", help="Which device to train on."
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        default="Figures",
        help="Path where Figures should be saved",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help='Path to an existing checkpoint. Used along with "--resume"',
    )
    return parser


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    args = create_arg_parser().parse_args()
    main(args)
