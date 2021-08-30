"""
Script for reconstructing multi-slice data with a trained 3D unrolled model.
"""

import os
import sys
import time
import torch
import logging
import argparse
import numpy as np

# import custom libraries
from utils import cfl
from utils import transforms as T
from utils import complex_utils as cplx

# import custom classes
from models.unrolled3D import UnrolledModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_and_args(checkpoint_file, device):

    checkpoint = torch.load(checkpoint_file)
    args = checkpoint["args"]
    model = UnrolledModel(args).to(device)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint["model"])
    return model, args


def preprocess(kspace, maps, args):
    # Batch size dimension must be the same!
    assert kspace.shape[0] == maps.shape[0]
    batch_size = kspace.shape[0]

    # Convert everything from numpy arrays to tensors
    kspace = cplx.to_tensor(kspace)
    maps = cplx.to_tensor(maps)

    # Initialize ESPIRiT model
    A = T.SenseModel(maps)

    # Compute normalization factor (based on 95% max signal level in view-shared dataa)
    averaged_kspace = T.time_average(kspace, dim=3)
    image = A(averaged_kspace, adjoint=True)
    magnitude_vals = cplx.abs(image).reshape(batch_size, -1)
    k = int(round(0.05 * magnitude_vals[0].numel()))
    scale = torch.min(torch.topk(magnitude_vals, k, dim=1).values, dim=1).values

    # Normalize k-space data
    kspace /= scale[:, None, None, None, None, None]

    # Compute network initialization
    if args.slwin_init:
        init_image = A(T.sliding_window(kspace, dim=3, window_size=5), adjoint=True)
    else:
        init_image = A(masked_kspace, adjoint=True)

    return kspace.unsqueeze(1), maps.unsqueeze(1), init_image.unsqueeze(1)


def main(args):
    start = time.time()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Figure out multi-GPU stuff
    if args.device > -1:
        logger.info(f"Using GPU device(s) {args.device}...")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        device = f"cuda:{args.device}"
    else:
        logger.info("Using CPU...")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = "cpu"

    logger.info("Loading DL Cine model...")
    model, model_args = load_model_and_args(args.checkpoint, args.device)

    # get filenames
    file_kspace = os.path.join(args.directory, args.kspace)
    file_maps = os.path.join(args.directory, args.maps)
    file_output = os.path.join(args.directory, args.output)

    logger.info("Reading input data...")
    kspace = cfl.read(file_kspace, order="F")
    maps = cfl.read(file_maps, order="F")

    # get data dimensions
    nx, ny, nslices, ncoils, _, nechoes, _, nphases = kspace.shape
    nmaps = maps.shape[4]

    # data summary
    logger.info("Detected data dimensions...")
    logger.info("  X (readout):   %d" % nx)
    logger.info("  Y (PE):        %d" % ny)

    logger.info("  Z (slices):    %d" % nslices)
    logger.info("  T (phases):    %d" % nphases)
    logger.info("  E (echoes):    %d" % nechoes)
    logger.info("  C (coils):     %d" % ncoils)
    logger.info("  M (ESPIRiT):   %d" % nmaps)

    # re-shape
    kspace = np.reshape(kspace, [nx, ny, nslices, ncoils, nechoes, nphases])
    # [sl, ec, x, y, t, coils]
    kspace = np.transpose(kspace, [2, 4, 0, 1, 5, 3])
    maps = np.reshape(maps, [nx, ny, nslices, ncoils, nmaps, 1, 1])
    # [sl, 1, x, y, 1, coils, maps]
    maps = np.transpose(maps, [2, 5, 0, 1, 6, 3, 4])

    # for now, just repeat maps across the echo dimension
    maps = np.tile(maps, [1, nechoes, 1, 1, 1, 1, 1])

    # flatten slice/echo dimension into a batch dimension
    kspace = np.reshape(kspace, [nslices * nechoes, nx, ny, nphases, ncoils])
    maps = np.reshape(maps, [nslices * nechoes, nx, ny, 1, ncoils, nmaps])

    logger.info("Pre-processing data...")
    kspace, maps, init = preprocess(kspace, maps, model_args)

    # Put all arrays on GPU.
    kspace = kspace.to(device)
    maps = maps.to(device)
    init = init.to(device)

    # Allocate memory for output array.
    images = torch.zeros(
        (nslices * nechoes, nx, ny, nphases, nmaps, 2), dtype=torch.float32
    )

    # Keep track of recon time
    recon_time = 0.0

    logger.info("Begin reconstruction.")
    with torch.no_grad():
        for i in range(nslices * nechoes):
            logger.info("  slice %d/%d" % (i + 1, nslices * nechoes))

            # Reconstruct on GPU.
            slice_start = time.time()
            images[i] = model(kspace[i], maps[i], init[i])
            slice_end = time.time()

            # Move image data to CPU.
            images[i] = images[i].squeeze(0).to("cpu")

            # Add to recon timer.
            recon_time += slice_end - slice_start

    images = cplx.to_numpy(images)
    images = np.reshape(images, [nslices, nechoes, nx, ny, nphases, nmaps])
    images = np.transpose(images, [2, 3, 0, 5, 1, 4])
    cfl.write(file_output, images[:, :, :, None, :, :, :], order="F")

    # Print summary.
    logger.info("Reconstruction time: %0.2f" % (recon_time))
    logger.info("Total elapsed time: %0.2f" % (time.time() - start))


def create_arg_parser():
    parser = argparse.ArgumentParser(
        description="Evaluation script for unrolled MRI recon."
    )
    parser.add_argument(
        "--directory", type=str, required=True, help="Path to working directory"
    )
    parser.add_argument(
        "--kspace", type=str, default="ks", help="Filename of k-space CFL data"
    )
    parser.add_argument(
        "--maps", type=str, default="maps", help="Filename of sensitivity map CFL data"
    )
    parser.add_argument(
        "--output", type=str, default="images", help="Filename of output image CFL data"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the Unrolled model"
    )
    parser.add_argument("--batch-size", default=4, type=int, help="Mini-batch size")
    parser.add_argument("--device", type=int, default=0, help="Which device to run on")
    parser.add_argument(
        "--data-parallel",
        action="store_true",
        help="If set, will allow for GPU parallelization.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="If set, will turn on verbose mode."
    )
    return parser


if __name__ == "__main__":
    args = create_arg_parser().parse_args(sys.argv[1:])
    main(args)
