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
from tqdm import tqdm
import sigpy as sp
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
from networks.clean_ufloss.model import Model

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
from unrolled2D_MoDL_error_order import UnrolledModel
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

    def augment(self, kspace, target, seed):
        """
        Apply random data augmentations. Not used in this project
        
        """
        self.rng.seed(seed)

        # Random flips through time
        if self.rng.rand() > 0.5:
            kspace = torch.flip(kspace, dims=(3,))
            target = torch.flip(target, dims=(3,))

        return kspace, target

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

        mask_slice = np.ones((640, 372))

        mk1 = self.mask_func((1, 1, 372, 2))[0, 0, :, 0]
        knee_masks = mask_slice * mk1
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
    train_mask = MaskFunc([0.08], [5])
    dev_mask = MaskFunc([0.08], [5])
    train_data = SliceData(
        root=os.path.join(str(args.data_path), "train"),
        transform=DataTransform(train_mask, args),
        sample_rate=args.sample_rate,
    )
    dev_data = SliceData(
        root=os.path.join(str(args.data_path), "validate"),
        transform=DataTransform(dev_mask, args),
        sample_rate=args.sample_rate,
    )
    return dev_data, train_data


def create_data_loaders(args):
    """
    Create Data loaders
    """

    dev_data, train_data = create_datasets(args)
    print(train_data[0])
    display_data = [dev_data[i] for i in range(0, len(dev_data), len(dev_data) // 16)]

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        prefetch_factor=4,
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        num_workers=16,
        pin_memory=True,
        prefetch_factor=4,
    )
    return train_loader, dev_loader


def compute_metrics(args, model, data, model_ufloss):
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
    target = target.to(args.device)
    mask = mask.to(args.device)

    # Forward pass through network
    model.initiate(input, maps)
    if args.meld_flag:
        output, X_cp, dim, num_emap = model.evaluate()
        output = output.detach().requires_grad_(True)

    else:
        if args.error:
            output, error_est = model.forward()
        else:
            output = model.forward()
        X_cp, dim, num_emap = None, None, None

    #     pl.ImagePlot(output.detach().cpu().numpy())
    # Undo normalization from pre-processing
    if args.loss_normalized == False:
        output = output * std + mean
        target = target * std + mean
    ufloss = None
    if args.loss_uflossdir is not None:
        target = torch.cat((target.real[:, None, ...], target.imag[:, None, ...]), 1)
        if args.efficient_ufloss:
            # Using efficient momentum-based method to compute UFLoss
            n_featuresq = args.uflossfreq
            ix = torch.randint(0, n_featuresq, (1,))
            iy = torch.randint(0, n_featuresq, (1,))
            output_roll = roll(output.clone(), ix, iy)
            target_roll = roll(target.clone(), ix, iy)
            ufloss = nn.MSELoss()(
                model_ufloss(output_roll)[0], model_ufloss(target_roll)[0]
            )

        else:
            # Using traditional method to compute UFLoss
            n_featuresq = args.uflossfreq
            ix = torch.randint(0, n_featuresq, (1,))
            iy = torch.randint(0, n_featuresq, (1,))
            output_roll = roll(output.clone(), ix, iy)
            target_roll = roll(target.clone(), ix, iy)
            #             print(output_roll.shape)
            arraytoblock = sp.linop.ArrayToBlocks(
                ishape=list(
                    (
                        output_roll.shape[0],
                        2,
                        output_roll.shape[2],
                        output_roll.shape[3],
                    )
                ),
                blk_shape=list((output_roll.shape[0], 2, 60, 60)),
                blk_strides=list((1, 1, n_featuresq, n_featuresq)),
            )

            reshape = sp.linop.Reshape(
                ishape=arraytoblock.oshape,
                oshape=(arraytoblock.oshape[2] * arraytoblock.oshape[3], 2, 60, 60),
            )

            Fa2b = sp.to_pytorch_function(reshape * arraytoblock).apply
            output_patch = Fa2b(output_roll)
            target_patch = Fa2b(target_roll)

            output_features = model_ufloss(output_patch)
            target_features = model_ufloss(target_patch)
            ufloss = nn.MSELoss()(output_features[0], target_features[0])

    else:
        target = torch.cat((target.real[:, None, ...], target.imag[:, None, ...]), 1)

    target = target.permute(0, 2, 3, 1)
    output = output.permute(0, 2, 3, 1)
    error_loss = None
    if args.error:
        error_est = error_est.permute(0, 2, 3, 1)
        error_gt = abs(target - output.detach())
        error_loss = nn.L1Loss()(error_est, error_gt)
    #     pl.ImagePlot(output.detach().cpu().numpy())
    scale = cplx.abs(target).max()

    # Compute image quality metrics from complex-valued images
    # cplx_error = cplx.abs(output - target)
    # cplx_l1 = torch.mean(cplx_error)
    cplx_l1 = nn.L1Loss()(output, target)
    cplx_l2 = nn.MSELoss()(output, target)
    cplx_psnr = 20 * torch.log10(scale / cplx_l2)

    # Compute image quality metrics from magnitude images
    mag_error = torch.abs(cplx.abs(output) - cplx.abs(target))
    mag_l1 = torch.mean(mag_error)
    mag_l2 = torch.sqrt(torch.mean(mag_error ** 2))
    mag_psnr = 20 * torch.log10(scale / mag_l2)

    return (
        cplx_l1,
        cplx_l2,
        cplx_psnr,
        mag_psnr,
        ufloss,
        output,
        X_cp,
        dim,
        num_emap,
        error_loss,
    )


def train_epoch(args, epoch, model, data_loader, optimizer, writer, model_ufloss):
    model.train()
    avg_l2 = None
    avg_ufloss = None
    avg_loss = 0.0
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    print(len(data_loader))
    for iter, data in enumerate(data_loader):
        # Compute image quality metrics
        (
            l1_loss,
            l2_loss,
            cpsnr,
            mpsnr,
            ufloss,
            output,
            X_cp,
            dim,
            num_emap,
            error_loss,
        ) = compute_metrics(args, model, data, model_ufloss)
        if ufloss is not None:
            ufloss = args.ufloss_weight * ufloss

        if args.loss_type == 1:
            loss = l1_loss.clone()
        else:
            loss = l2_loss.clone()
        if ufloss is not None:
            loss = loss + ufloss
        if error_loss is not None:
            loss = loss + error_loss
        optimizer.zero_grad()
        loss.backward()
        if args.meld_flag:
            qN = output.grad
            for p_ in model.parameters():
                p_.requires_grad_(True)
            qk = qN
            for ii in range(len(model.resnets) - 1, -1, -1):
                xkm1 = X_cp[ii, ...]
                xkm1 = xkm1.detach().requires_grad_(True)
                xk = xkm1
                xk = xk.reshape(dim[0:4] + (num_emap * 2,)).permute(0, 4, 3, 2, 1)
                xk = model.resnets[ii][0](xk)
                xk = xk.permute(0, 4, 3, 2, 1).reshape(dim[0:4] + (num_emap, 2))
                xk = model.resnets[ii][1](xk)
                xk.backward(qk, retain_graph=True)
                with torch.no_grad():
                    qk = xkm1.grad

        optimizer.step()

        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
        if error_loss is not None:
            avg_l1 = (
                0.99 * avg_l1 + 0.01 * l1_loss.item() if iter > 0 else l1_loss.item()
            )
            avg_loss_error = (
                0.99 * avg_loss_error + 0.01 * error_loss.item()
                if iter > 0
                else error_loss.item()
            )

        if ufloss is not None:
            avg_l2 = (
                0.99 * avg_l2 + 0.01 * l2_loss.item() if iter > 0 else l2_loss.item()
            )

            avg_ufloss = (
                0.99 * avg_ufloss + 0.01 * ufloss.item() if iter > 0 else ufloss.item()
            )

        # Write out summary
        writer.add_scalar("Train_Loss", loss.item(), global_step + iter)
        writer.add_scalar("Train_cPSNR", cpsnr.item(), global_step + iter)
        writer.add_scalar("Train_mPSNR", mpsnr.item(), global_step + iter)
        if error_loss is not None:
            writer.add_scalar(
                "Train_error_map_loss", error_loss.item(), global_step + iter
            )

        if ufloss is not None:
            writer.add_scalar("Train_L2loss", l2_loss.item(), global_step + iter)
            writer.add_scalar("Train_UFLoss", ufloss.item(), global_step + iter)

        if iter % 200 == 0:
            save_model(args, args.exp_dir, epoch, model, optimizer, 10231, False)

        if iter % args.report_interval == 0:
            if ufloss is not None:
                logging.info(
                    f"Epoch = [{epoch:3d}/{args.num_epochs:3d}] "
                    f"Iter = [{iter:4d}/{len(data_loader):4d}] "
                    f"Loss = {loss.item():.4g} L2 Loss = {l2_loss.item():.4g} UFLoss = {ufloss.item():.4g} Avg Loss = {avg_loss:.4g} Avg L2 Loss = {avg_l2:.4g} Avg UFLoss = {avg_ufloss:.4g} "
                    f"Time = {time.perf_counter() - start_iter:.4f}s",
                )
            else:
                if error_loss is not None:
                    logging.info(
                        f"Epoch = [{epoch:3d}/{args.num_epochs:3d}] "
                        f"Iter = [{iter:4d}/{len(data_loader):4d}] "
                        f"Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g} L1Loss = {l1_loss.item():.4g} Avg L1Loss = {avg_l1:.4g} ErrorLoss = {error_loss.item():.4g} Avg ErrorLoss = {avg_loss_error:.4g}"
                        f"Time = {time.perf_counter() - start_iter:.4f}s",
                    )
                else:
                    logging.info(
                        f"Epoch = [{epoch:3d}/{args.num_epochs:3d}] "
                        f"Iter = [{iter:4d}/{len(data_loader):4d}] "
                        f"Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g} "
                        f"Time = {time.perf_counter() - start_iter:.4f}s",
                    )

        start_iter = time.perf_counter()

    return avg_loss, avg_l2, avg_ufloss, time.perf_counter() - start_epoch


def evaluate(args, epoch, model, data_loader, writer, model_ufloss):
    model.eval()
    losses = []
    cpsnr_vals = []
    mpsnr_vals = []
    l2_vals = []
    ufloss_vals = []
    start = time.perf_counter()
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):
            # Compute image quality metrics
            l1_loss, l2_loss, cpsnr, mpsnr, ufloss, _, _, _, _, _ = compute_metrics(
                args, model, data, model_ufloss
            )
            if ufloss is not None:
                ufloss = args.ufloss_weight * ufloss

            if args.loss_type == 1:
                loss = l1_loss.clone()
            else:
                loss = l2_loss.clone()
            if ufloss is not None:
                loss = loss + ufloss

            losses.append(loss.item())
            cpsnr_vals.append(cpsnr.item())
            mpsnr_vals.append(mpsnr.item())
            if ufloss is not None:
                l2_vals.append(l2_loss.item())
                ufloss_vals.append(ufloss.item())

        writer.add_scalar("Val_Loss", np.mean(losses), epoch)
        writer.add_scalar("Val_cPSNR", np.mean(cpsnr_vals), epoch)
        writer.add_scalar("Val_mPSNR", np.mean(mpsnr_vals), epoch)
        if ufloss is not None:
            writer.add_scalar("Val_L2loss", np.mean(l2_vals), epoch)
            writer.add_scalar("Val_UFLoss", np.mean(ufloss_vals), epoch)
    return (
        np.mean(losses),
        np.mean(l2_vals),
        np.mean(ufloss_vals),
        time.perf_counter() - start,
    )


def visualize(args, epoch, model, data_loader, writer, is_training=True):
    """
    Not used in this project
    """

    def save_image(image, tag):
        image = image.permute(0, 3, 1, 2)
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=1, pad_value=1)
        writer.add_image(tag, grid, epoch)

    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            # Load all data arrays
            input, maps, init, target, mean, std, norm = data
            input = input.to(args.device)
            maps = maps.to(args.device)
            init = init.to(args.device)
            target = target.to(args.device)

            # Initialize signal model
            A = T.SenseModel(maps)

            # Compute DL recon
            output = model(input, maps, init_image=init)

            # Slice images
            init = init[:, :, :, 10, 0, None]
            output = output[:, :, :, 10, 0, None]
            target = target[:, :, :, 10, 0, None]
            mask = cplx.get_mask(input[:, -1, :, :, 0, :])  # [b, y, t, 2]

            # Save images to summary
            tag = "Train" if is_training else "Val"
            all_images = torch.cat((init, output, target), dim=2)
            save_image(cplx.abs(all_images), "%s_Images" % tag)
            save_image(cplx.angle(all_images), "%s_Phase" % tag)
            save_image(cplx.abs(output - target), "%s_Error" % tag)
            save_image(mask.permute(0, 2, 1, 3), "%s_Mask" % tag)

            break


def save_model(args, exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best):
    torch.save(
        {
            "epoch": epoch,
            "args": args,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_dev_loss": best_dev_loss,
            "exp_dir": exp_dir,
        },
        f=os.path.join(exp_dir, "model_epoch%d.pt" % (epoch)),
    )
    if is_new_best:
        shutil.copyfile(
            os.path.join(exp_dir, "model_epoch%d.pt" % (epoch)),
            os.path.join(exp_dir, "best_model.pt"),
        )


def build_model(args):
    print("Using MoDL for training")
    model = UnrolledModel(args).to(args.device)
    return model


def build_split_model(args):
    # Do not use yet - work in progress
    num_gpus = torch.cuda.device_count()
    models = [None] * num_gpus
    for i in range(num_gpus):
        device_name = torch.cuda.get_device_name(i)
        print("Initializing network %d on GPU%d (%s)" % (i, i, device_name))
        models[i] = UnrolledModel(args_device).to("cuda:%d" % i)
    return nn.ModuleList(models)


def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint["args"]
    model = build_model(args)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint["model"])

    optimizer = build_optim(args, model.parameters())
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint, model, optimizer


def build_optim(args, params):
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    return optimizer


def main(args):
    # Create model directory if it doesn't exist
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    writer = SummaryWriter(log_dir=args.exp_dir)

    if int(args.device_num) > -1:
        logger.info(f"Using GPU device {args.device_num}...")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_num
        args.device = "cuda"
    else:
        logger.info("Using CPU...")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        args.device = "cpu"

    if args.resume:
        checkpoint, model, optimizer = load_model(args.checkpoint)
        args = checkpoint["args"]
        best_dev_loss = checkpoint["best_dev_loss"]
        start_epoch = checkpoint["epoch"]
        del checkpoint
    else:
        model = build_model(args)
        if args.data_parallel:
            print(torch.cuda.device_count())
            model = torch.nn.DataParallel(model, device_ids=[2, 3])
        optimizer = build_optim(args, model.parameters())
        best_dev_loss = 1e9
        start_epoch = 0
    model_ufloss = None
    if args.loss_uflossdir is not None:
        print(args)
        # Add UFLoss for 2D
        if args.efficient_ufloss:
            model_ufloss = MomentumModel(SimpleNet)
            model_ufloss.load_state_dict(
                torch.load(args.loss_uflossdir, "cpu")["state_dict"]
            )
            model_ufloss = model_ufloss.target_network
            model_ufloss.to(args.device)
            model_ufloss.eval()
            model_ufloss.requires_grad_ = False
            print("Successfully loaded UFLoss model (Efficient)")

        else:

            # model_ufloss = resnet.resnet18(num_classes=args.num_features).to(
            #     args.device
            # )
            # model_ufloss.load_state_dict(
            #     (torch.load(args.loss_uflossdir, map_location=args.device))
            # )
            # model_ufloss.requires_grad_ = False

            model_re = Model(
                resnet.resnet18, feature_dim=args.num_features, data_length=2457600
            )

            model_re.load_state_dict(
                torch.load(args.loss_uflossdir, "cpu",)["state_dict"]
            )
            model_ufloss = model_re.network.to(args.device)
            model_ufloss.requires_grad_ = False

            print("Successfully loaded UFLoss model (Traditional)")

    logging.info(args)
    logging.info(model)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    torch.cuda._lazy_init()

    train_loader, dev_loader = create_data_loaders(args)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, args.lr_step_size, args.lr_gamma
    )

    for epoch in range(start_epoch, args.num_epochs):
        if args.loss_uflossdir is not None:
            train_loss, train_l2, train_ufloss, train_time = train_epoch(
                args, epoch, model, train_loader, optimizer, writer, model_ufloss
            )
            dev_loss, dev_l2, dev_ufloss, dev_time = evaluate(
                args, epoch, model, dev_loader, writer, model_ufloss
            )
        else:
            train_loss, _, _, train_time = train_epoch(
                args, epoch, model, train_loader, optimizer, writer, model_ufloss
            )
            dev_loss, _, _, dev_time = evaluate(
                args, epoch, model, dev_loader, writer, model_ufloss
            )
        scheduler.step(epoch)

        is_new_best = dev_loss < best_dev_loss
        best_dev_loss = min(best_dev_loss, dev_loss)
        save_model(
            args, args.exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best
        )
        if args.loss_uflossdir is not None:
            logging.info(
                f"Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} TrainL2Loss = {train_l2:.4g} TrainUFLoss = {train_ufloss:.4g}"
                f"DevLoss = {dev_loss:.4g} DevL2Loss = {dev_l2:.4g} DevUFLoss = {dev_ufloss:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s",
            )
        else:
            logging.info(
                f"Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} "
                f"DevLoss = {dev_loss:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s",
            )
    writer.close()


def create_arg_parser():
    parser = argparse.ArgumentParser(
        description="Training script for unrolled MRI recon."
    )
    # Network parameters
    parser.add_argument(
        "--num-grad-steps", type=int, default=5, help="Number of unrolled iterations"
    )
    parser.add_argument(
        "--num-cg-steps", type=int, default=6, help="Number of CG Steps"
    )
    parser.add_argument(
        "--num-resblocks",
        type=int,
        default=2,
        help="Number of ResNet blocks per iteration",
    )
    parser.add_argument(
        "--num-features", type=int, default=128, help="Number of ResNet channels"
    )
    parser.add_argument(
        "--kernel-size", type=int, default=3, help="Convolution kernel size"
    )
    parser.add_argument(
        "--drop-prob", type=float, default=0.0, help="Dropout probability"
    )
    parser.add_argument("--modl-lamda", type=float, default=0.05, help="MoDL-lamda")
    parser.add_argument(
        "--modl-flag", type=bool, default=False, help="Whether using MoDL"
    )
    parser.add_argument(
        "--meld-flag",
        action="store_true",
        help="If set, will use memory efficient learning.",
    )
    parser.add_argument(
        "--meld-cp",
        action="store_true",
        help="If set, will use checkpoints memory efficient learning.",
    )
    parser.add_argument(
        "--ufloss3d",
        action="store_true",
        help="If set, will use 3d ufloss instead of 2+1d.",
    )

    parser.add_argument(
        "--fix-step-size", type=bool, default=True, help="Fix unrolled step size"
    )
    parser.add_argument(
        "--circular-pad",
        type=bool,
        default=True,
        help="Flag to turn on circular padding",
    )
    parser.add_argument(
        "--slwin-init",
        action="store_true",
        help="If set, will use sliding window initialization.",
    )
    parser.add_argument(
        "--share-weights",
        action="store_true",
        default=False,
        help="If set, will use share weights between unrolled iterations.",
    )

    # Data parameters
    parser.add_argument(
        "--data-path", type=str, required=True, help="Path to the dataset"
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=1.0,
        help="Fraction of total volumes to include",
    )
    parser.add_argument(
        "--patch-size", default=64, type=int, help="Resolution of images"
    )
    parser.add_argument(
        "--num-emaps", type=int, default=1, help="Number of ESPIRiT maps"
    )

    # Undersampling parameters
    parser.add_argument(
        "--accelerations",
        nargs="+",
        default=[10, 15],
        type=int,
        help="Range of acceleration rates to simulate in training data.",
    )

    # Training parameters
    parser.add_argument("--batch-size", default=1, type=int, help="Mini batch size")
    parser.add_argument(
        "--num-epochs", type=int, default=2000, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--lr-step-size", type=int, default=20, help="Period of learning rate decay"
    )
    parser.add_argument(
        "--lr-gamma",
        type=float,
        default=0.5,  # 0.1
        help="Multiplicative factor of learning rate decay",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Strength of weight decay regularization",
    )

    # Miscellaneous parameters
    parser.add_argument(
        "--seed", default=42, type=int, help="Seed for random number generators"
    )
    parser.add_argument(
        "--report-interval", type=int, default=10, help="Period of loss reporting"
    )
    parser.add_argument(
        "--data-parallel",
        action="store_true",
        help="If set, use multiple GPUs using data parallelism",
    )
    parser.add_argument(
        "--device-num", type=str, default="0", help="Which device to train on."
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        default="checkpoints",
        help="Path where model and results should be saved",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If set, resume the training from a previous model checkpoint. "
        '"--checkpoint" should be set with this',
    )
    parser.add_argument(
        "--error",
        action="store_true",
        help="If set, we will output an error map estimation",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help='Path to an existing checkpoint. Used along with "--resume"',
    )
    parser.add_argument(
        "--efficient-ufloss",
        action="store_true",
        default=False,
        help="If set, will use momentum-based efficient UFLoss",
    )
    # For UFLoss
    parser.add_argument(
        "--loss-normalized",
        default=False,
        help="If set, will compute the loss function on the normalized data",
    )
    parser.add_argument("--loss-type", type=int, default=1, help="loss type (1 or 2)")
    parser.add_argument("--uflossfreq", type=int, default=10, help="ufloss frequency")

    parser.add_argument(
        "--loss-uflossdir",
        type=str,
        default=None,
        help="Path to the UFLoss mapping network",
    )
    parser.add_argument(
        "--ufloss-weight", type=float, default=2, help="Weight of the UFLoss"
    )

    return parser


if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)
    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
