"""

"""

import os, sys
import logging
import random
import shutil
import time
import argparse
import numpy as np
import sigpy.mri

import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader

# import custom libraries
from utils import transforms as T
from utils import subsample as ss
from utils import complex_utils as cplx

# import custom classes
from utils.datasets import SliceData
from utils.subsample import MaskFunc
from models.unrolled import UnrolledModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataTransform:
    """
    Data Transformer for training unrolled reconstruction models.
    """

    def __init__(self, mask_func, resolution, use_seed=True):
        """
        Args:
            mask_func (utils.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
            resolution (int): Resolution of the image.
            use_seed (bool): If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """
        self.mask_func = mask_func
        self.resolution = resolution
        self.use_seed = use_seed

    def __call__(self, kspace, maps, target, attrs, fname, slice):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (num_coils, rows, cols, 2) for multi-coil
                data or (rows, cols, 2) for single coil data.
            target (numpy.array): Target image
            attrs (dict): Acquisition related information stored in the HDF5 object.
            fname (str): File name
            slice (int): Serial number of the slice.
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Zero-filled input image.
                target (torch.Tensor): Target image converted to a torch Tensor.
                mean (float): Mean value used for normalization.
                std (float): Standard deviation value used for normalization.
                norm (float): L2 norm of the entire volume.
        """
        # Convert everything from numpy arrays to tensors
        kspace = cplx.to_tensor(kspace).unsqueeze(0)
        maps   = cplx.to_tensor(maps).unsqueeze(0)
        target = cplx.to_tensor(target).unsqueeze(0)
        norm = torch.sqrt(torch.mean(cplx.abs(target)**2))

        # squash dimensions
        kspace = kspace[:,:,:,0,:,:]
        maps = maps[:,:,:,0,:,:,:]
        target = target[:,:,:,0,:,:]

        print(kspace.size())
        print(maps.size())
        print(target.size())

        # Apply mask in k-space
        seed = None if not self.use_seed else tuple(map(ord, fname))
        mask = sigpy.mri.poisson(list(kspace.size())[1:3], 4,
                calib=[16, 16],
                dtype=np.complex64,
                seed=np.random.seed(seed))
        mask = cplx.to_tensor(mask)
        print(mask.size())
        masked_kspace *= mask[None,:,:,None,:]

        # Normalize data...
        if 1:
            # ... by magnitude of zero-filled reconstruction
            A = T.SenseModel(maps, weights=mask)
            image = A(masked_kspace, adjoint=True)
            magnitude_vals = cplx.abs(image).reshape(-1)
            k = int(round(0.05 * magnitude_vals.numel()))
            scale = torch.min(torch.topk(magnitude_vals, k).values)
        else:
            # ... by power within calibration region
            calib_size = 10
            calib_region = cplx.center_crop(masked_kspace, [calib_size, calib_size])
            scale = torch.mean(cplx.abs(calib_region)**2)
            scale = scale * (calib_size**2 / kspace.size(-3) / kspace.size(-2))

        masked_kspace /= scale
        target /= scale
        mean = torch.tensor([0.0], dtype=torch.float32)
        std = scale
        
        # Get rid of batch dimension...
        masked_kspace = masked_kspace.squeeze(0)
        maps = maps.squeeze(0)
        target = target.squeeze(0)

        return masked_kspace, maps, target, mean, std, norm


def create_datasets(args):
    train_mask = MaskFunc(args.center_fractions, args.accelerations)
    dev_mask = MaskFunc(args.center_fractions, args.accelerations)

    train_data = SliceData(
        root=os.path.join(str(args.data_path), 'train'),
        transform=DataTransform(train_mask, args.resolution),
        sample_rate=args.sample_rate
    )
    dev_data = SliceData(
        root=os.path.join(str(args.data_path), 'validate'),
        transform=DataTransform(dev_mask, args.resolution, use_seed=True),
        sample_rate=args.sample_rate
    )
    return dev_data, train_data


def create_data_loaders(args):
    dev_data, train_data = create_datasets(args)
    display_data = [dev_data[i] for i in range(0, len(dev_data), len(dev_data) // 16)]

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
    )
    display_loader = DataLoader(
        dataset=display_data,
        batch_size=16,
        num_workers=8,
        pin_memory=True,
    )
    return train_loader, dev_loader, display_loader


def compute_metrics(args, model, data):
    # Load input, sensitivity maps, and target images onto device
    input, maps, target, mean, std, norm = data
    input = input.to(args.device)
    maps = maps.to(args.device)
    target = target.to(args.device)
    mean = mean.to(args.device)
    std = std.to(args.device)
    # Forward pass through network
    output = model(input, maps)
    # Undo normalization from pre-processing
    output = output*std + mean
    target = target*std + mean
    # Compute metrics
    abs_error = cplx.abs(output - target)
    l1 = torch.mean(abs_error)
    l2 = torch.sqrt(torch.mean(abs_error**2))
    psnr = 20 * torch.log10(cplx.abs(target).max() / l2)
    return l1, l2, psnr


def train_epoch(args, epoch, model, data_loader, optimizer, writer):
    model.train()
    avg_loss = 0.
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    for iter, data in enumerate(data_loader):
        # Compute image quality metrics
        l1_loss, l2_loss, psnr = compute_metrics(args, model, data)

        # Choose loss function, then run backprop
        loss = l1_loss 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
        writer.add_scalar('Train_Loss', loss.item(), global_step + iter)
        writer.add_scalar('Train_PSNR', psnr.item(), global_step + iter)

        if iter % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()
    return avg_loss, time.perf_counter() - start_epoch


def evaluate(args, epoch, model, data_loader, writer):
    model.eval()
    losses = []
    psnr_vals = []
    start = time.perf_counter()
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            # Compute image quality metrics
            l1_loss, l2_loss, psnr = compute_metrics(args, model, data)

            losses.append(l1_loss.item())
            psnr_vals.append(psnr.item())

        writer.add_scalar('Dev_Loss', np.mean(losses), epoch)
        writer.add_scalar('Dev_PSNR', np.mean(psnr_vals), epoch)

    return np.mean(losses), time.perf_counter() - start


def visualize(args, epoch, model, data_loader, writer):
    def save_image(image, tag):
        image = cplx.abs(image).permute(0,3,1,2) # magnitude
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            # Load all data arrays
            input, maps, target, mean, std, norm = data
            input = input.to(args.device)
            maps = maps.to(args.device)
            target = target.to(args.device)

            # Compute zero-filled recon
            A = T.SenseModel(maps)
            zf = A(input, adjoint=True)

            # Compute DL recon
            output = model(input, maps)

            save_image(zf,  'Input')
            save_image(target, 'Target')
            save_image(output, 'Reconstruction')
            save_image(target - output, 'Error')
            break


def save_model(args, exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir': exp_dir
        },
        f=os.path.join(exp_dir, 'model.pt')
    )
    if is_new_best:
        shutil.copyfile(os.path.join(exp_dir, 'model.pt'), 
                        os.path.join(exp_dir, 'best_model.pt'))


def build_model(args):
    model = UnrolledModel(args).to(args.device)
    return model


def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = build_model(args)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])

    optimizer = build_optim(args, model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint, model, optimizer


def build_optim(args, params):
    # optimizer = torch.optim.RMSprop(params, lr=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)
    return optimizer


def main(args):
    # Create model directory if it doesn't exist
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    writer = SummaryWriter(log_dir=args.exp_dir)

    if int(args.device_num) > -1:
        logger.info(f'Using GPU device {args.device_num}...')
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device_num
        args.device = 'cuda'
    else:
        logger.info('Using CPU...')
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        args.device = 'cpu'

    if args.resume:
        checkpoint, model, optimizer = load_model(args.checkpoint)
        args = checkpoint['args']
        best_dev_loss = checkpoint['best_dev_loss']
        start_epoch = checkpoint['epoch']
        del checkpoint
    else:
        model = build_model(args)
        if args.data_parallel:
            model = torch.nn.DataParallel(model)
        optimizer = build_optim(args, model.parameters())
        best_dev_loss = 1e9
        start_epoch = 0

    logging.info(args)
    logging.info(model)

    train_loader, dev_loader, display_loader = create_data_loaders(args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)

    for epoch in range(start_epoch, args.num_epochs):
        scheduler.step(epoch)
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, writer)
        dev_loss, dev_time = evaluate(args, epoch, model, dev_loader, writer)
        visualize(args, epoch, model, display_loader, writer)

        is_new_best = dev_loss < best_dev_loss
        best_dev_loss = min(best_dev_loss, dev_loss)
        save_model(args, args.exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best)
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'DevLoss = {dev_loss:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
        )
    writer.close()


def create_arg_parser():
    parser = argparse.ArgumentParser(description="Training script for unrolled MRI recon.")
    # Network parameters
    parser.add_argument('--num-grad-steps', type=int, default=5, help='Number of unrolled iterations')
    parser.add_argument('--num-resblocks', type=int, default=2, help='Number of ResNet blocks per iteration')
    parser.add_argument('--num-features', type=int, default=256, help='Number of ResNet channels')
    parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--fix-step-size', type=bool, default=True, help='Fix unrolled step size')
    parser.add_argument('--circular-pad', type=bool, default=False, help='Flag to turn on circular padding')

    # Data parameters
    parser.add_argument('--data-path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--sample-rate', type=float, default=1.0, help='Fraction of total volumes to include')
    parser.add_argument('--resolution', default=320, type=int, help='Resolution of images')
    parser.add_argument('--num-emaps', type=int, default=1, help='Number of ESPIRiT maps')

    # Subsampling parameters
    parser.add_argument('--accelerations', nargs='+', default=[4], type=int,
                        help='Ratio of k-space columns to be sampled. If multiple values are '
                             'provided, then one of those is chosen uniformly at random for '
                             'each volume.')
    parser.add_argument('--center-fractions', nargs='+', default=[0.08], type=float,
                        help='Fraction of low-frequency k-space columns to be sampled. Should '
                            'have the same length as accelerations')

    # Training parameters
    parser.add_argument('--batch-size', default=2, type=int, help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=25,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.1, # 0.1 
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')

    # Miscellaneous parameters
    parser.add_argument('--seed', default=42, type=int, help='Seed for random number generators')
    parser.add_argument('--report-interval', type=int, default=100, help='Period of loss reporting')
    parser.add_argument('--data-parallel', action='store_true',
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--device-num', type=str, default='0',
                        help='Which device to train on.')
    parser.add_argument('--exp-dir', type=str, default='checkpoints',
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to an existing checkpoint. Used along with "--resume"')

    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
