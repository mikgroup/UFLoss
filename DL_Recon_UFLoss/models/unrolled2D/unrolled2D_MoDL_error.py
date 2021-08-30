"""
Unrolled Compressed Sensing (3D) 
by Christopher M. Sandino (sandino@stanford.edu), 2020.

"""

import os, sys
import torch
from torch import nn
import sigpy.plot as pl
import utils.complex_utils as cplx
from utils.transforms import SenseModel
from utils.layers3D import ResNet
from unet.unet_model import UNet

from utils.flare_utils import ConjGrad
import matplotlib

# matplotlib.use('TkAgg')


class IBNorm(nn.Module):
    """Combine Instance Norm and Batch Norm into One Layer"""

    def __init__(self, in_channels):
        super(IBNorm, self).__init__()
        in_channels = in_channels
        self.bnorm_channels = int(in_channels / 2)
        self.inorm_channels = in_channels - self.bnorm_channels

        self.bnorm = nn.BatchNorm2d(self.bnorm_channels, affine=True)
        self.inorm = nn.InstanceNorm2d(self.inorm_channels, affine=False)

    def forward(self, x):
        bn_x = self.bnorm(x[:, : self.bnorm_channels, ...].contiguous())
        in_x = self.inorm(x[:, self.inorm_channels :, ...].contiguous())

        return torch.cat((bn_x, in_x), 1)


class Conv2dIBNormRelu(nn.Module):
    """Convolution + IBNorm + ReLu"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        with_ibn=True,
        with_relu=True,
    ):
        super(Conv2dIBNormRelu, self).__init__()

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]

        if with_ibn:
            layers.append(IBNorm(out_channels))
        if with_relu:
            layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Operator(torch.nn.Module):
    def __init__(self, A):
        super(Operator, self).__init__()
        self.operator = A

    def forward(self, x):
        return self.operator(x)

    def adjoint(self, x):
        return self.operator(x, adjoint=True)

    def normal(self, x):
        out = self.adjoint(self.forward(x))
        return out


class CG_module(nn.Module):
    def __init__(self, A=None, adjoint=None, verbose=False, lam_l2=0, cg_max=10):
        super(CG_module, self).__init__()
        self.A = None
        self.adj = None
        self.lam = lam_l2
        self.cg = cg_max
        self.verbose = verbose

    def initiate(self, A, adjoint):
        self.A = A
        self.adj = adjoint

    def forward(self, x):
        rhs = self.adj + self.lam * x
        out = ConjGrad(
            Aop_fun=self.A.normal,
            b=rhs,
            verbose=self.verbose,
            l2lam=self.lam,
            max_iter=self.cg,
        ).forward(rhs)
        return out

    def reverse(self, x):
        out = (1 / self.lam) * ((self.A.normal(x) + self.lam * x) - self.adj)
        return out


class UnrolledModel(nn.Module):
    """
    PyTorch implementation of Unrolled Compressed Sensing.

    Implementation is based on:
        CM Sandino, et al. "DL-ESPIRiT: Accelerating 2D cardiac cine 
        beyond compressed sensing" arXiv:1911.05845 [eess.SP]
    """

    def __init__(self, params):
        """
        Args:
            params (dict): Dictionary containing network parameters
        """
        super().__init__()

        # Extract network parameters
        self.num_grad_steps = params.num_grad_steps
        #         num_resblocks = params.num_resblocks
        #         num_features = params.num_features
        #         kernel_size = params.kernel_size
        #         drop_prob = params.drop_prob
        #         circular_pad = params.circular_pad
        #         fix_step_size = params.fix_step_size
        share_weights = params.share_weights
        self.num_cg_steps = params.num_cg_steps
        self.modl_lamda = params.modl_lamda
        self.error = False
        if params.error:
            self.error = True
            UNet_output = 4
            self.conv_e = nn.Sequential(
                Conv2dIBNormRelu(
                    2 * self.num_grad_steps,
                    2 * self.num_grad_steps,
                    3,
                    stride=1,
                    padding=1,
                ),
                Conv2dIBNormRelu(
                    2 * self.num_grad_steps,
                    1 * self.num_grad_steps,
                    3,
                    stride=1,
                    padding=1,
                ),
                Conv2dIBNormRelu(
                    1 * self.num_grad_steps,
                    1 * self.num_grad_steps,
                    3,
                    stride=1,
                    padding=1,
                ),
                Conv2dIBNormRelu(
                    1 * self.num_grad_steps,
                    2,
                    1,
                    stride=1,
                    padding=0,
                    with_ibn=False,
                    with_relu=False,
                ),
            )
        else:
            UNet_output = 2
        #         print(self.modl_lamda)
        #         sys.exit()
        self.cp = params.meld_cp
        self.device = params.device
        # Data dimensions
        #         self.num_emaps = params.num_emaps
        #         convtype = params.conv_type
        #         # ResNet parameters
        #         resnet_params = dict(num_resblocks=num_resblocks,
        #                              in_chans=2 * self.num_emaps,
        #                              chans=num_features,
        #                              kernel_size=kernel_size,
        #                              drop_prob=drop_prob,
        #                              circular_pad=circular_pad,
        #                              conv_type=convtype
        #                             )
        self.CGM = CG_module(False, False, False, self.modl_lamda, self.num_cg_steps)

        # Declare ResNets and RNNs for each unrolled iteration
        if share_weights:
            print("shared weights")
            self.unets = nn.ModuleList(
                [nn.ModuleList([self.CGM, UNet(2, UNet_output)])] * self.num_grad_steps
            )
        else:
            print("No shared weights")
            self.unets = nn.ModuleList(
                [
                    nn.ModuleList([self.CGM, UNet(2, UNet_output)])
                    for i in range(self.num_grad_steps)
                ]
            )

        # Declare step sizes for each iteration

    #         init_step_size = torch.tensor([-2.0], dtype=torch.float32).to(params.device)
    #         if fix_step_size:
    #             self.step_sizes = [init_step_size] * num_grad_steps
    #         else:
    #             self.step_sizes = [torch.nn.Parameter(init_step_size) for i in range(num_grad_steps)]
    def complex2real(self, image):
        """
        Convert complex torch image to two-channels image (real, imag)
        Args:
            image (torch.Tensor, dtype=torch.complex64): complex image of size [N, height, weight]
        Returns:
            image (torch.Tensor, dtype=torch.float32): real image of size [N, 2, height, weight]
        """
        return torch.cat((image.real[:, None, ...], image.imag[:, None, ...]), 1)

    def real2complex(self, image):
        """
        Convert real torch image to complex image.
        Args:
            image (torch.Tensor, dtype=torch.float32): real image of size [N, 2, height, weight]
        Returns:
            image (torch.Tensor, dtype=torch.complex64): complex image of size [N, height, weight]
        """
        return image[:, 0, ...] + 1j * image[:, 1, ...]

    def initiate(self, kspace, maps, mask=None):
        #         if self.num_emaps != maps.size()[-2]:
        #             raise ValueError('Incorrect number of ESPIRiT maps! Re-prep data...')
        """
        From pytorch 1.8, it supports natural complex data, this branch uses torch.fft instead of the old version of two seperate channels.
        """

        #         print(kspace.shape)
        #         sys.exit()
        if mask is None:
            mask = abs(kspace) > 0
        kspace *= mask

        # Get data dimensions
        self.dims = tuple(kspace.size())

        # Declare signal model
        A = SenseModel(maps, weights=mask)
        self.Sense = Operator(A)
        # Compute zero-filled image reconstruction
        self.zf_image = self.Sense.adjoint(kspace)
        #         pl.ImagePlot(self.zf_image.detach().cpu().numpy())
        self.CGM.initiate(self.Sense, self.zf_image)

    def evaluate(self):
        with torch.no_grad():
            if self.cp:
                size = [len(self.resnets)] + [a for a in self.zf_image.shape]
                self.Xcp = torch.zeros(size, device=self.device)
            else:
                self.Xcp = None
                self.dims = None
                self.num_emaps = None
            image = self.zf_image.clone()
            # Begin unrolled proximal gradient descent
            cpp = 0
            for resnet in self.resnets:
                if self.cp:
                    self.Xcp[cpp, ...] = image
                    cpp += 1
                # dc update
                #             pl.ImagePlot(image.detach().cpu())
                image = image.reshape(self.dims[0:4] + (self.num_emaps * 2,)).permute(
                    0, 4, 3, 2, 1
                )
                image = resnet[0](image)
                image = image.permute(0, 4, 3, 2, 1).reshape(
                    self.dims[0:4] + (self.num_emaps, 2)
                )
                image = resnet[1](image)

            #                 print("I Love you")

            return image, self.Xcp, self.dims, self.num_emaps

    def forward(self):
        """
        Args:
            kspace (torch.Tensor): Input tensor of shape [batch_size, height, width, time, num_coils, 2]
            maps (torch.Tensor): Input tensor of shape   [batch_size, height, width,    1, num_coils, num_emaps, 2]
            mask (torch.Tensor): Input tensor of shape   [batch_size, height, width, time, 1, 1]

        Returns:
            (torch.Tensor): Output tensor of shape       [batch_size, height, width, time, num_emaps, 2]
        """

        #         if self.num_emaps != maps.size()[-2]:
        #             raise ValueError('Incorrect number of ESPIRiT maps! Re-prep data...')

        #         CG_alg = ConjGrad(Aop_fun=Sense.normal,b=zf_image,verbose=False,l2lam=0.05,max_iter=self.c)
        #         cg_image = CG_alg.forward(zf_image)
        #         pl.ImagePlot(zf_image.detach().cpu())

        #         sys.exit()
        image = self.zf_image.clone()
        # Begin unrolled proximal gradient descent
        for i, unet in enumerate(self.unets):
            # dc update
            #             pl.ImagePlot(image.detach().cpu())
            image = unet[0](image)

            image = self.complex2real(image)
            image = unet[1](image)
            if self.error:
                error = image[:, 2:, :, :]

            image = image[:, :2, :, :]
            if self.error:
                if i == 0:
                    error_all = error
                else:
                    error_all = torch.cat((error_all, error), 1)
            image = self.real2complex(image)
        #             pl.ImagePlot(image.detach().cpu().numpy())
        if self.error:
            error_est = self.conv_e(error_all)
        image = self.complex2real(image)

        #             print("I Love you")
        if self.error:
            return image, error_est
        else:
            return image

