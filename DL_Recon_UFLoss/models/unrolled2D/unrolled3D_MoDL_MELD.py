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

from utils.flare_utils import ConjGrad
import matplotlib

# matplotlib.use('TkAgg')


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
        num_grad_steps = params.num_grad_steps
        num_resblocks = params.num_resblocks
        num_features = params.num_features
        kernel_size = params.kernel_size
        drop_prob = params.drop_prob
        circular_pad = params.circular_pad
        fix_step_size = params.fix_step_size
        share_weights = params.share_weights
        self.num_cg_steps = params.num_cg_steps
        self.modl_lamda = params.modl_lamda
        self.cp = params.meld_cp
        self.device = params.device
        # Data dimensions
        self.num_emaps = params.num_emaps
        convtype = params.conv_type
        # ResNet parameters
        resnet_params = dict(
            num_resblocks=num_resblocks,
            in_chans=2 * self.num_emaps,
            chans=num_features,
            kernel_size=kernel_size,
            drop_prob=drop_prob,
            circular_pad=circular_pad,
            conv_type=convtype,
        )
        self.CGM = CG_module(False, False, False, self.modl_lamda, self.num_cg_steps)

        # Declare ResNets and RNNs for each unrolled iteration
        if share_weights:
            print("shared weights")
            self.resnets = nn.ModuleList(
                nn.ModuleList([ResNet(**resnet_params), self.CGM]) * num_grad_steps
            )
        else:
            print("No shared weights")
            self.resnets = nn.ModuleList(
                [
                    nn.ModuleList([ResNet(**resnet_params), self.CGM])
                    for i in range(num_grad_steps)
                ]
            )

        # Declare step sizes for each iteration

    #         init_step_size = torch.tensor([-2.0], dtype=torch.float32).to(params.device)
    #         if fix_step_size:
    #             self.step_sizes = [init_step_size] * num_grad_steps
    #         else:
    #             self.step_sizes = [torch.nn.Parameter(init_step_size) for i in range(num_grad_steps)]

    def initiate(self, kspace, maps, mask=None):
        if self.num_emaps != maps.size()[-2]:
            raise ValueError("Incorrect number of ESPIRiT maps! Re-prep data...")

        if mask is None:
            mask = cplx.get_mask(kspace)
        kspace *= mask

        # Get data dimensions
        self.dims = tuple(kspace.size())

        # Declare signal model
        A = SenseModel(maps, weights=mask)
        self.Sense = Operator(A)
        # Compute zero-filled image reconstruction
        self.zf_image = self.Sense.adjoint(kspace)
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
        for resnet in self.resnets:
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

        #             print("I Love you")

        return image
