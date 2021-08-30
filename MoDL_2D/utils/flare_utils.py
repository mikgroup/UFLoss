#!/usr/bin/env python
#Borrowed from Jon Tamir Basic 
import sys
import numpy as np
import sigpy.plot as pl
from torch.autograd import Variable
import torch
# import complex_torch
import progressbar
from tqdm import tqdm
# import torchrecon_utils as recon
from pytorch_wavelets import DWTForward, DWTInverse
def roll(im, ix,iy):  
    imx = torch.cat((im[:,:,-ix:,...], im[:,:,:-ix,...]),2)
    return torch.cat((imx[:,:,:,-iy:,...], imx[:,:,:,:-iy,...]),3)
def RMSE_im(gt,target):
    n = np.prod(gt.shape)
    return np.sqrt(np.sum((abs(gt-target)**2/n)))/np.sqrt(np.sum((abs(gt)**2/n)))

#torchversion fft
# def roll(tensor, shift, axis):
#     if shift == 0:
#         return tensor

#     if axis < 0:
#         axis += tensor.dim()

#     dim_size = tensor.size(axis)
#     after_start = dim_size - shift
#     if shift < 0:
#         after_start = -shift
#         shift = dim_size - abs(shift)

#     before = tensor.narrow(axis, 0, dim_size - shift)
#     after = tensor.narrow(axis, after_start, shift)
#     return torch.cat([after, before], axis)
def torch_fftshift(im):
    t = len(im.shape)
    n = int(np.floor(im.shape[t-3]/2))
    m = int(np.floor(im.shape[t-2]/2))
    P_torch1 = roll(roll(im,m,t-2),n,t-3)
    return P_torch1
def torch_ifftshift(im):
    t = len(im.shape)
    n = int(np.ceil(im.shape[t-3]/2))
    m = int(np.ceil(im.shape[t-2]/2))
    P_torch1 = roll(roll(im,m,t-2),n,t-3)
    return P_torch1
def torch_fft2c(im):
    t = len(im.shape)
    par = torch.sqrt(torch.tensor(im.shape[t-3]*im.shape[t-2],dtype=torch.float64)).cuda()
    return (1/(par))*torch_fftshift(torch.fft(torch_ifftshift(im),2))
def torch_ifft2c(im):
    t = len(im.shape)
    par = torch.sqrt(torch.tensor(im.shape[t-3]*im.shape[t-2],dtype=torch.float64)).cuda()
#     if centered == False:
    return (par)*torch_ifftshift(torch.ifft(torch_fftshift(im),2))
#     else:
#         return (par)*torch_ifftshift(torch.ifft(torch_fftshift(im),2))
        
def fft2c(x):
    return 1 / np.sqrt(np.prod(x[0,...].shape)) * np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))

def ifft2c(y):
    return np.sqrt(np.prod(y[0,...].shape)) * np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(y)))
# Wavelet
def cat_wavelet(Yl,Yh):
    n = len(Yh)
    WL = Yl[0,:,:,:]
    for t in range(n):
        P1 = Yh[n-1-t][0,:,0,:,:]
        P2 = Yh[n-1-t][0,:,1,:,:]
        P3 = Yh[n-1-t][0,:,2,:,:]
        WU = torch.cat((WL,P1),2)
        WD = torch.cat((P2,P3),2)
        WL = torch.cat((WU,WD),1)
    return WL
def torch_wavelet(P_torch):
    xfm = DWTForward(J=3, mode='per', wave='db3').cuda()
    PP = P_torch[:,:,0][None,None,:,:]
    PP = PP.type(torch.float)
    PP1 = P_torch[:,:,1][None,None,:,:]
    PP1 = PP1.type(torch.float)
    PP_complex = torch.cat((PP,PP1),1)
    Yl, Yh = xfm(PP_complex)
    Pwave = cat_wavelet(Yl, Yh)
    Pwave=Pwave.permute((1,2,0))
    return Pwave
def torch2np(im_torch,data_complex = True):
    # im_torch: torch tensor whose shape is [512,512,2]
    if data_complex:
        im_np = im_torch.detach().cpu().numpy()
        return im_np[...,0] + 1j*im_np[...,1]
    else:
        return im_torch.detach().cpu().numpy()

def np2torch(im_torch,data_complex = True):
    # im_torch: torch tensor whose shape is [512,512,2]
    if data_complex:
        return torch.tensor(np.stack((im_torch.real, im_torch.imag), axis=-1))
    else:
        return torch.tensor(im_torch)
'''
Defines complex-valued arithmetic for ndarrays, where the real and imaginary
channels are stored in the last dimension
'''

def c2r(z):
    ''' Convert from complex to 2-channel real '''
    assert type(z) is np.ndarray, 'Must be numpy.ndarray'
    return np.stack((z.real, z.imag), axis=-1)

def r2c(x):
    ''' Convert from 2-channel real to complex '''
    assert type(x) is np.ndarray, 'Must be numpy.ndarray'
    return x[...,0] + 1j *  x[...,1]
def rrs(x):
    # dim 0: channels
    # dim 1: 2d image
    im_sos = ((x**2).sum(dim=0).sum(dim=-1))**(0.5)
    return im_sos
def zmul(x1, x2):
    ''' complex-valued multiplication '''
    xr = x1[...,0] * x2[...,0] -  x1[...,1] * x2[...,1]
    xi = x1[...,0] * x2[...,1] +  x1[...,1] * x2[...,0]
    if type(x1) is np.ndarray:
        return np.stack((xr, xi), axis=-1)
    elif type(x1) is torch.Tensor:
        return torch.stack((xr, xi), dim=-1)
    else:   
        return xr, xi

def zconj(x):
    ''' complex-valued conjugate '''
    if type(x) is np.ndarray:
        return np.stack((x[...,0], -x[...,1]), axis=-1)
    elif type(x) is torch.Tensor:
        return torch.stack((x[...,0], -x[...,1]), dim=-1)
    else:   
        return x[...,0], -x[...,1]

def zabs(x):
    ''' complex-valued magnitude '''
    if type(x) is np.ndarray:
        return np.sqrt(zmul(x, zconj(x)))[...,0]
    elif type(x) is torch.Tensor:
        return torch.sqrt(zmul(x, zconj(x)))
    else:   
        return -1.
def dot(x1, x2):
    return torch.sum(x1*x2)

def ip(x):
    return dot(x, x)

def dot_batch(x1, x2):
    batch = x1.shape[0]
    return torch.reshape(x1.conj()*x2,(batch,-1)).sum(1)

def ip_batch(x):
    return dot_batch(x, x)
    
class ConjGrad(torch.nn.Module):
    def __init__(self, b, Aop_fun, max_iter=20, l2lam=0., eps=1e-6, verbose=True):
        super(ConjGrad, self).__init__()

        self.b = b
        self.Aop_fun = Aop_fun
        self.max_iter = max_iter
        self.l2lam = l2lam
        self.eps = eps
        self.verbose = verbose

    def forward(self, x):
        #return MyConjGrad(self.b, self.Aop_fun, self.max_iter, self.l2lam, self.eps, True)(x)
        return conjgrad(x=x, b=self.b, Aop_fun=self.Aop_fun, max_iter=self.max_iter, l2lam=self.l2lam, eps=self.eps, verbose=self.verbose)
    
def conjgrad(x, b, Aop_fun, max_iter=10, l2lam=0., eps=1e-4, verbose=True):
    ''' batched conjugate gradient descent. assumes the first index is batch size '''

    # explicitly remove r from the computational graph
    r = b.new_zeros(b.shape, requires_grad=False)
#     pl.ImagePlot(b.detach().cpu().numpy())
#     print(r.shape)
    # the first calc of the residual may not be necessary in some cases...
    if l2lam > 0:
        r = b - (Aop_fun(x) + l2lam * x)
    else:
        r = b - Aop_fun(x)
    p = r

    rsnot = ip_batch(r)
    rsold = rsnot
    rsnew = rsnot
    eps_squared = eps ** 2

    reshape = (-1,) + (1,) * (len(x.shape) - 1)

    for i in range(max_iter):

        if verbose:
            print('{i}: {rsnew}'.format(i=i, rsnew=dbp.utils.itemize(torch.sqrt(rsnew))))

        if rsnew.real.max() < eps:
            break

        if l2lam > 0:
            Ap = Aop_fun(p) + l2lam * p
        else:
            Ap = Aop_fun(p)
        pAp = dot_batch(p, Ap)

        #print(dbp.utils.itemize(pAp))

        alpha = (rsold / pAp).reshape(reshape)

        x = x + alpha * p
#         print(x.shape)
        r = r - alpha * Ap

        rsnew = ip_batch(r)

        beta = (rsnew / rsold).reshape(reshape)

        rsold = rsnew

        p = beta * p + r


    if verbose:
        print('FINAL: {rsnew}'.format(rsnew=torch.sqrt(rsnew)))

    return x

    
    
def maps_forw(img, maps):
    return zmul(img[:,None,...], maps)

def maps_adj(cimg, maps):
    return torch.sum(zmul(zconj(maps), cimg), 1, keepdim=False)

def fft_forw(x, ndim=2):
    return torch.fft(x, signal_ndim=ndim, normalized=True)

def fft_adj(x, ndim=2):
    return torch.ifft(x, signal_ndim=ndim, normalized=True)

def mask_forw(y, mask):
#     print(y.shape)
#     print(mask.shape)
    return y * mask.unsqueeze(1).unsqueeze(-1)
def mask_forw_3D(y, mask):
#     print(y.shape)
#     print(mask.shape)
    return y * mask.unsqueeze(-1)

def sense_forw(img, maps, mask):
    return mask_forw(fft_forw(maps_forw(img, maps)), mask)
def sense_forw_3D(img, maps, mask):
    return mask_forw_3D(fft_forw(maps_forw(img, maps)), mask)

def sense_adj(ksp, maps, mask):
    return maps_adj(fft_adj(mask_forw(ksp, mask)), maps)
def sense_adj_3D(ksp, maps, mask):
    return maps_adj(fft_adj(mask_forw_3D(ksp, mask)), maps)

def sense_normal(img, maps, mask):
    return maps_adj(fft_adj(mask_forw(fft_forw(maps_forw(img, maps)), mask)), maps)

class SenseModel_3D(torch.nn.Module):
    def __init__(self, maps, mask, l2lam=False):
        super(SenseModel_3D, self).__init__()
        self.maps = maps
        self.mask = mask
        self.l2lam = l2lam

        #if normal is None:
            #self.normal_fun = self._normal
        #else:
            #self.normal_fun = normal

    def forward(self, x):
        x = x.squeeze(0).permute(1,2,3,0)
        return sense_forw_3D(x, self.maps, self.mask)

    def adjoint(self, y):
        return sense_adj_3D(y, self.maps, self.mask).permute(3,0,1,2).unsqueeze(0)

    def normal(self, x):
        out = self.adjoint(self.forward(x))
        if self.l2lam:
            out = out + self.l2lam * x
        return out

class SenseModel(torch.nn.Module):
    def __init__(self, maps, mask, l2lam=False):
        super(SenseModel, self).__init__()
        self.maps = maps
        self.mask = mask
        self.l2lam = l2lam

        #if normal is None:
            #self.normal_fun = self._normal
        #else:
            #self.normal_fun = normal

    def forward(self, x):
#         x = x.squeeze(0).permute(1,2,3,0)
        return sense_forw(x, self.maps, self.mask)

    def adjoint(self, y):
        return sense_adj(y, self.maps, self.mask)

    def normal(self, x):
        out = self.adjoint(self.forward(x))
        if self.l2lam:
            out = out + self.l2lam * x
        return out

def CG_adj(ksp,mps,mask):
    SenseModel = flare.SenseModel(mps,mask) 
    adj = SenseModel.adjoint(ksp)
    return SenseModel,adj
def CG_adj_3D(ksp,mps,mask):
    SenseModel = flare.SenseModel_3D(mps,mask) 
    adj = SenseModel.adjoint(ksp)
    return SenseModel,adj
def CG_MoDL_3D(ksp,mps,mask,lam = 0):
    SenseModel = flare.SenseModel_3D(mps,mask)    
    adj = SenseModel.adjoint(ksp)
    CG_alg = flare.ConjGrad(Aop_fun=SenseModel.normal,b=adj,verbose=False,l2lam=lam)
    return CG_alg.forward(adj)
def CG_MoDL(ksp,mps,mask,lam = 0):
    SenseModel = flare.SenseModel(mps,mask)    
    adj = SenseModel.adjoint(ksp)
    CG_alg = flare.ConjGrad(Aop_fun=SenseModel.normal,b=adj,verbose=False,l2lam=lam)
    return CG_alg.forward(adj)