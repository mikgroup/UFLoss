import sys
import os
from optparse import OptionParser
import numpy as np
import sys
from tqdm import tqdm
import matplotlib

matplotlib.use("TkAgg")
import sigpy.plot as pl
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import models.unrolled2D.networks.resnet as resnet

# models/unrolled2D/networks
import random
from pympler import muppy, summary

# from unet.vgg import Vgg16
# from eval import eval_net
# # from unet import UNet
# from unet.unet_model import UNet
# from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch,get_imgs_and_masks_triple
# import resnet
# import sigpy.plot as pl
# from utils.utils_vgg import gram_matrix


def dir2dic(dir_list):
    n = len(dir_list)
    output = list([])
    for i in range(n):
        dir_n = dir_list[i]
        dk = list([dir_n])
        dk.extend(dir_n[:-4].split("_"))
        output.append(dk)
    return output


def index2torch(index_dic, dir_dic, folder, mag=0, pa=False):
    n_index = index_dic.shape[0]
    c = 1
    outp_shape = np.load(folder + dir_dic[index_dic[0]][0])[None, None, :, :].shape
    outp = np.zeros([n_index] + list(outp_shape[1:]), dtype=np.complex)
    for jj in range(n_index):
        if pa == True:
            c = random.uniform(0, 1)
            s = random.uniform(0.95, 1.05)
            c = np.exp(1j * 2 * np.pi * c)
        #             print(s)
        outp[jj, ...] = (
            np.load(folder + dir_dic[index_dic[jj]][0])[None, None, :, :] * c * s
        )
    if mag == 0:
        outtorch = torch.tensor(
            np.concatenate((outp.real, outp.imag), 1), dtype=torch.float32
        )
        del outp
        return outtorch
    else:
        return torch.tensor(abs(outp))


# files_knee_valid[0][:-4].split("_")
def get_args():
    parser = OptionParser()
    #     parser.add_option("-v", action="store_true", dest="verbose")
    parser.add_option(
        "--pa",
        "--phase-augmentation",
        dest="pa",
        action="store_true",
        default=False,
        help="Wheter apply random constant phase augmentation",
    )
    parser.add_option(
        "--na",
        "--noise-augmentation",
        dest="na",
        action="store_true",
        default=False,
        help="Wheter apply noise augmentation",
    )
    parser.add_option(
        "-l",
        "--load",
        dest="load",
        default=False,
        help="Folder directory contains 2D patch data",
    )
    parser.add_option(
        "-g",
        "--gpu",
        dest="gpu",
        default=False,
        type="int",
        help="GPU number, default is None (-g 0 means use gpu 0)",
    )

    parser.add_option(
        "-t",
        "--target",
        dest="target",
        default=False,
        help="Target folder directory for checkpoints",
    )
    parser.add_option(
        "-f",
        "--features",
        dest="features",
        default=128,
        type="int",
        help="Dimensions of features space",
    )
    parser.add_option(
        "-p",
        "--patch",
        dest="patch",
        default=20,
        type="int",
        help="Patchs each images to train the model",
    )
    parser.add_option(
        "--lr",
        "--learning-rate",
        dest="lr",
        default=0.0001,
        type="float",
        help="learning rate for the model",
    )
    parser.add_option(
        "-r",
        "--temperature-parameter",
        dest="temp",
        default=1,
        type="float",
        help="temperature parameter default: 1",
    )
    parser.add_option(
        "-b",
        "--batchsize",
        dest="batchsize",
        default=16,
        type="int",
        help="batch size for training",
    )
    parser.add_option(
        "-e", "--epochs", dest="epochs", default=200, type="int", help="epochs to train"
    )
    parser.add_option(
        "-m", "--model", dest="model", default=False, help="load checkpoints"
    )
    parser.add_option(
        "--mg", "--mag", dest="mag", default=0, type="int", help="magnitude"
    )

    #     parser.add_option('-p', '--patch', dest='patch',
    #                       default=20,type='int', help='Patch number for a single image (default 20)')
    #     parser.add_option('-s', '--patchsize', dest='patchsize',
    #                       default=40,type='int', help='patch size (default 40)')
    #     parser.add_option('-x', '--sx', dest='sx',
    #                       default=256,type='int', help='image dim: x')
    #     parser.add_option('-y', '--sy', dest='sy',
    #                       default=320,type='int', help='image dim: y')
    #     parser.add_option('--lr', '--learning-rate', dest='lg', default=0.0001,
    #                       type='float', help='learning rate of generator')

    #     parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
    #                       help='number of epochs')
    #     parser.add_option('-b', '--batch-size', dest='batchsize', default=10,
    #                       type='int', help='batch size')
    #     parser.add_option('--lg', '--learning-rate-generator', dest='lg', default=0.0001,
    #                       type='float', help='learning rate of generator')
    #     parser.add_option('--ld', '--learning-rate-discrinimator', dest='ld', default=0.0001,
    #                       type='float', help='learning rate of discrinimator')
    #     parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
    #                       default=False, help='use cuda')
    #     parser.add_option('-c', '--load', dest='load',
    #                       default=False, help='load file model: generator')
    #     parser.add_option('-d', '--loadb', dest='loadb',
    #                       default=False, help='load file model: discriminator')
    #     parser.add_option('-s', '--save', dest='save', type='int',
    #                       default=5, help='saving frequencey')
    #     parser.add_option('-a', '--gan', dest='gan', type='float',
    #                       default=0.01, help='GAN factor')
    #     parser.add_option('-i', '--interval', dest='interval', type='int',
    #                       default=1, help='Interval of GAN')
    #     parser.add_option('-n', '--name', dest='name',
    #                       default='mr', help='name of the expirement')
    #     parser.add_option('-p', '--per', dest='per', type='float',
    #                       default=0.05, help='perceptual factor')
    (options, args) = parser.parse_args()
    return options


if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda:%d" % (args.gpu))
    print(device)
    print("Whether we use magnitude", args.mag)
    target = args.target[:-1] + "_%d_%d/" % (args.patch, args.features)
    os.system("mkdir %s" % (target))
    os.system("mkdir %s%s" % (target, "checkpoints/"))

    n_features = args.features
    # Knee data folders:
    folder_knee = args.load
    print(folder_knee)
    files_knee = os.listdir(folder_knee)
    # len(files_knee)/
    files_knee_valid = list([])
    for i in files_knee:
        if os.path.splitext(i)[1] == ".npy":
            n_patch = int(os.path.splitext(i)[0].split("_")[2])
            if n_patch < args.patch:
                if (n_patch > 120) and (n_patch <= 150):
                    continue
                files_knee_valid.append(i)
    # Con23struct dir dictionary and permuta the number
    dir_dic = dir2dic(files_knee_valid)
    n_elements = len(dir_dic)
    print("patch number:", n_elements)
    #     n_features = 128
    # Define the bank
    B_tensor_cuda = torch.zeros((n_elements, n_features)).to(device)
    if args.mag == 0:
        ksnet = resnet.resnet18(num_classes=n_features).to(device)
    if args.mag == 1:
        ksnet = resnet.resnet18_m(num_classes=n_features).to(device)
    if args.model:
        ksnet.load_state_dict((torch.load(args.model)))
        print("Finish loading the pre-trained model")
    print("Start initializing the Memory Bank")
    n = 256
    steps = int(n_elements / n)
    ksnet.eval()
    if args.pa:
        print("We are using augmentation!! You are the best Ke!")
    with torch.no_grad():
        for j in tqdm(range(steps)):
            ip = np.array(range(j * n, (j + 1) * n))
            #             print(args.pa)
            #             sys.exit()
            inp = index2torch(ip, dir_dic, folder_knee, args.mag, args.pa).to(device)
            #             if args.
            #         break
            out = ksnet(inp)[0]
            #         break
            B_tensor_cuda[j * n : (j + 1) * n, :] = out
    #             print(j)
    print("Finish initialization")
    all_objects = muppy.get_objects()
    sum1 = summary.summarize(all_objects)
    # Prints out a summary of the large objects
    summary.print_(sum1)
    #     sys.exit()

    optimizer = torch.optim.Adam(ksnet.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    tau = args.temp
    loss_all = list([])
    #     index_dic = np.random.permutation(n_elements)
    batchsize = args.batchsize
    n_step = int(n_elements / batchsize)
    EPOCHS = args.epochs
    num = np.floor(n_elements / batchsize).astype(int)

    for epoch in range(EPOCHS):
        index_dic = np.random.permutation(n_elements)
        ksnet.train()
        #         epoch_loss = 0
        optimizer.zero_grad()
        print("Starting Epoch: %d" % (epoch + 1))
        torch.save(
            ksnet.state_dict(),
            target + "checkpoints/" "CP_unsuper_patch{}.pth".format(epoch + 1),
        )
        #         np.save(target+"kspace_unsupervised_resnet_patch.npy",B_tensor_cuda.detach().cpu().numpy())

        for index in tqdm(range(n_step)):
            ipd = index_dic[index * batchsize : (index + 1) * batchsize]
            #         break
            input_dic = index2torch(ipd, dir_dic, folder_knee, args.mag, args.pa).to(
                device
            )
            if args.na:
                noise = random.uniform(0, 1)
                noise = 0.05 * noise
                input_dic = (1 - noise) * input_dic + noise * torch.normal(
                    0, 1, input_dic.size()
                ).to(device)
            #                 pl.ImagePlot(input_dic.cpu().detach().numpy())
            output_dic = ksnet(input_dic)[0]
            #         break
            ## error: Trying to backward through the graph a second time, but the buffers have already been freed. Specify retain_graph=True when calling backward the first time.
            #             print(output_dic)
            #             B_tensor_cuda.requires_grad = False
            output_dic1 = (
                torch.mm(output_dic, B_tensor_cuda.detach().float().transpose(0, 1))
                / tau
            )
            #             print(output_dic.requires_grad)
            #             print(output_dic.shape)
            #         ipd = ipd % 22031
            loss = criterion(output_dic1, torch.LongTensor(ipd).to(device))
            if index % 200 == 0:
                print(loss.item())
                loss_all.append(loss.item())

            #             epoch_loss = epoch_loss+ loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            B_tensor_cuda[ipd, :] = output_dic
        all_objects = muppy.get_objects()
        sum1 = summary.summarize(all_objects)
        # Prints out a summary of the large objects
        summary.print_(sum1)
        np.save(target + "loss_all.npy", np.array(loss_all))

