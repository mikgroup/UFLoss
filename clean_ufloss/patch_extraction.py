import sys
import os
from optparse import OptionParser
import numpy as np
from tqdm import tqdm
import sigpy.plot as pl

import h5py


def dir2dic(dir_list):
    n = len(dir_list)
    output = list([])
    for i in range(n):
        dir_n = dir_list[i]
        dk = list([dir_n])
        dk.extend(dir_n[:-3].split("_"))
        output.append(dk)
    return output


# files_knee_valid[0][:-4].split("_")
def get_args():
    parser = OptionParser()
    parser.add_option(
        "-l",
        "--load",
        dest="load",
        default=False,
        help="Folder directory contains 2D image data",
    )
    parser.add_option(
        "-t",
        "--target",
        dest="target",
        default=False,
        help="Target folder directory for 3D image patchs",
    )
    parser.add_option(
        "-p",
        "--patch",
        dest="patch",
        default=20,
        type="int",
        help="Patch number for a single image (default 20)",
    )
    parser.add_option(
        "-s",
        "--patchsize",
        dest="patchsize",
        default=40,
        type="int",
        help="patch size (default 40)",
    )
    parser.add_option(
        "-x", "--sx", dest="sx", default=256, type="int", help="image dim: x"
    )
    parser.add_option(
        "-y", "--sy", dest="sy", default=320, type="int", help="image dim: y"
    )

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
    n_patch = args.patch
    patch_size = args.patchsize
    n1 = int(patch_size / 2)
    n2 = patch_size - n1
    dim_x = args.sx
    dim_y = args.sy
    target = args.target[:-1] + "_%d_%d/" % (n_patch, patch_size)
    os.system("mkdir %s" % (target))
    os.system("mkdir %s%s" % (target, "patch_data/"))

    folder_knee = args.load
    files_knee = os.listdir(folder_knee)
    dir_dic = dir2dic(files_knee)
    n_elements = len(dir_dic)
    index_x = np.random.randint(n1, dim_x - n2, size=(n_elements, n_patch))
    index_y = np.random.randint(n1, dim_y - n2, size=(n_elements, n_patch))
    index = np.concatenate((index_x[..., None], index_y[..., None]), -1)

    np.save(target + "index_patch.npy", index)
    t = 0
    for dir_knee in tqdm(dir_dic):
        im = h5py.File(folder_knee + dir_knee[0], "r")["target"][()]
        #         pl.ImagePlot(im)
        #         im = np.load(folder_knee+dir_knee[0])
        for kp in range(n_patch):
            #                 print(dir_knee[1:])
            pat = im[
                index[t, kp, 0] - n1 : index[t, kp, 0] + n2,
                index[t, kp, 1] - n1 : index[t, kp, 1] + n2,
            ]
            np.save(
                target
                + "patch_data/"
                + "%s_%s_%d.npy" % (dir_knee[1], dir_knee[2], kp),
                pat,
            )
        t += 1
#         print(t)


#     net_generator = UNet(n_channels=1000, n_classes=3)
#     net_generator = nn.DataParallel(net_generator)
#     net_discriminator = resnet.resnet18(num_classes=1,inputchannel=3)
#     net_discriminator = nn.DataParallel(net_discriminator)
#     if args.load:
#         net_generator.load_state_dict(torch.load(args.load))
#         print('Model loaded from {}'.format(args.load))
#     if args.loadb:
#         net_discriminator.load_state_dict(torch.load(args.loadb))
#         print('Model loaded from {}'.format(args.loadb))

#     if args.gpu:
#         net_generator.cuda()
#         net_discriminator.cuda()
#         # cudnn.benchmark = True # faster convolutions, but more memory

#     try:
#         train_net(net_g=net_generator,
#                   net_d = net_discriminator,
#                   epochs=args.epochs,
#                   batch_size=args.batchsize,
#                   lr_g=args.lg,
#                   lr_d = args.ld,
#                   gpu=args.gpu,
#                   img_scale=0.5,
#                   svf = args.save,
#                   GAN_lamda = args.gan,perceptual_lamda = args.per,opt_fre = args.interval,name = args.name)
#     except KeyboardInterrupt:
#         torch.save(net_generator.state_dict(), 'INTERRUPTED.pth')
#         print('Saved interrupt')
#         try:
#             sys.exit(0)
#         except SystemExit:
#             os._exit(0)
