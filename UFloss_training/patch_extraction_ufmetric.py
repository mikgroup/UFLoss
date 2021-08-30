import sys
import os
from optparse import OptionParser
import numpy as np
from tqdm import tqdm
import sys

sys.path.append("../DL_Recon_UFLoss/")
from models.unrolled2D.subsample_fastmri import MaskFunc

# models/unrolled2D/subsample_fastmri
import sigpy as sp
import matplotlib
import random

matplotlib.use("TkAgg")
import sigpy.plot as pl
import bart
import cv2
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
    mk_2 = MaskFunc([0.16], [2])
    mk_3 = MaskFunc([0.12], [3])
    mk_4 = MaskFunc([0.08], [4])
    mk_5 = MaskFunc([0.08], [5])

    folder_knee = args.load
    files_knee = os.listdir(folder_knee)

    #     n_elements = len(files_knee_valid)
    dir_dic = dir2dic(files_knee)
    n_elements = len(dir_dic)
    index_x = np.random.randint(n1, dim_x - n2, size=(n_elements, n_patch))
    index_y = np.random.randint(n1, dim_y - n2, size=(n_elements, n_patch))
    index = np.concatenate((index_x[..., None], index_y[..., None]), -1)

    np.save(target + "index_patch.npy", index)
    t = 0
    for dir_knee in tqdm(dir_dic):
        im = h5py.File(folder_knee + dir_knee[0], "r")["target"][()]
        under_rate = np.random.randint(2, 6)
        if under_rate == 2:
            mk1 = mk_2((1, 1, 372, 2))[0, 0, :, 0]
        if under_rate == 3:
            mk1 = mk_3((1, 1, 372, 2))[0, 0, :, 0]
        if under_rate == 4:
            mk1 = mk_4((1, 1, 372, 2))[0, 0, :, 0]
        if under_rate == 5:
            mk1 = mk_5((1, 1, 372, 2))[0, 0, :, 0]
        ksp = sp.fft(im)
        lowres_factor = random.uniform(0.15, 1)
        im_lowres = sp.ifft(
            sp.resize(
                sp.resize(ksp, (int(640 * lowres_factor), int(372 * lowres_factor))),
                (640, 372),
            )
        )
        ksp_under = ksp * mk1[None, ...]
        im_alias = sp.ifft(ksp_under)
        pics_factor = random.uniform(0, 1) * 0.3
        im_recon = bart.bart(
            1,
            "pics -l1 -r %f -S -d1" % (pics_factor),
            ksp_under[:, :, None, None],
            np.ones(ksp_under[:, :, None, None].shape),
        )
        print(under_rate)
        #         pl.ImagePlot(im_lowres)
        for kp in range(n_patch):
            pat = im[
                index[t, kp, 0] - n1 : index[t, kp, 0] + n2,
                index[t, kp, 1] - n1 : index[t, kp, 1] + n2,
            ]

            blur_k = np.random.randint(2, 5)
            noise_l = np.random.rand() * 0.2
            pat2 = cv2.blur(pat.real, (blur_k, blur_k)) + 1j * cv2.blur(
                pat.imag, (blur_k, blur_k)
            )
            pat3 = pat * (1 - noise_l) + noise_l * np.random.normal(0, 0.2, pat.shape)
            pat4 = im_alias[
                index[t, kp, 0] - n1 : index[t, kp, 0] + n2,
                index[t, kp, 1] - n1 : index[t, kp, 1] + n2,
            ]
            pat5 = im_recon[
                index[t, kp, 0] - n1 : index[t, kp, 0] + n2,
                index[t, kp, 1] - n1 : index[t, kp, 1] + n2,
            ]
            pat6 = im_lowres[
                index[t, kp, 0] - n1 : index[t, kp, 0] + n2,
                index[t, kp, 1] - n1 : index[t, kp, 1] + n2,
            ]

            np.save(
                target
                + "patch_data/"
                + "%s_%s_%d.npy" % (dir_knee[1], dir_knee[2], kp),
                pat,
            )
            np.save(
                target
                + "patch_data/"
                + "%s_%s_%d.npy" % (dir_knee[1], dir_knee[2], kp + n_patch),
                pat2,
            )
            np.save(
                target
                + "patch_data/"
                + "%s_%s_%d.npy" % (dir_knee[1], dir_knee[2], kp + 2 * n_patch),
                pat3,
            )
            np.save(
                target
                + "patch_data/"
                + "%s_%s_%d.npy" % (dir_knee[1], dir_knee[2], kp + 3 * n_patch),
                pat4,
            )
            np.save(
                target
                + "patch_data/"
                + "%s_%s_%d.npy" % (dir_knee[1], dir_knee[2], kp + 4 * n_patch),
                pat5,
            )
            np.save(
                target
                + "patch_data/"
                + "%s_%s_%d.npy" % (dir_knee[1], dir_knee[2], kp + 5 * n_patch),
                pat6,
            )

        t += 1
