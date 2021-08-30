import os, sys
import numpy as np
import matplotlib.pyplot as plt

# load custom modules
from utils import cfl

top_dir = '/mnt/dense/sandino/Studies_CineRetro/Exam5050/cine.tmp-recon.lrtest'

# load CFL files
im_zf = np.squeeze(cfl.read(os.path.join(top_dir, 'im.zf')))
im_cs = np.squeeze(cfl.read(os.path.join(top_dir, 'im.pics')))
im_dl3 = np.squeeze(cfl.read(os.path.join(top_dir, 'im.dl3d')))
im_dl21 = np.squeeze(cfl.read(os.path.join(top_dir, 'im.lr')))
im_gt = np.squeeze(cfl.read(os.path.join(top_dir, 'im.full')))

print(im_zf.shape)
print(im_cs.shape)
print(im_dl3.shape)
print(im_dl21.shape)
print(im_gt.shape)

# all should be the same dimensions
ncph, nmaps, nslices, yres, xres = im_gt.shape # 20, 2, 14, 180, 200

# from matlab..
# crop_x = 60:142;
# crop_y = 22:92;

# crop images to [ncph, y, x]
#im_zf = im_zf[:,0,3,21:92,59:142]
#im_cs = im_cs[:,0,3,21:92,59:142]
#im_dl3 = im_dl3[:,0,3,21:92,59:142]
#im_dl21 = im_dl21[:,0,3,21:92,59:142]
#im_gt = im_gt[:,0,3,21:92,59:142] 
im_zf = im_zf[:,0,3,:,:]
im_cs = im_cs[:,0,3,:,:]
im_dl3 = im_dl3[:,0,3,:,:]
im_dl21 = im_dl21[:,0,3,:,:]
im_gt = im_gt[:,0,3,:,:] 

# normalize each reconstruction separately
blood_idx = int(0.90*np.prod(im_gt.shape))
im_zf /= np.sort(np.absolute(im_zf), axis=None)[blood_idx]
im_cs /= np.sort(np.absolute(im_cs), axis=None)[blood_idx]
im_dl3 /= np.sort(np.absolute(im_dl3), axis=None)[blood_idx]
im_dl21  /= np.sort(np.absolute(im_dl21), axis=None)[blood_idx]
im_gt /= np.sort(np.absolute(im_gt), axis=None)[blood_idx]

# im_zf /= np.amax(np.abs(im_zf.flatten()))
# im_cs /= np.amax(np.abs(im_cs.flatten()))
# im_dl3 /= np.amax(np.abs(im_dl3.flatten()))
# im_dl21 /= np.amax(np.abs(im_dl21.flatten()))
# im_gt /= np.amax(np.abs(im_gt.flatten()))

# concatenate all datasets along Y
im_save = np.concatenate((im_zf, im_cs, im_dl3, im_dl21, im_gt), axis=1)
im_save_error = np.concatenate((np.abs(im_cs)-np.abs(im_gt), 
	                            np.abs(im_dl3)-np.abs(im_gt), 
	                            np.abs(im_dl21)-np.abs(im_gt)), axis=1)
# im_save = np.flip(im_save, axis=2)

cfl.write('recon_images', im_save.swapaxes(1,2))
cfl.write('recon_error', im_save_error.swapaxes(1,2))

# take magnitude
im_save = np.abs(im_save)
im_save_error = np.abs(im_save_error)

# use matplotlib to save video frames as PNG
for cph in range(ncph):
	plt.imsave('./out_%02d.png' % cph, im_save[cph,:,:].T, vmin=0.0, vmax=2.0, cmap="gray")

# use ffmpeg to generate video
os.system("ffmpeg -f image2 -r 20 -i ./out_%02d.png -vb 20M -qscale 0 -vcodec mpeg4 -y ./fig3_video.mp4")
# frank's ffmpeg command
# os.system("ffmpeg -r 20 -i './out_%02d.png' -vf crop=floor(iw/2)*2-10:floor(ih/2)*2-10 -pix_fmt yuv420p -crf 1 -vcodec libx264 -preset veryslow fig3_video.mp4")


# use matplotlib to to save error frames as PNG
for cph in range(ncph):
	plt.imsave('./out_error_%02d.png' % cph, im_save_error[cph,:,:].T, vmin=0.0, vmax=0.25, cmap="viridis")

# use ffmpeg to generate video
os.system("ffmpeg -f image2 -r 20 -i ./out_error_%02d.png -vb 20M -qscale 0 -vcodec mpeg4 -y ./fig4_video.mp4")

# remove PNG files
os.system('rm -rf ./out_*.png')

