#!/bin/bash

# Network hyperparameters
num_grad_steps=4
device=2
modlflag=True
CG_steps=6
lr=2e-4
batch_size=1
date=202104052
loss_type=2
loss_normalized=True
ufloss_flag=True
ufloss_dir=/mikQNAP/NYU_knee_data/knee_train/NYU_knee_04022021_checkpoints_80_128/checkpoints/CP_unsuper_patch130.pth
ufloss_weight=5
uflossfreq=16
checkpoint=/mikQNAP/NYU_knee_data/knee_train_h5/checkpoints/summary/train-3D_MELD_4steps_MoDLflag0_CGsteps_6date_202104052_ufloss0_ufloss_weight_5/model_epoch13.pt
# convtype='conv2p1d'
# Name of model
model_name=train-3D_MELD_$((num_grad_steps))steps_MoDLflag$((modlflag))_CGsteps_$((CG_steps))date_$((date))_ufloss$((ufloss_flag))_ufloss_weight_$((ufloss_weight))

# Set folder names
dir_data=/home/kewang/data/NYU_data/
dir_save=/home/kewang/data/NYU_checkpoints/
dir_summary=$dir_save/summary/$model_name

python3 ../train_ufloss.py --data-path $dir_data \
				 --exp-dir $dir_summary \
				 --num-grad-steps $num_grad_steps \
                 --modl-flag $modlflag \
                 --num-cg-steps $CG_steps \
				 --slwin-init \
				 --device-num $device \
				 --loss-normalized $loss_normalized \
				 --loss-type $loss_type \
                 --batch-size $batch_size \
                 --lr $lr \
                 --loss-uflossdir $ufloss_dir \
                 --ufloss-weight $ufloss_weight \
                 --uflossfreq $uflossfreq \
#                  --data-parallel \