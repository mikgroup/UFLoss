#!/bin/bash

# Network hyperparameters
num_grad_steps=4
device=1
modlflag=True
CG_steps=6
lr=2e-4
batch_size=1
date=20210415
loss_type=2
loss_normalized=True
ufloss_flag=False
ufloss_dir=/home/kewang/cp_temp/CP_unsuper_patch130_traditional.pth
ufloss_weight=0
uflossfreq=8
shared=True
# convtype='conv2p1d'
# Name of model
model_name=train-3D_MELD_$((num_grad_steps))steps_MoDLflag$((modlflag))_CGsteps_$((CG_steps))date_$((date))_ufloss$((ufloss_flag))_ufloss_weight_$((ufloss_weight))_Shared_$shared
checkpoint=~/data/NYU_checkpoints/summary/train-3D_MELD_4steps_MoDLflag0_CGsteps_6date_20210415_ufloss0_ufloss_weight_0_Shared_True/model_epoch34.pt
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
                 --share-weights \
                  --resume \
                 --checkpoint $checkpoint
#                  --loss-uflossdir $ufloss_dir \
#                  --ufloss-weight $ufloss_weight \
#                  --uflossfreq $uflossfreq \
#                  --resume \
#                  --checkpoint $checkpoint
#                  --data-parallel \
