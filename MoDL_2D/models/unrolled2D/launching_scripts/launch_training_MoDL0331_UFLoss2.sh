#!/bin/bash

# Network hyperparameters
num_grad_steps=4
device=3
modlflag=True
CG_steps=6
lr=2e-4
batch_size=1
date=202103312
loss_type=2
loss_normalized=True
ufloss_flag=True
ufloss_dir=/mikQNAP/NYU_knee_data/knee_train/NYU_knee_0214_checkpoints_60_128/checkpoints/CP_unsuper_patch200.pth
ufloss_weight=15
uflossfreq=16
# convtype='conv2p1d'
# Name of model
model_name=train-3D_MELD_$((num_grad_steps))steps_MoDLflag$((modlflag))_CGsteps_$((CG_steps))date_$((date))_ufloss$((ufloss_flag))_ufloss_weight_$((ufloss_weight))

# Set folder names
dir_data=/mikQNAP/NYU_knee_data/knee_train_h5/data/
dir_save=/mikQNAP/NYU_knee_data/knee_train_h5/checkpoints/
dir_summary=$dir_save/summary/$model_name

python3 train_ufloss.py --data-path $dir_data \
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