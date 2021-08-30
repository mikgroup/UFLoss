#!/bin/bash

# Network hyperparameters
num_grad_steps=5
num_resblocks=2
num_features=64
device=3
modlflag=True
CG_steps=8
date=6084
loss_type=2
loss_normalized=True
ufloss_flag=False
ufloss_dir=/home/kewang/cardiac_cine/cp_patchlearning_augmented_200_512/checkpoints/CP_unsuper_patch128.pth
ufloss_weight=30
uflossfreq=8
# Name of model
model_name=train-3D_$((num_grad_steps))steps_$((num_resblocks))resblocks_$((num_features))features_MoDLflag$((modlflag))_CGsteps_$((CG_steps))date_$((date))_ufloss$((ufloss_flag))

# Set folder names
dir_data=/home/kewang/cardiac_cine
dir_summary=$dir_data/summary/$model_name

python3 train.py --data-path $dir_data \
				 --exp-dir $dir_summary \
				 --num-grad-steps $num_grad_steps \
				 --num-resblocks $num_resblocks \
				 --num-features $num_features \
				 --num-emaps 1 \
				 --slwin-init \
				 --device-num $device \
				 --loss-normalized $loss_normalized \
				 --loss-type $loss_type \