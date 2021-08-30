#!/bin/bash

# Network hyperparameters
num_grad_steps=5
num_resblocks=2
num_features=32
device=4
modlflag=True
CG_steps=8
date=6116
loss_type=2
loss_normalized=True
ufloss_flag=False
ufloss_dir=/home/kewang/cardiac_cine/cp_patchlearning_augmented_200_512/checkpoints/CP_unsuper_patch128.pth
ufloss_weight=50
convtype='conv2p1d'
uflossfreq=10
# Name of model
model_name=train-3D_$((num_grad_steps))steps_$((num_resblocks))resblocks_$((num_features))features_MoDLflag$((modlflag))_CGsteps_$((CG_steps))date_$((date))_ufloss_${ufloss_flag}_convtype_$convtype

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
				 --modl-flag $modlflag \
				 --num-cg-steps $CG_steps \
				 --loss-normalized $loss_normalized \
				 --loss-type $loss_type \
                 --conv-type $convtype
# 				 --loss-uflossdir $ufloss_dir \
#                  --ufloss-weight $ufloss_weight \
#                  --uflossfreq $uflossfreq \
#                  --conv-type $convtype