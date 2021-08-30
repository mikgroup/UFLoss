#!/bin/bash

# Network hyperparameters
num_grad_steps=4
device=1
modlflag=True
CG_steps=6
lr=1e-4
batch_size=1
date=20210412
loss_type=2
efficient_ufloss=True
loss_normalized=True
ufloss_flag=True
ufloss_dir=/home/kewang/cp_temp/ckpt200.pth
ufloss_weight=1
uflossfreq=8
model_name=train-3D_MELD_$((num_grad_steps))steps_MoDLflag$((modlflag))_CGsteps_$((CG_steps))date_$((date))_ufloss$((ufloss_flag))_ufloss_weight_$((ufloss_weight))_efficient_$efficient_ufloss

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
                 --efficient-ufloss \
                 --modl-lamda 0
#                  --data-parallel \