#!/bin/bash

# Network hyperparameters
num_grad_steps=4
device=0
modlflag=True
CG_steps=6
lr=4e-4
batch_size=3
date=20210617
loss_type=1
loss_normalized=True
# ufloss_flag=True
# ufloss_dir=/mikQNAP/NYU_knee_data/knee_train_h5/checkpoints_ufloss_mapping/train_UFLoss_feature_256_features_date_202104270_temperature_1/checkpoints/ckpt200.pth
# uflossfreq=8
# ufloss_weight=10

# dimension=256
# checkpoint=~/data/NYU_checkpoints/summary/train-3D_MELD_4steps_MoDLflag0_shared_CGsteps_6date_202104301_ufloss0_ufloss_weight_10_dimension_256/model_epoch89.pt

# convtype='conv2p1d'
# Name of model
model_name=train-error_estimation_$((num_grad_steps))steps_MoDLflag$((modlflag))_no_shared_CGsteps_$((CG_steps))date_$((date))_first_try_with_error

# Set folder names
dir_data=/home/kewang/data/NYU_data/
dir_save=/home/kewang/data/NYU_checkpoints/
dir_summary=$dir_save/summary/$model_name
python3 ../train_ufloss_errormap.py --data-path $dir_data \
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
                 --error \
                #  --share-weights \
                #  --resume \
                #  --checkpoint $checkpoint
#                  --data-parallel \
