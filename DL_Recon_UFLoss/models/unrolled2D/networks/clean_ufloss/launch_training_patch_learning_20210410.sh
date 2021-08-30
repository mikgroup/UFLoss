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
dir_data=/home/kewang/NYU_knee_patches_20210402_augmented_ufmetric_30_60/patch_data
dir_save=/mikQNAP/NYU_knee_data/knee_train_h5/checkpoints_ufloss_mapping/$model_name
device=0
dimension=128
lr=1e-4
temperature=1
batch=256
epochs=200
date=202104110

model_name=train_UFLoss_feature_$((dimension))_features_date_$((date))


python patch_learning_momentum.py --datadir $dir_data\
                 -g $device \
                 --logdir $dir_save \
                 -f $dimension \
                 --learning-rate $lr \
                 --temperature $temperature \
                 --batchsize $batch \
                 -e $epochs \
                 --use_phase_augmentation \