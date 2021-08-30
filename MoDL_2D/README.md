<!-- # dl-cs-torch
This repository is borrowed from Chris Sandino. This repository contains implementations of various unrolled neural networks for 
accelerated MRI in PyTorch. You can find a very similar package in TensorFlow 
[here](https://github.com/MRSRL/dl-cs).

In particular, you can find training and reconstruction scripts inside
the `models` folder for each of the following networks:
* Unrolled Proximal Gradient Descent (2D)
* Unrolled Proximal Gradient Descent (3D)
* Deep subspace learning (3D)

Much of the training scripts are based on [fastMRI](http://fastMRI.org).

## To-Do
* Implement multi-GPU training with model parallelism
* Implement multi-GPU inference with data parallelism
* Implement non-Cartesian SENSE module using SigPy

## Installation
To use this package, install the required python packages (tested with python 3.6 on Ubuntu 16.04LTS):

```bash
pip install -r requirements.txt
```

## Directory Structure
* `datasets` contains prep scripts for various MRI datasets. These include:
	* FastMRI Knee Dataset
	* Dual-echo steady state (DESS) Knee Dataset
	* Cardiac Cine Dataset
* `models` contains networks and associated training scripts
* `utils` contains utility functions and classes used to load data, define network layers, 
  support complex values, create undersampling masks, etc.

## Training a model

Finally, to launch a training session, run the following script:

```bash
python3 train.py --data-path <path to dataset folder> --exp-dir <path to summary folder> --device-num 0
``` -->
