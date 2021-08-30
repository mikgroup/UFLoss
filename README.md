# Unsupervised Feature Loss (UFLoss) for High Fidelity Deep learning (DL)-based reconstruction
<img src="github_images/Figure_1.jpg" width="900px"/>
Official github repository for the paper "High Fidelity Deep Learning-based MRI Reconstruction with Instance-wise Discriminative Feature Matching Loss". In this work, a novel patch-based Unsupervised Feature loss (UFLoss) is proposed and incorporated into the training of DL-based reconstruction frameworks in order to preserve perceptual similarity and high-order statistics. In-vivo experiments indicate that adding the UFLoss encourages sharper edges with higher overall image quality under DL-based reconstruction framework. Our implementations are in [here](https://profs.etsmtl.ca/hlombaert/public/medneurips2019/107_CameraReadySubmission_NeurIPS_2019_DCS_CR.pdf)

## Installation
To use this package, install the required python packages (tested with python 3.8 on Ubuntu 20.04 LTS):
```bash
pip install -r requirements.txt
```