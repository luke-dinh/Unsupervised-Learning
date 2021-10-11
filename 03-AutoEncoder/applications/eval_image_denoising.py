import argparse
import os, sys
import numpy as np

import torch 
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader 
import torchvision.transforms as transforms

parser = argparse.ArgumentParser("Image Denoising Evaluation")
parser.add_argument( 
    "--main_path",
    default="/home/lukedinh/Desktop/Unsupervised-Learning/03-AutoEncoder",
    type=str,
    help="Your main path"
)
opt = parser.parse_args()
main_path = opt.main_path
sys.path.append(main_path)