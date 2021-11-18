import sys 
sys.path.append('.')

import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import argparse

parser = argparse.ArgumentParser("Train DCGAN")
parser.add_argument("--main_path", type=str, default="05-GAN", help="Main Path")
parser.add_argument("--num_epocchs", type=int, default=20, help="Number of epochs for training")
parser.add_argument("--n_gpu", type=int, default=0, help="Number of GPUs for training")

opt = parser.parse_args()
main_path = opt.main_path
num_epochs = opt.num_epochs
n_gpu = opt.n_gpu

from model.dcgan import *
 