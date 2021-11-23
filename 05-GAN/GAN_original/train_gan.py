import sys
import argparse
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Parser arguments
parser = argparse.ArgumentParser()
parser.add_argument("--main_path", type=str, default="05-GAN/GAN_original", help="Main path")
parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs for training")
parser.add_argument("--n_gpu", type=int, default=0, help="Number of GPUs for training")
parser.add_argument("--num_channels", default=3, type=int, help="Number of channels")
parser.add_argument("--input_dim", default=100, type=int, help="Number of dimensions of inputs")
parser.add_argument("--negative_slope", type=float, default=0.01, help="Negative slope parameter for LeakyReLU")
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for optimizer")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size")

opt = parser.parse_args()

num_channels = opt.num_channels
input_dim = opt.input_dim
n_gpu = opt.n_gpu
neg_slope = opt.negative_slope
lr = opt.lr
batch_size = opt.batch_size
main_path = opt.main_path
num_epochs = opt.num_epochs
n_gpu = opt.n_gpu