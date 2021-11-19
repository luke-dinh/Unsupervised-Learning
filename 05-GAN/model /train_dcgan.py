import sys
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import argparse

parser = argparse.ArgumentParser("Train DCGAN")
parser.add_argument("--main_path", type=str, default="Unsupervised-Learning/05-GAN", help="Main Path")
parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs for training")
parser.add_argument("--n_gpu", type=int, default=0, help="Number of GPUs for training")
parser.add_argument("--num_channels", default=3, type=int, help="Number of channels")
parser.add_argument("--input_dim", default=100, type=int, help="Number of dimensions of inputs")
parser.add_argument("--feature_map", default=32, type=int, help="Size of feature maps")
parser.add_argument("--n_gpu", default=0, type=int, help="Number of GPUs (0 for CPU)")
parser.add_argument("--negative_slope", type=float, default=0.01, help="Negative slope parameter for LeakyReLU")

opt = parser.parse_args()
num_channels = opt.num_channels
input_dim = opt.input_dim
feature_map = opt.feature_map
n_gpu = opt.n_gpu
neg_slope = opt.negative_slope

main_path = opt.main_path
num_epochs = opt.num_epochs
n_gpu = opt.n_gpu

sys.path.append(main_path)
from dcgan import generator, discriminator, _init_weights

generator = generator()
discriminator = discriminator()




 