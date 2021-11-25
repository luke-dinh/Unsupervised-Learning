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
parser.add_argument("--input_dim", default=784, type=int, help="MNIST dim")
parser.add_argument("--z_dim", default=100, type=int, help="Number of dimensions of inputs")
parser.add_argument("--negative_slope", type=float, default=0.01, help="Negative slope parameter for LeakyReLU")
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for optimizer")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size")

opt = parser.parse_args()

input_dim = opt.input_dim
z_dim = opt.z_dim
n_gpu = opt.n_gpu
neg_slope = opt.negative_slope
lr = opt.lr
batch_size = opt.batch_size
main_path = opt.main_path
num_epochs = opt.num_epochs
n_gpu = opt.n_gpu

# Dataset
dataset = MNIST(root=main_path, train=True, download=True,
                transform=transforms.Compose([ 
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]))

dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Load the model
sys.path.append(main_path)
from gan import *

device = torch.device("cpu")
g = gan.Generator(input_dim, z_dim, neg_slope).to(device)
d = gan.Discriminator(input_dim, neg_slope).to(device)

