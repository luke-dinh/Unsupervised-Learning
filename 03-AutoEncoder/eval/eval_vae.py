import torch, os 
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import sys
from model.vae import VAE

# Change path
main_path = "/home/lukedinh/Desktop/Unsupervised-Learning/03-AutoEncoder"
sys.path.append(main_path)

model = VAE(in_dims=784, encod_dims=64)
model.load_state_dict(torch.load(main_path + '/checkpoint/vae.pth', map_location="cpu")['state_dict'])
model.eval()