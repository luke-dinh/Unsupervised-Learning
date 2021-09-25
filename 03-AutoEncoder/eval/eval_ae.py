import torch, os
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader 
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import sys

# Change this path based on your filepath
ult_path = '/home/lukedinh/Desktop/Unsupervised-Learning/03-AutoEncoder'
sys.path.append(ult_path)
from model.ae import AE 

# Initialize the model
model = AE(in_dims=784, encod_dims=64)
model.load_state_dict(torch.load(ult_path + '/checkpoint/ae.pth', map_location='cpu')['state_dict'])
model.eval()