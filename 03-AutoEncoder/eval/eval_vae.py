import torch, os 
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import sys
from model.vae import vae 

main_path = "/home/lukedinh/Desktop/Unsupervised-Learning/03-AutoEncoder"
sys.path.append(main_path)
