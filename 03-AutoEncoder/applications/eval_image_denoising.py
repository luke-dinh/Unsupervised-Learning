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
parser.add_argument("--batch_size", default=64, type=int, help="Batch size")

opt = parser.parse_args()
main_path = opt.main_path
batch_size=opt.batch_size

sys.path.append(main_path)
# Load model
from model.ae import AE

model = AE(in_dims=784, encod_dims=64)
model.load_state_dict(torch.load(main_path + "/checkpoint/denoise_ae.pth", map_location="cpu")["state_dict"])
model.eval()

# Load dataset

test_data = MNIST( 
    root=main_path, train=False,
    download=True, 
    transform= transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Normalize((0.137, ), (0.226, ))
    ]))

test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=4)