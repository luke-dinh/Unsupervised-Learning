import argparse
import os, sys
import numpy as np
from numpy.core.defchararray import mod

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
parser.add_argument("--noise_factor", default=0.4, type=float, help="Noise factor")

opt = parser.parse_args()
main_path = opt.main_path
batch_size=opt.batch_size
noise_factor = opt.noise_factor

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

dataiter = iter(test_loader)
images, labels = dataiter.next()

# Evaluate
noisy_img = images + noise_factor *torch.randn(*images.shape)
noisy_img = np.clip(noisy_img, 0., 1.)
noisy_img = noisy_img.view(noisy_img.shape[0], -1)

outputs= model(noisy_img)