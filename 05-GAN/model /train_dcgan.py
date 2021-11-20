import sys
import torch
import torch.nn as nn
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
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for optimizer")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")

opt = parser.parse_args()

num_channels = opt.num_channels
input_dim = opt.input_dim
feature_map = opt.feature_map
n_gpu = opt.n_gpu
neg_slope = opt.negative_slope
lr = opt.lr
batch_size = opt.batch_size
main_path = opt.main_path
num_epochs = opt.num_epochs
n_gpu = opt.n_gpu

sys.path.append(main_path)
from dcgan import generator, discriminator, _init_weights

device = torch.device("cpu")
generator = generator(input_dim, feature_map, num_channels, neg_slope, n_gpu).to(device)
discriminator = discriminator(num_channels, feature_map, neg_slope, n_gpu).to(device)

# Apply initialize weights
generator.apply(_init_weights)
discriminator.apply(_init_weights)

# Define loss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, input_dim, 1, 1, device=device)

# Establish real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizer for G and D
optimG = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimD = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Prepare the dataset
dataset = CIFAR10(
    root=main_path,
    train=True,
    download=True,
    transform=transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
)

dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4)