import torch
import torch.nn as nn
import argparse

from torch.nn.modules.activation import LeakyReLU
from torch.nn.modules.conv import ConvTranspose2d

# Args parser
parser = argparse.ArgumentParser("DCGAN Implementation")

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

class generator(nn.Module):

    def __init__(self, input_dim, feature_map, num_channels, neg_slope, n_gpu):
        super(generator, self).__init__()

        self.n_gpu = n_gpu
        self.input_dim = input_dim
        self.feature_map = feature_map
        self.num_channels = num_channels
        self.neg_slope = neg_slope

        self.generator = nn.Sequential( 

            # First Block
            nn.ConvTranspose2d(input_dim, feature_map * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_map * 8),
            nn.LeakyReLU(neg_slope, inplace=True),

            # Second Block
            nn.ConvTranspose2d(feature_map * 8, feature_map *4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map * 4),
            nn.LeakyReLU(neg_slope, inplace=True),

            # Third Block
            nn.ConvTranspose2d(feature_map * 4, feature_map * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map * 2),
            nn.LeakyReLU(neg_slope, inplace=True),

            # Forth block
            nn.ConvTranspose2d(feature_map * 2, feature_map, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map, inplace=True),
            nn.LeakyReLU(neg_slope),

            # Final block
            nn.ConvTranspose2d(feature_map, num_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.generator(x)

class discriminator(nn.Module):

    def __init__(self, num_channels, feature_map, neg_slope, n_gpu):
        super(discriminator, self).__init__()

        self.num_channels = num_channels
        self.feature_map = feature_map
        self.neg_slope = neg_slope
        self.n_gpu = n_gpu

        self.discriminator = nn.Sequential( 

            # First block
            nn.Conv2d(num_channels, feature_map, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(neg_slope, inplace=True),

            # Second block
            nn.Conv2d(feature_map, feature_map * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map * 2),
            nn.LeakyReLU(neg_slope, inplace=True),

            # Third block
            nn.Conv2d(feature_map * 2, feature_map * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map * 4),
            nn.LeakyReLU(neg_slope, inplace=True),
        )