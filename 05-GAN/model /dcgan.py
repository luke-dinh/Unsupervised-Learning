import torch
import torch.nn as nn
import argparse

from torch.nn.modules.activation import LeakyReLU

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

    def __init__(self, input_dim, feature_map, neg_slope, n_gpu):
        super(generator, self).__init__()

        self.n_gpu = n_gpu
        self.input_dim = input_dim
        self.feature_map = feature_map
        self.neg_slope = neg_slope

        self.generator = nn.Sequential( 
            # First Block
            nn.ConvTranspose2d(input_dim, feature_map * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_map * 8),
            nn.LeakyReLU(neg_slope)
        )