import torch
import torch.nn as nn
import argparse

# Args parser
parser = argparse.ArgumentParser("DCGAN Implementation")

parser.add_argument("--num_channels", default=3, type=int, help="Number of channels")
parser.add_argument("--input_dim", default=100, type=int, help="Number of dimensions of inputs")
parser.add_argument("--feature_map", default=32, type=int, help="Size of feature maps")
parser.add_argument("--n_gpu", default=0, type=int, help="Number of GPUs (0 for CPU)")

opt = parser.parse_args()
num_channels = opt.num_channels
input_dim = opt.input_dims
feature_map = opt.feature_map
n_gpu = opt.n_gpu