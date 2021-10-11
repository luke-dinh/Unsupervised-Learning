import argparse
import os, sys
import numpy as np

import torch 
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader 
import torchvision.transforms as transforms