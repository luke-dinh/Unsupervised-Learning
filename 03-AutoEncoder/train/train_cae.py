import torch
from torch import nn 
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader 
import torchvision.transforms as transforms 
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import os 

# Define model 
class cae(nn.Module):

    def __init__(self, negative_slope=0.1):
        super(cae, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(OrderedDict([ 
            ('conv1', nn.Conv2d(3, 16, 3, padding=1)),
            ('leakyrelu', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
            ('pooling1', nn.MaxPool2d(2,2)),
            ('conv2', nn.Conv2d(16, 32, 3, padding=1)),
            ('leakyrelu2', nn.LeakyReLU(negative_slope, inplace=True)),
            ('pooling2', nn.MaxPool2d(2,2)),
        ]))

        #Decoder
        self.decoder = nn.Sequential(OrderedDict([ 
            ('conv1', nn.ConvTranspose2d(32, 16, 3, stride=2)),
            ('relu1', nn.LeakyReLU(negative_slope, inplace=True)),
            ('conv2', nn.ConvTranspose2d(16, 3, 3, stride=2)),
            # ('relu2', nn.LeakyReLU(negative_slope, inplace=True)),
            ('sigmoid', nn.Sigmoid()),
        ]))

class ImproveChecker():
	def __init__(self, mode='min', best_val=None):
		assert mode in ['min', 'max']
		self.mode = mode
		if best_val is not None:
			self.best_val = best_val
		else:
			if self.mode=='min':
				self.best_val = np.inf
			elif self.mode=='max':
				self.best_val = 0.0

	def check(self, val):
		if self.mode=='min':
			if val < self.best_val:
				print("[%s] Improved from %.4f to %.4f" % (self.__class__.__name__, self.best_val, val))
				self.best_val = val
				return True
			else:
				print("[%s] Not improved from %.4f" % (self.__class__.__name__, self.best_val))
				return False
		else:
			if val > self.best_val:
				print("[%s] Improved from %.4f to %.4f" % (self.__class__.__name__, self.best_val, val))
				self.best_val = val
				return True
			else:
				print("[%s] Not improved from %.4f" % (self.__class__.__name__, self.best_val))
				return False