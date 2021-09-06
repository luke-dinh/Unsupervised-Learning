import torch
from torch.optim import Adam
import numpy as np
from torchvision import datasets
from torchvision.utils import make_grid
from torch.utils.data import dataloader
import torchvision.transforms as transforms 
import torch.nn.functional as F
from model import ae

def loss_fn(output_x, x):
    return F.mse_loss(output_x, x)


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

# Load model
model = ae.ae(in_dims=784, encod_dims=64)

# Load dataset
dataset = datasets.MNIST(root='.', train=True, download=True,
                            transform= transforms.Compose([ 
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ]))

dataloader = dataloader.DataLoader(dataset, batch_size=128, num_workers=4, shuffle=True, pin_memory=True)

# Optimizer
optimizer = Adam(model.parameters(), lr=1e-3)

# Improve Checker
improvechecker = ImproveChecker(mode='min')