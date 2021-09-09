import torch
from torch import nn 
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader 
import torchvision.transforms as transforms 
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np

# Define model
class VAE(nn.Module):
    def __init__(self, in_dims=784, encod_dims=64, negative_slope=0.1):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(OrderedDict([
            ('layer1', nn.Linear(in_dims, 512)),
            ('relu1', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
            ('layer2', nn.Linear(512, 256)),
            ('relu2', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
            ('layer3', nn.Linear(256, 128)),
            ('relu3', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
        ]))
        self.fc_mu = nn.Linear(128, encod_dims)
        self.fc_var = nn.Linear(128, encod_dims)
        # Decoder
        self.decoder = nn.Sequential(OrderedDict([
            ('layer1', nn.Linear(encod_dims, 128)),
            ('relu1', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
            ('layer2', nn.Linear(128, 256)),
            ('relu2', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
            ('layer3', nn.Linear(256, 512)),
            ('relu3', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
            ('layer4', nn.Linear(512, in_dims)),
            ('sigmoid', nn.Sigmoid()),
        ]))
        self._init_weights()

    def forward(self, x):
        if self.training:
            h = self.encoder(x)
            mu, logvar = self.fc_mu(h), self.fc_var(h)
            z = self._reparameterize(mu, logvar)
            y = self.decoder(z)
            return y, mu, logvar
        else:
            z = self.represent(x)
            y = self.decoder(z)
            return y

    def represent(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_var(h)
        z = self._reparameterize(mu, logvar)
        return z

    def _reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).type_as(mu)
        z = mu + std * esp
        return z

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

# Or 
# from model import vae
# model = vae.vae(in_dims = in_dims=784, encod_dims=64, negative_slope=0.1)

# Load checker

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
model = VAE(in_dims=784, encod_dims=64)

