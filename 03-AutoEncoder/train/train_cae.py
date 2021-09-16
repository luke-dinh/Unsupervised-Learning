import torch
from torch import nn 
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms 
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import pytorch_lightning as pl 


class LitCAE(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(OrderedDict([ 
            ('conv1', nn.Conv2d(3, 16, 3, padding=1)),
            ('leakyrelu', nn.LeakyReLU(negative_slope=0.1, inplace=True)),
            ('pooling1', nn.MaxPool2d(2,2)),
            ('conv2', nn.Conv2d(16, 32, 3, padding=1)),
            ('leakyrelu2', nn.LeakyReLU(negative_slope=0.1, inplace=True)),
            ('pooling2', nn.MaxPool2d(2,2)),
        ]))
        self.decoder = nn.Sequential(OrderedDict([ 
            ('conv1', nn.ConvTranspose2d(32, 16, 3, stride=2)),
            ('relu1', nn.LeakyReLU(negative_slope=0.1, inplace=True)),
            ('conv2', nn.ConvTranspose2d(16, 3, 3, stride=2)),
            # ('relu2', nn.LeakyReLU(negative_slope, inplace=True)),
            ('sigmoid', nn.Sigmoid()),
        ]))

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


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

dataset = CIFAR10( 
	root='.',
	train=True,
	download=False,
	transform=transforms.Compose([ 
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	])
)

dataloader = DataLoader(dataset, batch_size=128, num_workers=4, shuffle=True, pin_memory=True)
model = LitCAE()

# Training
trainer = pl.Trainer(precision=16, limit_train_batches=0.5)
trainer.fit(model, dataloader)