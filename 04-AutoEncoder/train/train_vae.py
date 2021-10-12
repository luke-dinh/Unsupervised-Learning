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

# Loss function
def loss_fn(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

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

dataset = MNIST(root='.', train=True, download=True,
                            transform= transforms.Compose([ 
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ]))

dataloader = DataLoader(dataset, batch_size=128, num_workers=4, shuffle=True, pin_memory=True)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Improve Checker
improvechecker = ImproveChecker(mode='min')

# Making directory to save weights
new_path = '03-AutoEncoder/checkpoint'
if not os.path.exists(new_path):
	os.makedirs(new_path)


model.train()
for epoch in range(1, 301):
	for i, (imgs, _) in enumerate(dataloader):
		# Prepare input
		inputs = imgs.view(imgs.shape[0], -1)
		# inputs = inputs.cuda()

		# Train
		optimizer.zero_grad()
		outputs, mu, logvar = model(inputs)
		loss = loss_fn(outputs, inputs, mu, logvar)
		loss.backward()
		optimizer.step()

	# ImproveChecker
	print("[EPOCH %.3d] Loss: %.6f" % (epoch, loss.item()))
	if improvechecker.check(loss.item()):
		checkpoint = dict(
			epoch=epoch,
			loss=loss.item(),
			state_dict=model.state_dict(),
			optimizer=optimizer.state_dict(),
		)
		save_file = os.path.join(new_path, "vae.pth")
		torch.save(checkpoint, save_file)
		print("Best checkpoint is saved at %s" % (save_file))
