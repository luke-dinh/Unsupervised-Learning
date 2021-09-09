import torch
from torch import nn 
from collections import OrderedDict
from torch.optim import Adam
import numpy as np
from torchvision import datasets
from torch.utils.data import dataloader
import torchvision.transforms as transforms 
import torch.nn.functional as F
from model import ae 
import os 

def loss_fn(output_x, x):
    return F.mse_loss(output_x, x)

# In case this module cannot load the model from model:
# class ae(nn.Module):
#     def __init__(self, in_dims=784, encod_dims=64, negative_slope=0.1):

#         super(ae, self).__init__()

#         # Encoder
#         self.encoder = nn.Sequential(OrderedDict([ 
#             ('layer1', nn.Linear(in_dims, 512)),
#             ('relu1', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
#             ('layer2', nn.Linear(512, 256)),
#             ('relu2', nn.LeakyReLU(negative_slope, inplace=True)),
#             ('layer3', nn.Linear(256, 128)),
#             ('relu3', nn.LeakyReLU(negative_slope, inplace=True)),
#             ('layer4', nn.Linear(128, encod_dims)),
#             ('relu4', nn.LeakyReLU(negative_slope, inplace=True)),
#         ]))

#         # Decoder
#         self.decoder = nn.Sequential(OrderedDict([ 
#             ('layer1', nn.Linear(encod_dims, 128)),
#             ('relu1', nn.LeakyReLU(negative_slope, inplace=True)),
#             ('layer2', nn.Linear(128, 256)),
#             ('relu2', nn.LeakyReLU(negative_slope, inplace=True)),
#             ('layer3', nn.Linear(256, 512)),
#             ('relu3', nn.LeakyReLU(negative_slope, inplace=True)),
#             ('layer4', nn.Linear(512, in_dims)),
#             ('sigmoid', nn.Sigmoid()),
#         ]))

#         self._init_weights()

#     def forward(self, x):

#         z = self.encoder(x)
#         out = self.decoder(z)

#         return out

#     def _init_weights(self):

#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 m.weight.data.normal_(0, 0.01)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

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

# Or using class ae in this module:
# model = ae(in_dims=784, encod_dims=64)

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

# Making directory to save weights
new_path = '03-AutoEncoder/checkpoint'
if not os.path.exists(new_path):
	os.makedirs(new_path)

model.train()
for epoch in range(1, 50):
	for i, (imgs, _) in enumerate(dataloader):
		# Prepare input
		inputs = imgs.view(imgs.shape[0], -1)
		# inputs = inputs.cuda()

		# Train
		optimizer.zero_grad()
		outputs = model(inputs)
		loss = loss_fn(outputs, inputs)
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
		save_file = os.path.join(new_path, "/ae.pth")
		torch.save(checkpoint, save_file)
		print("Best checkpoint is saved at %s" % (save_file))