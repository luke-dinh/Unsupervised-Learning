import argparse
from torch.utils.data import DataLoader
import sys
import torch.nn.functional as F
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch 
import numpy as np

parser = argparse.ArgumentParser("Denoising Images using AE")
parser.add_argument( 
    "--main_path",
    default="/home/lukedinh/Desktop/Unsupervised-Learning/03-AutoEncoder",
    type=str,
    help="Path to your folder"
)
parser.add_argument("--save_weight", 
                    default="/home/lukedinh/Desktop/Unsupervised-Learning/03-AutoEncoder/checkpoint",
                    type=str,
                    help="Path to save weights")

opt = parser.parse_args()
main_path = opt.main_path
save_path = opt.save_path
sys.path.append(main_path)

# Load model
from model.ae import AE
model = AE(in_dims=784, encod_dims=64)

# Checker
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

# Dataset

train_data = MNIST( 
    root='.',
    train=True,
    download=True,
    transform=transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Normalize((0.137,), (0.226,))
    ])
)

train_loader = DataLoader(dataset=train_data, batch_size=64, num_workers=4, shuffle=True)

# Loss Function
def loss_fn(output_x, x):
    return F.mse_loss(output_x, x)

# Optimizer 
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Improve Checker
improve_checker = ImproveChecker(mode='min')

# Training

n_epochs = 20
noise_factor = 0.4

model.train()
for i in range(1,n_epochs+1):

    for data in train_loader:

        images, _ = data

        ## Create noisy data
        noisy_img = images + noise_factor * torch.randn(*images.shape)
        # Clip the images in range 0,1
        noisy_img = np.clip(noisy_img, 0. , 1.)

        # Train
        # Clear all of the gradients
        optimizer.zero_grad() 
        # Forward pass
        outputs = model(noisy_img)
        # Loss function
        loss = loss_fn(outputs, images)
        # Backpropagation
        loss.backward()
        optimizer.step()