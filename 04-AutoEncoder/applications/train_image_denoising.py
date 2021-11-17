import argparse
from torch.utils.data import DataLoader
import sys, os
import torch.nn.functional as F
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch 
import numpy as np

parser = argparse.ArgumentParser("Denoising Images using AE")
parser.add_argument( 
    "--main_path",
    default="04-AutoEncoder",
    type=str,
    help="Path to your folder"
)
parser.add_argument("--save_path", 
                    default="04-AutoEncoder/checkpoint",
                    type=str,
                    help="Path to save weights")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size")

opt = parser.parse_args()
main_path = opt.main_path
save_path = opt.save_path
batch_size = opt.batch_size

sys.path.append(main_path)
# Load model

from applications.model.conv_ae import conv_ae
model = conv_ae(negative_slope=0.1)

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
    root=main_path,
    train=True,
    download=True,
    transform=transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Normalize((0.137,), (0.226,))
    ])
)

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=4, shuffle=True)

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
for epoch in range(1,n_epochs+1):

    for data in train_loader:

        images, _ = data
        # flatten_img = images.view(images.shape[0], -1)

        ## Create noisy data
        noisy_img = images + noise_factor * torch.randn(*images.shape)
        # Clip the images in range 0,1
        noisy_img = np.clip(noisy_img, 0. , 1.)
        # noisy_img = noisy_img.view(noisy_img.shape[0], -1)

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

    # ImproveChecker
    print("[Epoch %.3d] Loss: %.6f" % (epoch,loss.item()))
    if improve_checker.check(loss.item()):
        checkpoint = dict( 
            epoch=epoch,
            loss = loss.item(),
            state_dict=model.state_dict(),
            optimizer = optimizer.state_dict(), 
        )
        save_file = os.path.join(save_path, "denoise_ae.pth")
        torch.save(checkpoint, save_file)
     