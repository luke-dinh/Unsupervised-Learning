import torch, os 
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import argparse
import matplotlib.pyplot as plt
import sys

# Change path

parser = argparse.ArgumentParser(description="VAE Evaluation")
parser.add_argument("--main_path", default="/home/lukedinh/Desktop/Unsupervised-Learning/03-AutoEncoder", 
                        type=str, help="Define main path")
opt = parser.parse_args()
main_path = opt.main_path
sys.path.append(main_path)

# Load model

from model.vae import VAE
model = VAE(in_dims=784, encod_dims=64)
model.load_state_dict(torch.load(main_path + '/checkpoint/vae.pth', map_location="cpu")['state_dict'])
model.eval()

# Dataloader
data_dir = main_path + '/datasets'

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

dataset = MNIST(root=data_dir, train=False, download=True, 
                transform=transforms.Compose([ 
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,),(0.3081))
                ]))

# Generate Image
with torch.no_grad():
    inputs = torch.cat([torch.randn([1,64]) for i in range(32)], dim=0)
    outputs = model.decoder(inputs)
    outputs = outputs.view(-1,1,28,28)
    grid_img = make_grid(outputs.data, nrow=8, normalize=True).cpu().numpy().transpose((1,2,0))

plt.figure(figsize=(20,20))
plt.imshow(grid_img)
plt.axis('off')
plt.title('Generated Images from VAE')
plt.show()