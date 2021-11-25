import sys
import argparse
import torch
from torch import optim
import torch.nn as nn
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Parser arguments
parser = argparse.ArgumentParser()
parser.add_argument("--main_path", type=str, default="05-GAN/GAN_original", help="Main path")
parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs for training")
parser.add_argument("--n_gpu", type=int, default=0, help="Number of GPUs for training")
parser.add_argument("--input_dim", default=784, type=int, help="MNIST dim")
parser.add_argument("--z_dim", default=100, type=int, help="Number of dimensions of inputs")
parser.add_argument("--negative_slope", type=float, default=0.01, help="Negative slope parameter for LeakyReLU")
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for optimizer")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size")

opt = parser.parse_args()

input_dim = opt.input_dim
z_dim = opt.z_dim
n_gpu = opt.n_gpu
neg_slope = opt.negative_slope
lr = opt.lr
batch_size = opt.batch_size
main_path = opt.main_path
num_epochs = opt.num_epochs
n_gpu = opt.n_gpu

# Dataset
dataset = MNIST(root=main_path, train=True, download=True,
                transform=transforms.Compose([ 
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))

dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Load the model
sys.path.append(main_path)
from gan import Generator, Discriminator

if n_gpu != 0:
    device = torch.device("gpu")
else:
    device = torch.device("cpu")

g = Generator(input_dim, z_dim, neg_slope).to(device)
d = Discriminator(input_dim, neg_slope).to(device)

# Training

## 1. Define losses
criterion = nn.BCELoss()

## 2. Create batch of latent vectors that we will use to visualize
## the progression of the generator
fixed_noise = torch.randn(28, input_dim, 1, 1, device=device)

## 3. Establish real and fake labels for training
real_label = 1
fake_label = 0

## 4. Setup Adam optimizer for G and D
optimG = torch.optim.Adam(g.parameters(), lr=lr, betas=(0.5, 0.999))
optimD = torch.optim.Adam(d.parameters(), lr=lr, betas=(0.5, 0.999))

## 5. Training
img_list = []
g_loss = []
d_loss = []
iters = 0

for epoch in range(num_epochs):

    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        d.zero_grad()
        # Format batch

        real_data_cpu = data[0].to(device)
        b_size = real_data_cpu.size(0)
        label = torch.full((b_size, ), real_label, dtype=torch.float, device=device)

        # Load the data to D
        ## 1. Classify real data
        output = d(real_data_cpu).view(-1)
        error_D_real = criterion(output, label)
        error_D_real.backward()
        D_x = output.mean().item()

        ## 2. Discriminate the data from G (Train with all fake batch)
        noise = torch.randn(b_size, input_dim, 1, 1, device=device)
        # Generate fake images
        fake_data_cpu = g(noise)
        label.fill_(fake_label)

        # 3. Classify fake batch
        output = d(fake_data_cpu.detach()).view(-1)
        error_D_fake = criterion(output,label)
        error_D_fake.backward()
        D_G_z1 = output.mean().item()

        # 4. Loss from D
        error_D = error_D_real + error_D_fake

        # 5. Update D
        optimD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        g.zero_grad()

        ## 1. Load the data to G
        label.fill_(real_label)

        # 2. Perform another forward pass of all-fake batch through D
        output = d(fake_data_cpu).view(-1)

        # 3. Loss
        error_G = criterion(output, label)
        error_G.backward()
        D_G_z2 = output.mean().item()

        # 4. Update G
        optimG.step()