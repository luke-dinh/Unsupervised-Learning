import sys
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import argparse

parser = argparse.ArgumentParser("Train DCGAN")
parser.add_argument("--main_path", type=str, default="05-GAN", help="Main Path")
parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs for training")
parser.add_argument("--n_gpu", type=int, default=0, help="Number of GPUs for training")
parser.add_argument("--num_channels", default=3, type=int, help="Number of channels")
parser.add_argument("--input_dim", default=100, type=int, help="Number of dimensions of inputs")
parser.add_argument("--feature_map", default=32, type=int, help="Size of feature maps")
parser.add_argument("--negative_slope", type=float, default=0.01, help="Negative slope parameter for LeakyReLU")
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for optimizer")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")

opt = parser.parse_args()

num_channels = opt.num_channels
input_dim = opt.input_dim
feature_map = opt.feature_map
n_gpu = opt.n_gpu
neg_slope = opt.negative_slope
lr = opt.lr
batch_size = opt.batch_size
main_path = opt.main_path
num_epochs = opt.num_epochs
n_gpu = opt.n_gpu

sys.path.append(main_path)
from dcgan import generator, discriminator, _init_weights

device = torch.device("cpu")
generator = generator(input_dim, feature_map, num_channels, neg_slope, n_gpu).to(device)
discriminator = discriminator(num_channels, feature_map, neg_slope, n_gpu).to(device)

# Apply initialize weights
generator.apply(_init_weights)
discriminator.apply(_init_weights)

# Define loss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, input_dim, 1, 1, device=device)

# Establish real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizer for G and D
optimG = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimD = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Prepare the dataset
dataset = CIFAR10(
    root=main_path,
    train=True,
    download=True,
    transform=transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
)

dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# training
img_list = []
G_loss = []
D_loss = []
iters = 0

for epoch in range(num_epochs):

    # For each batch in DataLoader
    for i, data in enumerate(DataLoader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        discriminator.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

        # Forward pass to D
        output = discriminator(real_cpu).view(-1)
        # Calculate loss
        error_D_real = criterion(output, label)
        # Upgrade gradient
        error_D_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, input_dim, 1, 1, device=device)
        # Generate fake images
        fake_G = generator(noise)
        label.fill_(fake_label)
        # Classify fake batch with D
        output = discriminator(fake_G.detach()).view(-1)
        error_D_fake = criterion(output, label)
        error_D_fake.backward()
        D_G_z1 = output.mean().item()
        
        