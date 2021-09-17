import torch
from torch.utils.data import DataLoader
import numpy as np
from torch import nn 
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from collections import OrderedDict

def dataset(batch_size):

    valid_ratio = 0.2

    transforms = transforms.ToTensor()

    training_data = MNIST( 
        root = '.',
        train=True,
        download=True,
        transform= transforms
    )

    test_data = MNIST( 
        root='.',
        train=False,
        download=True,
        transform=transforms
    )

    # Seperate into training ad validation data

    num_train = len(training_data)
    idx = list(range(num_train))
    np.random.shuffle(idx)
    split = int(np.floor(valid_ratio * num_train))
    train_idx, val_idx = idx[split:], idx[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    # Load training data into batches
    train_loader = DataLoader(training_data, batch_size=batch_size, sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(training_data, batch_size=batch_size, sampler=val_sampler, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=0)

    return train_loader, val_loader, test_loader

# Define model

class cae(torch.nn.Module):

    def __init__(self, negative_slope=0.1):

        super(cae, self).__init__()

        # Encoder
        self.encoder= nn.Sequential(OrderedDict([ 
            ('layer1', nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)),
            ('relu1', nn.LeakyReLU(negative_slope, inplace=True)),
            ('pooling1', nn.MaxPool2d(kernel_size=2)),
            ('layer2', nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)),
            ('relu2', nn.LeakyReLU(negative_slope, inplace=True)),
            ('pooling2', nn.MaxPool2d(kernel_size=2)),
        ]))

        # Decoder
        self.decoder = nn.Sequential(OrderedDict([ 
            ('layer1', nn.ConvTranspose2d(32, 16, 4, stride=2)),
            ('relu1', nn.LeakyReLU(negative_slope, inplace=True)),
            ('layer2', nn.ConvTranspose2d(16, 3, 4, stride=2)),
            ('sigmoid', nn.Sigmoid()),
        ]))