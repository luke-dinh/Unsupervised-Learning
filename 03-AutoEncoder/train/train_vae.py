import torch
from torch import nn 
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader 
import torchvision.transforms as transforms 
import torch.nn.functional as F
from collections import OrderedDict

# Define model
class VAE(nn.Module):
    def __init__(self, in_dims=784, hid_dims=100, negative_slope=0.1):
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
        self.fc_mu = nn.Linear(128, hid_dims)
        self.fc_var = nn.Linear(128, hid_dims)
        # Decoder
        self.decoder = nn.Sequential(OrderedDict([
            ('layer1', nn.Linear(hid_dims, 128)),
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

