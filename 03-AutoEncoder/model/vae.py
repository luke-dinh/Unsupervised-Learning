from collections import OrderedDict

import torch
from torch import nn 
import torch.nn.functional as F

# Define model

class vae(nn.Module):

    def __init__(self, in_dims=784, encod_dims=64, negative_slope=0.1):

        super(vae, self).__init__
        # Encoder

        self.encoder = nn.Sequential(OrderedDict([ 
            ('layer1', nn.Linear(in_dims, 512)),
            ('relu1', nn.LeakyReLU(negative_slope, inplace=True)),
            ('layer2', nn.Linear(512, 256)),
            ('relu2', nn.LeakyReLU(negative_slope, inplace=True)),
            ('layer3', nn.Linear(256, 128)),
            ('relu3', nn.LeakyReLU(negative_slope, inplace=True)),
         ]))

        self.fc_muy = nn.Linear(128, encod_dims)
        self.fc_var = nn.Linear(128, encod_dims)

        # Decoder

        self.decoder = nn.Sequential(OrderedDict([ 
            ('layer1', nn.Linear(encod_dims, 128)),
            ('relu1', nn.LeakyReLU(negative_slope, inplace=True)),
            ('layer2', nn.Linear(128, 256)),
            ('relu2', nn.LeakyReLU(negative_slope, inplace=True)),
            ('layer3', nn.Linear(256, 512)),
            ('relu3', nn.LeakyReLU(negative_slope, inplace=True)),
            ('layer4', nn.Linear(512, in_dims)),
            ('sigmoid', nn.Sigmoid()), 
        ]))

        self._init_weights()

    def _init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def represent(self, x):
        h = self.encoder(x)
        muy, logvar = self.fc_muy(h), self.fc_var(h)
        z = self._reparameterize(muy, logvar)
        return z 

    def _reparameterize(self, muy, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*muy.size()).type_as(muy)
        z = muy + std * esp 
        return z 

    