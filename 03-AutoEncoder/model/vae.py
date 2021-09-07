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