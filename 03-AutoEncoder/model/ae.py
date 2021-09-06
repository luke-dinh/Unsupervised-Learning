from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm1d

#------------
# Define model
#------------

class ae(nn.Module):
    def __init__(self, in_dims=784, encod_dims=64, negative_slope=0.1):

        super(ae, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(OrderedDict([ 
            ('layer1', nn.Linear(in_dims, 512)),
            ('relu1', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
            ('layer2', nn.Linear(512, 256)),
            ('relu2', nn.LeakyReLU(negative_slope, inplace=True)),
            ('layer3', nn.Linear(256, 128)),
            ('relu3', nn.LeakyReLU(negative_slope, inplace=True)),
            ('layer4', nn.Linear(128, encod_dims)),
            ('relu4', nn.LeakyReLU(negative_slope, inplace=True)),
        ]))

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

    def forward(self, x):

        z = self.encoder(x)
        out = self.decoder(z)

        return out

    def _init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

#------------
# Test
#------------

if __name__ == "__main__":
    model = ae(in_dims=784, encod_dims=64, negative_slope=0.1)
    model.eval()

    input = torch.rand([1, 784])
    output = model(input)
    print(output.shape)