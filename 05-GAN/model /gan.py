import torch 
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

# Model
class GAN_model(nn.Module):

    def __init__(self, inp_dim=784, z_dim=100, negative_slope=0.1):

        super(GAN_model, self).__init__()

        # Generator
        self.generator = nn.Sequential(OrderedDict([ 
            ('layre1', nn.Linear(z_dim, 256)),
            ('relu1', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
            ('layer2', nn.Linear(256, 512)),
            ('relu2', nn.LeakyReLU(negative_slope, inplace=True)),
            ('layer3', nn.Linear(512, 1024)),
            ('relu3', nn,LeakyReLU(negative_slope, inplace=True)),
            ('layer4', nn.Linear(1024, inp_dim)),
            ('sigmoid', nn.Sigmoid())
        ]))

        self.discriminator = nn.Sequential(OrderedDict([ 
            ('layer1', nn.Linear(inp_dim, 1024)),
            ('relu1', nn.LeakyReLU(negative_slope, inplace=True)),
            ('layer2', nn.Linear(1024, 512)),
            ('relu2', nn.LeakyReLU(negative_slope, inplace=True)),
            ('layer3', nn.Linear(512, 256)),
            ('relu3', nn.LeakyReLU(negative_slope, inplace=True)),
            ('layer4', nn.Linear(256, 1)),
            ('sigmoid', nn.Sigmoid())
        ]))

        self._init_weights()

    def forward(self, x):

        gen = self.generator(x)
        out = self.discriminator(gen)

        return gen, out

    def _init_weights(self): 

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0,0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()