import torch 
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
import argparse

parser = argparse.ArgumentParser("GAN model")
parser.add_argument("--input_dim", tyoe=int, default=784, help="Input size of D")
parser.add_argument("--z_dim", type=int, default=784, help="Input size of G")
parser.add_argument("--neg_slope", type=float, default=0.1, help="Negative slope of ReLU")
parser.add_argument("n_gpu", type=int, default=0, help="Number of GPUs for training")

opt = parser.parse_args()
inp_dim = opt.input_dim
z_dim = opt.z_dim
neg_slope = opt.neg_slope
n_gpu = opt.n_gpu

# Model
class Generator(nn.Module):

    def __init__(self, inp_dim, z_dim, neg_slope):

        super(Generator, self).__init__()
        self.inp_dim = inp_dim
        self.z_dim = z_dim
        self.neg_slope = neg_slope 

        # Generator
        self.Generator = nn.Sequential(OrderedDict([ 
            ('layre1', nn.Linear(z_dim, 256)),
            ('relu1', nn.LeakyReLU(negative_slope=neg_slope, inplace=True)),
            ('layer2', nn.Linear(256, 512)),
            ('relu2', nn.LeakyReLU(neg_slope, inplace=True)),
            ('layer3', nn.Linear(512, 1024)),
            ('relu3', nn.LeakyReLU(neg_slope, inplace=True)),
            ('layer4', nn.Linear(1024, inp_dim)),
            ('sigmoid', nn.Sigmoid())
        ]))

        self._init_weights()

    def forward(self, x):
        return self.Generator(x)

    def _init_weights(self): 

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0,0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Discriminator(nn.Module):

    def ___init__(self, inp_dim, z_dim, neg_slope):
        super(Discriminator, self).__init__()
        self.inp_dim = inp_dim 
        self.neg_slope = neg_slope 

        self.Discriminator = nn.Sequential(OrderedDict([ 
            ('layer1', nn.Linear(inp_dim, 1024)),
            ('relu1', nn.LeakyReLU(neg_slope, inplace=True)),
            ('layer2', nn.Linear(1024, 512)),
            ('relu2', nn.LeakyReLU(neg_slope, inplace=True)),
            ('layer3', nn.Linear(512, 256)),
            ('relu3', nn.LeakyReLU(neg_slope, inplace=True)),
            ('layer4', nn.Linear(256, 1)),
            ('sigmoid', nn.Sigmoid())
        ]))

        self._init_weights()

    def forward(self, x):
        out = self.Discriminator(x)

        return out

    def _init_weights(self): 

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0,0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

# Test

# if __name__=="__main__":
#     model = GAN_model()
#     model.eval()
#     inputs = torch.randn([1, 100])
#     res_g_block = model.generator(inputs)
#     res_dis_block = model.discriminator(res_g_block)

#     print(res_g_block.shape)
#     print(res_dis_block.shape)
