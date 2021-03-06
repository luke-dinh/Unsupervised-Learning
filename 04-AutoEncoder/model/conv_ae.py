from collections import OrderedDict

import torch
from torch import nn 
import torch.nn.functional as F
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

class conv_ae(nn.Module):

    def __init__(self, negative_slope=0.1):

        super(conv_ae, self).__init__()

        # Encoder
        self.encoder= nn.Sequential(OrderedDict([ 
            ('layer1', nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)),
            ('relu1', nn.LeakyReLU(negative_slope, inplace=True)),
            ('pooling1', nn.MaxPool2d(kernel_size=2)),
            ('layer2', nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)),
            ('relu2', nn.LeakyReLU(negative_slope, inplace=True)),
            ('pooling2', nn.MaxPool2d(kernel_size=2)),
        ]))

        # Decoder
        self.decoder = nn.Sequential(OrderedDict([ 
            ('layer1', nn.ConvTranspose2d(32, 16, 2, stride=2)),
            ('relu1', nn.LeakyReLU(negative_slope, inplace=True)),
            ('layer2', nn.ConvTranspose2d(16, 1, 2, stride=2)),
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

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)

        return out 

# Test 
if __name__ == "__main__":
    model = conv_ae(negative_slope=0.1)
    model.eval()

    test_data = MNIST(root='.', train=False, download=True, 
                        transform=transforms.Compose([ 
                            transforms.ToTensor(),
                            transforms.Normalize((0.137, ), (0.229, ))
                        ]))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, num_workers=4, shuffle=True)

    test_iter = iter(test_loader)
    images, labels = test_iter.next()

    outputs = model(images)
    print(outputs.shape)
