import torch
import torch.nn as nn

class generator(nn.Module):

    def __init__(self, input_dim, feature_map, num_channels, neg_slope, n_gpu):
        super(generator, self).__init__()

        # self.n_gpu = n_gpu
        # self.input_dim = input_dim
        # self.feature_map = feature_map
        # self.num_channels = num_channels
        # self.neg_slope = neg_slope

        self.generator = nn.Sequential( 

            # First Block
            nn.ConvTranspose2d(input_dim, feature_map * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_map * 8),
            nn.LeakyReLU(neg_slope, inplace=True),

            # Second Block
            nn.ConvTranspose2d(feature_map * 8, feature_map *4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map * 4),
            nn.LeakyReLU(neg_slope, inplace=True),

            # Third Block
            nn.ConvTranspose2d(feature_map * 4, feature_map * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map * 2),
            nn.LeakyReLU(neg_slope, inplace=True),

            # Forth block
            nn.ConvTranspose2d(feature_map * 2, feature_map, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map),
            nn.LeakyReLU(neg_slope, inplace=True),

            # Final block
            nn.ConvTranspose2d(feature_map, num_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.generator(x)

class discriminator(nn.Module):

    def __init__(self, num_channels, feature_map, neg_slope, n_gpu):
        super(discriminator, self).__init__()

        # self.num_channels = num_channels
        # self.feature_map = feature_map
        # self.neg_slope = neg_slope
        # self.n_gpu = n_gpu

        self.discriminator = nn.Sequential( 

            # First block
            nn.Conv2d(num_channels, feature_map, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(neg_slope, inplace=True),

            # Second block
            nn.Conv2d(feature_map, feature_map * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map * 2),
            nn.LeakyReLU(neg_slope, inplace=True),

            # Third block
            nn.Conv2d(feature_map * 2, feature_map * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map * 4),
            nn.LeakyReLU(neg_slope, inplace=True),

            # Forth block
            nn.Conv2d(feature_map * 4, feature_map * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map * 8),
            nn.LeakyReLU(neg_slope, inplace=True),

            # Final block
            nn.Conv2d(feature_map * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.discriminator(x)

def _init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)