import torch
import torch.nn as nn
import torch.nn.functional as F

###########################
###   Generator Model   ###
###########################

class Generator(nn.Module):
    def __init__(self, z_dim=128, cc_dim=2, dc_dim=10):
        super(Generator, self).__init__()
        i_dim = z_dim + cc_dim + dc_dim
        self.main = nn.Sequential(
            # -> 1024 x 4 x 4
            nn.ConvTranspose2d(i_dim, 128, 4, 1, 0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # -> 512 x 8 x 8
            nn.ConvTranspose2d( 128,  64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # -> 256 x 16 x 16
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # -> 128 x 28 x 28
            nn.ConvTranspose2d(32, 1, 2, 2, 2),
            nn.Tanh()
    )
    def forward(self, z):
        out = self.main(z)
        return out

###########################
### Discriminator Model ###
###########################

class Discriminator(nn.Module):
    def __init__(self, cc_dim = 1, dc_dim = 10):
        super(Discriminator, self).__init__()
        self.cc_dim = cc_dim
        self.dc_dim = dc_dim
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 3),
            nn.LeakyReLU(0.1, inplace=True),

            # [-1, 256, 8, 8]
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),

            # [-1, 512, 4, 4]
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            # [10, 512, 1, 1]
            nn.Conv2d(256, 512, 4, 2, 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),

        )
        self.linear = nn.Sequential(
            nn.Conv2d(512, 1, 1)
        )
    def front_end(self, x):
        return self.conv(x)
    
    def forward(self, x):
        out = self.front_end(x)
        out = self.linear(out).squeeze()
        return torch.sigmoid(out) #prob of being real

class Q(nn.Module):
    def __init__(self, cc_dim=2, dc_dim=10):
        super(Q, self).__init__()
        self.conv = nn.Conv2d(512, 128, 1, bias=False)
        self.bn = nn.BatchNorm2d(128)
        self.lReLU = nn.LeakyReLU(0.1, inplace=True)
        
        self.conv_disc = nn.Conv2d(128, dc_dim, 1)
        self.conv_mu   = nn.Conv2d(128, cc_dim, 1)
        self.conv_var  = nn.Conv2d(128, cc_dim, 1)
    def forward(self, x):
        y = self.lReLU(self.bn(self.conv(x)))
        
        disc_logits = self.conv_disc(y).squeeze()
        mu = self.conv_mu(y).squeeze()
        var = self.conv_var(y).squeeze().exp()

        return disc_logits, mu, var