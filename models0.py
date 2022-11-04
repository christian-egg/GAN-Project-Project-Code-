from tkinter.tix import IMAGE
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

GEN_OUT_MULT = 32
OUTPUT_DIMENSION = 32
Z_DIM = 100
IMAGE_CHANNELS = 3
DISCRIM_OUT_MULT = 32

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is (z dim)
            nn.ConvTranspose2d(Z_DIM, GEN_OUT_MULT, kernel_size=8, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(GEN_OUT_MULT),
            nn.LeakyReLU(negative_slope= 0.05, inplace = True),
            # size (GEN_OUT_MULT) x 8 x 8
            nn.ConvTranspose2d(GEN_OUT_MULT, IMAGE_CHANNELS, kernel_size=4, stride=1, padding=0, bias=False),                     
            nn.Tanh()
            # size (IMAGE_CHANNELS) x 32 x 32    
        )

    def forward(self, x): 
        return self.main(x)





class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (IMAGE_CHANNELS) x 32 x 32 
            nn.Conv2d(IMAGE_CHANNELS, DISCRIM_OUT_MULT, kernel_size=3, stride=8, padding=1, bias=False),
            nn.BatchNorm2d(DISCRIM_OUT_MULT),
            nn.LeakyReLU(negative_slope= 0.05, inplace = True),
            # size (GEN_OUT_MULT) x 4 x 4
            nn.Conv2d(DISCRIM_OUT_MULT, 1, kernel_size=3, stride=4, padding=1, bias=False),
            nn.Sigmoid())



    def forward(self, x):
        return self.main(x)

