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
            
            nn.ConvTranspose2d(Z_DIM, GEN_OUT_MULT, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(GEN_OUT_MULT),
            nn.LeakyReLU(negative_slope= 0.05, inplace = True),

            nn.ConvTranspose2d(GEN_OUT_MULT, GEN_OUT_MULT * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(GEN_OUT_MULT * 2),
            nn.LeakyReLU(negative_slope= 0.05, inplace = True),

            nn.ConvTranspose2d(GEN_OUT_MULT * 2, GEN_OUT_MULT * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(GEN_OUT_MULT * 4),
            nn.LeakyReLU(negative_slope= 0.05, inplace = True),
            
            nn.ConvTranspose2d(GEN_OUT_MULT * 4, IMAGE_CHANNELS, kernel_size=4, stride=2, padding=1, bias=False),                     
            nn.Tanh() 
        )

    def forward(self, x): 
        return self.main(x)





class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            
            nn.Conv2d(IMAGE_CHANNELS, DISCRIM_OUT_MULT, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(DISCRIM_OUT_MULT),
            nn.LeakyReLU(negative_slope= 0.05, inplace = True),

            nn.Conv2d(DISCRIM_OUT_MULT, DISCRIM_OUT_MULT, kernel_size=3, stride=4, padding=1, bias=False),
            nn.BatchNorm2d(DISCRIM_OUT_MULT),
            nn.LeakyReLU(negative_slope= 0.05, inplace = True),

            nn.Conv2d(DISCRIM_OUT_MULT, DISCRIM_OUT_MULT, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(DISCRIM_OUT_MULT),
            nn.LeakyReLU(negative_slope= 0.05, inplace = True),
            
            nn.Conv2d(DISCRIM_OUT_MULT, DISCRIM_OUT_MULT, kernel_size=3, stride=4, padding=1, bias=False),
            nn.BatchNorm2d(DISCRIM_OUT_MULT),
            nn.LeakyReLU(negative_slope= 0.05, inplace = True),
            
            #nn.Conv2d(DISCRIM_OUT_MULT, 1, kernel_size=3, stride=2, padding=1, bias=False),
            #nn.Sigmoid()
        )



    def forward(self, x):
        return self.main(x)

