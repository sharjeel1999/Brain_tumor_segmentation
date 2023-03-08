import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Linear_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.block = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            )
        
    def forward(self, x):
        x = self.block(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels, classes):
        super().__init__()
        
        self.classes = classes
        
        self.conv1 = DoubleConv(in_channels, 64)
        
    def set_layers(self, x):
        b, f = x.shape
        self.linear1 = Linear_block(f, 1024)
        self.linear2 = Linear_block(1024, 512)
        self.linear3 = Linear_block(512, 256)
        self.linear4 = Linear_block(256, 128)
        self.linear5 = Linear_block(128, 64)
        self.linear6 = Linear_block(64, 32)
        self.linear7 = Linear_block(32, self.classes)
    
    def forward(self, x):
        x = self.conv1(x)
        self.set_layers(x)
        
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.linear5(x)
        x = self.linear6(x)
        x = self.linear7(x)
        
        return x