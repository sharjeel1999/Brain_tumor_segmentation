import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv_2d(nn.Module):

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



class Up_2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv_2d(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv_2d(in_channels // 2, out_channels)

    def forward(self, x1):
        x1 = self.up(x1)
        
        return self.conv(x1)


class OutConv_2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv_2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        return self.conv(x)
    

class Decoder_2D(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Decoder_2D, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        
        self.sync_channels = nn.Sequential(
                nn.Conv2d(n_channels, 1024, kernel_size = 1),
                nn.BatchNorm2d(1024),
                nn.ReLU()
            )
        
        self.up1 = (Up_2d(1024, 512, bilinear))
        self.up2 = (Up_2d(512, 256, bilinear))
        self.up3 = (Up_2d(256, 128, bilinear))
        self.up4 = (Up_2d(128, 64, bilinear))
        self.outc = (OutConv_2d(64, n_classes))

    def forward(self, x):
        
        x = self.sync_channels(x)
        
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.outc(x)
        return x