import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv_3d(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)



class Up_3d(nn.Module):
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv_3d(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv_3d(in_channels // 2, out_channels)

    def forward(self, x1):
        x1 = self.up(x1)
        
        return self.conv(x1)


class OutConv_2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv_2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        return self.conv(x)
    

class Decoder_3D(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Decoder_3D, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.dout = nn.Dropout2d(0.2)
        
        self.sync_channels = nn.Sequential(
                nn.Conv3d(n_channels, 1024, kernel_size = 1),
                nn.BatchNorm3d(1024),
                nn.ReLU()
            )
        
        self.up1 = (Up_3d(1024, 512, bilinear))
        self.up2 = (Up_3d(512, 256, bilinear))
        self.up3 = (Up_3d(256, 128, bilinear))
        self.up4 = (Up_3d(128, 64, bilinear))
        self.outc = (OutConv_2d(64*96, n_classes))

    def forward(self, x):
        
        x = self.sync_channels(x)
        
        x = self.dout(self.up1(x))
        x = self.up2(x)
        x = self.dout(self.up3(x))
        x = self.up4(x)
        
        B, C, Z, X, Y = x.shape
        x = torch.reshape(x, (B, C*Z, X, Y))
        
        x = self.outc(x)
        return x