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
            nn.GroupNorm(8, mid_channels), #nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            
            
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels), #nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DoubleConv_2d_new(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True)
        )
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
        )
        
        self.end_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.double_conv(x1)
        x = x1 + x2
        x = self.end_conv(x)
        
        return x

class Correlation_unit(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.conv_f = nn.Conv2d(in_channels, in_channels, kernel_size = 3, padding = 1)
        self.conv_w = nn.Conv2d(in_channels, in_channels, kernel_size = 3, padding = 1)
    
    def correlation_operation(self, xi, xj):
        f_diff = xi - xj
        #print('shape: ', f_diff.shape)
        f_diff = torch.abs(torch.tanh(self.conv_w(f_diff)))
        
        xj = self.conv_f(xj)
        
        mult = f_diff * xj
        return mult
    
    def forward(self, feats_2d, feats_3d):
        #print('2d feats shape: ', feats_2d.shape)
        #print('3d feats shape: ', feats_3d.shape)
        slices = feats_3d.shape[2]
        
        comb_feats = torch.zeros_like((feats_3d))
        for s in range(slices):
            slice_3d = feats_3d[:, :, s, :, :]
            ms = self.correlation_operation(feats_2d, slice_3d)
            comb_feats[:, :, s, :, :] = ms
        
        return comb_feats

class Correlation_unit_2(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.conv_f = nn.Conv2d(in_channels, in_channels, kernel_size = 3, padding = 1)
        self.conv_w = nn.Conv2d(in_channels, in_channels, kernel_size = 3, padding = 1)
    
    def correlation_operation(self, xi, xj):
        f_diff = xi - xj
        #print('shape: ', f_diff.shape)
        f_diff = torch.abs(torch.tanh(self.conv_w(f_diff)))
        
        xj = self.conv_f(xj)
        
        mult = f_diff * xj
        return mult
    
    def forward(self, feats_2d, feats_3d):
        #print('2d feats shape: ', feats_2d.shape)
        #print('3d feats shape: ', feats_3d.shape)
        slices = feats_3d.shape[2]
        
        # comb_feats = torch.zeros_like((feats_3d))

        for s in range(slices):
            slice_3d = feats_3d[:, :, s, :, :]
            ot = self.correlation_operation(feats_2d, slice_3d)
            ms = ot + feats_2d
            feats_2d = torch.clone(ms)
            # comb_feats[:, :, s, :, :] = ms
        
        return ms

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
        self.dout = nn.Dropout2d(0.2)
        
        self.corr = Correlation_unit_2(n_channels)
        self.corr2 = Correlation_unit_2(256)
        self.corr3 = Correlation_unit_2(64)
        self.corr4 = Correlation_unit_2(32)
        
        
        self.up1 = Up_2d(512, 256, bilinear)
        self.up2 = Up_2d(256, 128, bilinear)
        self.up3 = Up_2d(64, 32, bilinear)
        self.up4 = Up_2d(32, 16, bilinear)
        # self.outc = DoubleConv_2d(32, 8)#OutConv_2d(32, 8)
        
        self.sync_channels = nn.Sequential(
                nn.Conv2d(512*7, 512, kernel_size = 1),
                nn.BatchNorm2d(512),
                nn.ReLU()
            )
        
        self.sync_channels_21 = nn.Sequential(
                nn.Conv2d(256*7, 256, kernel_size = 1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
        
        self.sync_channels_31 = nn.Sequential(
                nn.Conv2d(128*7, 128, kernel_size = 1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            )
        
        self.sync_channels_41 = nn.Sequential(
                nn.Conv2d(64*7, 64, kernel_size = 1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
        
        self.outc = nn.Sequential(
            nn.Conv2d(16, 4, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, 4), #nn.BatchNorm2d(4),
            nn.ReLU(),
        )
        
        self.final_layer = nn.Sequential(
                nn.Conv2d(8, 4, kernel_size = 1),
            )
        
        
    def set_feats(self, out_3d):
        B, C, Z, X, Y = out_3d.shape
        out_3d = torch.reshape(out_3d, (B, C*Z, X, Y))
        return out_3d
    
    def forward(self, out_2d_ar, out_3d_ar):
        
        # out_2d = out_2d_ar[0]
        # out_3d = out_3d_ar[0]
        # feats = self.corr(out_2d, out_3d)
        # x = self.dout(self.up1(feats))
        
        # out_2d = out_2d_ar[0]
        # out_3d = out_3d_ar[0]
        # feats = self.corr2(out_2d, out_3d)
        # # feats = self.set_feats(feats)
        # '''            can add features from here      '''
        # # combined_features = self.sync_channels_21(feats)
        # # print('feats/x shape: ', feats.shape, x.shape)
        # x = feats# + x
        # x = self.up2(x)
        
        out_2d = out_2d_ar[0]
        out_3d = out_3d_ar[0]
        feats1 = self.corr3(out_2d, out_3d)
        x = feats1# + x
        
        x = self.dout(self.up3(x))
        
        # print('first upsamp: ', x.shape)
        out_2d = out_2d_ar[1]
        out_3d = out_3d_ar[1]
        feats = self.corr4(out_2d, out_3d)

        x = feats + x
        x = self.dout(self.up4(x))
        
        x = self.outc(x)
        # print('feats 1 unique: ', torch.unique(x))
        # print('output shape: ', x.shape)
        # p = self.final_layer(x)
        return x