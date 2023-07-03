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
    

class Dense_Decoder_2D(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Dense_Decoder_2D, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.dout = nn.Dropout2d(0.05)
        
        self.sync_channels = nn.Sequential(
                nn.Conv2d(n_channels, 1024, kernel_size = 1),
                nn.BatchNorm2d(1024),
                nn.ReLU()
            )
        
        self.sync_channels_21 = nn.Sequential(
                nn.Conv2d(1024*6, 1024, kernel_size = 1),
                nn.BatchNorm2d(1024),
                nn.ReLU()
            )
        
        self.sync_channels_31 = nn.Sequential(
                nn.Conv2d(512*6, 512, kernel_size = 1),
                nn.BatchNorm2d(512),
                nn.ReLU()
            )
        
        self.sync_channels_41 = nn.Sequential(
                nn.Conv2d(256*6, 256, kernel_size = 1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
        
        self.const_conv = Up_2d(1024, 1024, bilinear) #DoubleConv_2d(1024, 1024)
        self.up1 = Up_2d(1024, 512, bilinear)

        self.up2 = Up_2d(512, 256, bilinear)
        self.up3 = Up_2d(256, 64, bilinear)
        #self.up4 = Up_2d(128, 64, bilinear)
        self.outc = OutConv_2d(64, n_classes)
        
    def set_feats(self, out_3d):
        B, C, Z, X, Y = out_3d.shape
        out_3d = torch.reshape(out_3d, (B, C*Z, X, Y))
        return out_3d
    
    def forward(self, out_2d_ar, out_3d_ar):
        
        out_2d = out_2d_ar[0]
        out_3d = out_3d_ar[0]
        combined_features = torch.cat((out_2d, out_3d), dim = 1)
        #print('first comb: ', combined_features.shape)
        x = self.sync_channels(combined_features)
        x = self.const_conv(x)
        
        out_2d = out_2d_ar[1]
        out_3d = out_3d_ar[1]
        out_3d = self.set_feats(out_3d)
        combined_features = torch.cat((out_2d, out_3d), dim = 1)
        #print('first comb: ', combined_features.shape)
        combined_features = self.sync_channels_21(combined_features)
        #print('comb/x: ', combined_features.shape, x.shape)
        x = x + combined_features
        x = self.up1(x)
        
        ################################### 
        
        out_2d = out_2d_ar[2]
        out_3d = out_3d_ar[2]
        out_3d = self.set_feats(out_3d)
        #print('out shapes: ', out_2d.shape, out_3d.shape)
        combined_features = torch.cat((out_2d, out_3d), dim = 1)
        #print('first comb: ', combined_features.shape)
        combined_features = self.sync_channels_31(combined_features)
        #print('comb/x: ', combined_features.shape, x.shape)
        x = x + combined_features
        x = self.up2(x)
        
        out_2d = out_2d_ar[3]
        out_3d = out_3d_ar[3]
        out_3d = self.set_feats(out_3d)
        #print('out shapes: ', out_2d.shape, out_3d.shape)
        combined_features = torch.cat((out_2d, out_3d), dim = 1)
        #print('first comb: ', combined_features.shape)
        combined_features = self.sync_channels_41(combined_features)
        #print('comb/x: ', combined_features.shape, x.shape)
        x = x + combined_features
        x = self.dout(self.up3(x))
        
        # out_2d = out_2d_ar[3]
        # out_3d = out_3d_ar[3]
        # out_3d = self.set_feats(out_3d)
        # #print('out shapes: ', out_2d.shape, out_3d.shape)
        # combined_features = torch.cat((out_2d, out_3d), dim = 1)
        # combined_features = self.sync_channels_41(combined_features)
        # x = x + combined_features
        # x = self.up4(x)
        
        x = self.outc(x)
        return x