import torch.nn as nn
import torch
from torch import nn as nn
from torch.nn import functional as F

class Initial_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Initial_conv, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
        
    def forward(self, x):
        x = self.conv(x)
        return x

class Normal_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Normal_conv, self).__init__()
        
        num_groups = 8
        num_channels = in_channels

        if num_channels < num_groups:
            num_groups = 1
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
        # self.bn = nn.BatchNorm3d(in_channels)
        self.bn = nn.GroupNorm(num_groups, num_channels)
        self.act = nn.LeakyReLU()
        
    def forward(self, x):
        x = self.bn(x)
        x = self.act(x)
        x = self.conv(x)
        return x

class Const_Residual_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Const_Residual_block, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        
        self.bn1 = nn.GroupNorm(8, in_channels) #nn.BatchNorm3d(in_channels)
        self.bn2 = nn.GroupNorm(8, out_channels) #nn.BatchNorm3d(out_channels)
        self.act = nn.LeakyReLU()
        
    def forward(self, x_in):
        x = self.bn1(x_in)
        x = self.act(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.conv2(x)
        # print('shapes: ', x.shape, x_in.shape)
        x = x + x_in
        return x
    
class Down_Residual_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1):
        super(Down_Residual_block, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
        
        self.bn1 = nn.GroupNorm(8, in_channels) #nn.BatchNorm3d(in_channels)
        self.bn2 = nn.GroupNorm(8, out_channels) #nn.BatchNorm3d(out_channels)
        self.act = nn.LeakyReLU()
        
    def forward(self, x_in):
        x = self.bn1(x_in)
        x = self.act(x)
        x2 = self.conv1(x)
        x = self.bn2(x2)
        x = self.act(x)
        x = self.conv2(x)
        x = x + x2
        return x

class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='bilinear', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)

class Reduce_channs(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Reduce_channs, self).__init__()
        
        self.bn = nn.GroupNorm(8, in_channels) #nn.BatchNorm3d(in_channels)
        self.act = nn.LeakyReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1)
        
    def forward(self, x):
        x = self.bn(x)
        x = self.act(x)
        x = self.conv(x)
        return x

class Transposed_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transposed_conv, self).__init__()
        
        self.bn = nn.GroupNorm(8, in_channels)
        self.act = nn.LeakyReLU()      
        self.trans = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 3, stride = 2, padding = 1)
        
    def forward(self, x):
        x = self.bn(x)
        x = self.act(x)
        x = self.trans(x)
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


class OutConv_2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv_2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        return self.conv(x)
    

class Decoder_2D(nn.Module):
    def __init__(self, n_channels, inter_channels, n_classes, bilinear=False):
        super(Decoder_2D, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.dout = nn.Dropout2d(0.2)
        
        self.corr = Correlation_unit_2(n_channels)
        self.corr2 = Correlation_unit_2(256)
        self.corr3 = Correlation_unit_2(128)
        self.corr4 = Correlation_unit_2(64)
        
        # self.red_chan_bottleneck = Reduce_channs(int(inter_channels*32), int(inter_channels*16))
        # self.up_bottlenect = Upsample(scale_factor = 2) #Transposed_conv(int(inter_channels*32), int(inter_channels*16))
        # self.norm_bottleneck = Normal_conv(int(inter_channels*16), int(inter_channels*16))
        
        # self.red_chan_c5 = Reduce_channs(int(inter_channels*16), int(inter_channels*8))
        # self.up_c5 = Upsample(scale_factor = 2) #Transposed_conv(int(inter_channels * 16), int(inter_channels * 8))
        # self.norm_c5 = Normal_conv(int(inter_channels * 8), int(inter_channels * 8))
        
        # self.red_chan_c4 = Reduce_channs(int(inter_channels*8), int(inter_channels*4))
        # self.up_c4 = Upsample(scale_factor = 2) #Transposed_conv(int(inter_channels * 8), int(inter_channels * 4))
        # self.norm_c4 = Normal_conv(int(inter_channels * 4), int(inter_channels * 4))
        
        self.red_chan_c3 = Reduce_channs(int(inter_channels*4), int(inter_channels*2))
        self.up_c3 = Upsample(scale_factor = 2) #Transposed_conv(int(inter_channels * 4), int(inter_channels * 2))
        self.C_norm_c3 = Normal_conv(int(inter_channels * 2), int(inter_channels * 2)) # this ones are random weights
        
        self.red_chan_c2 = Reduce_channs(int(inter_channels*2), int(inter_channels))
        self.up_c2 = Upsample(scale_factor = 2) #Transposed_conv(int(inter_channels * 2), int(inter_channels))
        self.C_norm_c2 = Normal_conv(int(inter_channels), int(inter_channels)) # this ones are random weights
        
        self.outc = nn.Sequential(
            nn.Conv2d(32, 4, kernel_size=3, padding=1, bias=False),
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
        
        # print('3d / 2d shapes: ', out_2d.shape, out_3d.shape)
        feats1 = self.corr3(out_2d, out_3d)
        x = feats1# + x
        # print('correlated output: ', feats1.shape)
        x = self.red_chan_c3(x)
        x = self.up_c3(x)
        x = self.C_norm_c3(x)
        
        # print('first upsamp: ', x.shape)
        out_2d = out_2d_ar[1]
        out_3d = out_3d_ar[1]
        feats = self.corr4(out_2d, out_3d)

        x = feats + x
        x = self.red_chan_c2(x)
        x = self.up_c2(x)
        x = self.C_norm_c2(x)
        
        
        x = self.outc(x)
        # print('feats 1 unique: ', torch.unique(x))
        # print('output shape: ', x.shape)
        # p = self.final_layer(x)
        return x