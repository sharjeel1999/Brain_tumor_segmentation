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

class Encoder_2d(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(Encoder_2d, self).__init__()
        
        self.dout = nn.Dropout2d(0.15)
        
        self.C_initial_normal = Initial_conv(in_channels, inter_channels) # these ones are random weights
        self.C_d1c2 = Const_Residual_block(inter_channels, inter_channels) # these ones are random weights
        
        self.down_res_d2 = Down_Residual_block(inter_channels, int(inter_channels*2), kernel_size = 3, stride = 2)
        self.d2c2 = Const_Residual_block(int(inter_channels*2), int(inter_channels*2))
        
        self.down_res_d3 = Down_Residual_block(int(inter_channels*2), int(inter_channels*4), kernel_size = 3, stride = 2)
        self.d3c2 = Const_Residual_block(int(inter_channels*4), int(inter_channels*4))
        self.d3c3 = Const_Residual_block(int(inter_channels*4), int(inter_channels*4))
        # self.d3c4 = Const_Residual_block(int(inter_channels*4), int(inter_channels*4))
        
        # self.down_res_d4 = Down_Residual_block(int(inter_channels*4), int(inter_channels*8), kernel_size = 3, stride = 2)
        # self.d4c2 = Const_Residual_block(int(inter_channels*8), int(inter_channels*8))
        # self.d4c3 = Const_Residual_block(int(inter_channels*8), int(inter_channels*8))
        # # self.d4c4 = Const_Residual_block(int(inter_channels*8), int(inter_channels*8))
        # # self.d4c5 = Const_Residual_block(int(inter_channels*8), int(inter_channels*8))
        
        # self.down_res_d5 = Down_Residual_block(int(inter_channels*8), int(inter_channels*16), kernel_size = 3, stride = 2)
        # self.d5c2 = Const_Residual_block(int(inter_channels*16), int(inter_channels*16))
        # self.d5c3 = Const_Residual_block(int(inter_channels*16), int(inter_channels*16))
        # # self.d5c4 = Const_Residual_block(int(inter_channels*16), int(inter_channels*16))
        # # self.d5c5 = Const_Residual_block(int(inter_channels*16), int(inter_channels*16))
        
        # self.down_bottleneck = Down_Residual_block(int(inter_channels*16), int(inter_channels*32), kernel_size = 3, stride = 2)
        # self.bottleneck_c2 = Const_Residual_block(int(inter_channels*32), int(inter_channels*32))
        # self.bottleneck_c3 = Const_Residual_block(int(inter_channels*32), int(inter_channels*32))
        # # self.bottleneck_c4 = Const_Residual_block(int(inter_channels*32), int(inter_channels*32))
        # # self.bottleneck_c5 = Const_Residual_block(int(inter_channels*32), int(inter_channels*32))
        
    def forward(self, x):
        x = self.C_initial_normal(x)
        x1 = self.C_d1c2(x)
        
        x2 = self.down_res_d2(x1)
        x2 = self.d2c2(x2)
        
        x3 = self.down_res_d3(x2)
        x3 = self.d3c2(x3)
        x3 = self.d3c3(x3)
        # x3 = self.d3c4(x3)
        x3 = self.dout(x3)
        
        # x4 = self.down_res_d4(x3)
        # x4 = self.d4c2(x4)
        # x4 = self.d4c3(x4)
        # # x4 = self.d4c4(x4)
        # # x4 = self.d4c5(x4)
        # x4 = self.dout(x4)
        
        # x5 = self.down_res_d5(x4)
        # x5 = self.d5c2(x5)
        # x5 = self.d5c3(x5)
        # # x5 = self.d5c4(x5)
        # # x5 = self.d5c5(x5)
        # x5 = self.dout(x5)
        
        # x6 = self.down_bottleneck(x5)
        # x6 = self.bottleneck_c2(x6)
        # x6 = self.bottleneck_c3(x6)
        # # x6 = self.bottleneck_c4(x6)
        # # x6 = self.bottleneck_c5(x6)
        # x6 = self.dout(x6)
        
        return [x3, x2, x1] # [x6, x5, x4, x3, x2, x1]