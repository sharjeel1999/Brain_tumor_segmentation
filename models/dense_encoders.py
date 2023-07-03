import time
import torch
import torch.nn as nn

from .densenet_3d import generate_model

class Dense_encoder_2d(nn.Module):
    def __init__(self, in_channels):
        super(Dense_encoder_2d, self).__init__()
        model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained = False)
        
        self.conv0 = torch.nn.Conv2d(in_channels, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.norm0 = model.features.norm0
        self.relu0 = model.features.relu0
        self.pool0 = model.features.pool0
        
        self.db1 = model.features.denseblock1
        self.t1 = model.features.transition1
        self.db2 = model.features.denseblock2
        self.t2 = model.features.transition2
        self.db3 = model.features.denseblock3
        self.t3 = model.features.transition3
        self.db4 = model.features.denseblock4

    def forward(self, x):
        x1 = self.conv0(x)
        x1 = self.norm0(x1)
        x1 = self.relu0(x1)
        x1 = self.pool0(x1)
        
        x1 = self.db1(x1)
        
        x2 = self.t1(x1)
        x2 = self.db2(x2)
        
        x3 = self.t2(x2)
        x3 = self.db3(x3)
        
        x4 = self.t3(x3)
        x4 = self.db4(x4)
        
        return [x4, x3, x2, x1]


class Dense_encoder_3d(nn.Module):
    def __init__(self, in_channels):
        super(Dense_encoder_3d, self).__init__()
        model3d = generate_model(121)

        self.conv0 = torch.nn.Conv3d(in_channels, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.norm0 = model3d.features.norm1
        self.relu0 = model3d.features.relu1
        self.pool0 = model3d.features.pool1
        
        self.db1 = model3d.features.denseblock1
        self.t1 = model3d.features.transition1
        self.db2 = model3d.features.denseblock2
        self.t2 = model3d.features.transition2
        self.db3 = model3d.features.denseblock3
        self.t3 = model3d.features.transition3
        self.db4 = model3d.features.denseblock4
        
        self.sync_layer = nn.Sequential(
                nn.Conv2d(1024*5, 1024, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True)
            )
        
    def forward(self, x):
        x1 = self.conv0(x)
        x1 = self.norm0(x1)
        x1 = self.relu0(x1)
        x1 = self.pool0(x1)
        
        x1 = self.db1(x1)
        
        x2 = self.t1(x1)
        x2 = self.db2(x2)
        
        x3 = self.t2(x2)
        x3 = self.db3(x3)
        
        x4 = self.t3(x3)
        x4 = self.db4(x4)
        
        B, C, Z, X, Y = x4.shape
        x4 = torch.reshape(x4, (B, C*Z, X, Y))
        x4 = self.sync_layer(x4)
        
        return [x4, x3, x2, x1]
