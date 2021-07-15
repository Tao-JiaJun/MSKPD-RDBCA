import torch
import torch.nn as nn
import sys

class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x

class SeparableConvBnActivation(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, activation="none", bn=True):
        super().__init__()
        self.conv = SeparableConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = activation
        if activation == "mish":
            self.activation = Mish()
        elif activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "leaky":
            self.activation = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act != "none":
            x = self.activation(x)
        return x
class ConvBnActivation(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, activation="none"):
        super(ConvBnActivation, self).__init__()
        self.act = activation
        if activation == "mish":
            self.activation = Mish()
        elif activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "leaky":
            self.activation = nn.LeakyReLU(0.1, inplace=True)
       
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act != "none":
            x = self.activation(x)
        return x
    
class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()
        self.depthwise = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self,x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
        
