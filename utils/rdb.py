import torch
import torch.nn as nn
import sys
sys.path.append("./")
from utils.modules import ConvBnActivation, SeparableConvBnActivation, SeparableConv2d
from utils import get_parameter_number
from utils.coordatt import CoordAtt
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size,stride,padding):
        super(ResidualBlock, self).__init__()
        self.conv = SeparableConvBnActivation(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding,activation='none')
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out + x)
        return out

class RDB(nn.Module):
    def __init__(self, planes):
        super(RDB, self).__init__() 
        self.conv1 = ResidualBlock(planes, planes, 3, 1, 1)
        self.conv2 = ResidualBlock(planes, planes, 3, 1, 1)
        self.conv3 = ResidualBlock(planes, planes, 3, 1, 1)
        self.conv1x1 = ConvBnActivation(planes*4, planes, kernel_size=1, activation="none")
        self.ca = CoordAtt(planes,planes)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out = torch.cat([x, out1, out2, out3],dim=1)
        out = self.conv1x1(out)
        out = self.relu(out + x)
        out = self.ca(out)
        out = self.relu(out + x)
        return out
if __name__=='__main__':
    device = torch.device("cpu")
    C3 = torch.randn(1, 128, 40, 40).to(device)
    neck = RDB(128)
    neck.to(device)
    fpn = neck(C3)
    for i in fpn:
        print(i.size())
    print(get_parameter_number(neck))