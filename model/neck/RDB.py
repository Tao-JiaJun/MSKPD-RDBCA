import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("./")
from utils.modules import ConvBnActivation, SeparableConvBnActivation

from utils import get_parameter_number
from utils.rdb import RDB

class RDBFPN(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size, use_p7=False):
        super().__init__()
        self.use_p7 = use_p7
        if use_p7:
            self.rdb_7 = RDB(feature_size)
        self.maxpool = nn.MaxPool2d(3,2,1)
        self.rdb_6 = RDB(feature_size)
        self.conv1x1_5 = ConvBnActivation(C5_size, feature_size, 1, activation='relu')
        self.rdb_5 = RDB(feature_size)
        self.up_5 = ConvBnActivation(feature_size, feature_size // 2, 1)
     
        self.conv1x1_4 = ConvBnActivation(C4_size, feature_size // 2, 1, activation='relu')
        self.rdb_4 = RDB(feature_size)
        self.up_4 = ConvBnActivation(feature_size, feature_size // 2, 1)

        self.conv1x1_3 = ConvBnActivation(C3_size, feature_size // 2, 1, activation='relu')
        self.rdb_3 = RDB(feature_size)
       
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, features):

        C3, C4, C5 = features[:]
        
        D5 = self.conv1x1_5(C5)

        D6 = self.maxpool(D5)
        D6 = self.rdb_6(D6)
        if self.use_p7:
            D7 = self.rdb_7(self.maxpool(D6))
        D5 = self.rdb_5(D5) 
        D5_up = F.interpolate(self.up_5(D5), scale_factor=2.0, mode='bilinear', align_corners=True) 

        D4 = self.conv1x1_4(C4)
        D4 = torch.cat([D4, D5_up], dim=1) 
        D4 = self.rdb_4(D4)
        D4_up = F.interpolate(self.up_4(D4), scale_factor=2.0, mode='bilinear', align_corners=True) 
        
        D3 = self.conv1x1_3(C3)
        D3 = torch.cat([D3, D4_up], dim=1) 
        D3 = self.rdb_3(D3) 
        if self.use_p7:
            return D3, D4, D5, D6, D7
        else: 
            return D3, D4, D5, D6, 


if __name__=='__main__':
    device = torch.device("cpu")
    C3 = torch.randn(1, 128, 40, 40).to(device)
    C4 = torch.randn(1, 256, 20, 20).to(device)
    C5 = torch.randn(1, 512, 10, 10).to(device)
    neck = RDBFPN(128,256,512,128)
    neck.to(device)
    fpn = neck([C3, C4, C5])
    for i in fpn:
        print(i.size())
    print(get_parameter_number(neck))