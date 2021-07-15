import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("./")
from utils.modules import SeparableConvBnActivation, SeparableConv2d
from utils import get_parameter_number
import math
class Pred(nn.Module):
    def __init__(self,totol_pred_num, in_size, out_size):
        super(Pred, self).__init__()
        self.conv1 = SeparableConvBnActivation(in_size, out_size, kernel_size=3, padding=1, stride=1,activation='relu')
        self.conv2 = nn.Conv2d(out_size, totol_pred_num, 1)
        self.apply(self.init_conv_RandomNormal)
        nn.init.constant_(self.conv2.bias,-math.log((1 - 0.01) / 0.01))
    def init_conv_RandomNormal(self,module,std=0.01):
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, std=std)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
class Head(nn.Module):
    def __init__(self,num_cls, in_size, out_size, use_p7=False):
        super(Head, self).__init__()
        self.use_p7 = use_p7
        # total = cls + reg + nsk
        self.totol_pred_num = num_cls + 4 + 1
        self.pred_3 = Pred(self.totol_pred_num, in_size, out_size)
        self.pred_4 = Pred(self.totol_pred_num, in_size, out_size)
        self.pred_5 = Pred(self.totol_pred_num, in_size, out_size)
        self.pred_6 = Pred(self.totol_pred_num, in_size, out_size)
        if self.use_p7 :
            self.pred_7 = Pred(self.totol_pred_num, in_size, out_size)

    def forward(self, features):
        if self.use_p7:
            F_3, F_4, F_5, F_6, F_7 = features[:]
            # get batch
            B = F_3.shape[0]
            pred_3 = self.pred_3(F_3).view(B, self.totol_pred_num, -1)
            pred_4 = self.pred_4(F_4).view(B, self.totol_pred_num, -1)
            pred_5 = self.pred_5(F_5).view(B, self.totol_pred_num, -1)
            pred_6 = self.pred_6(F_6).view(B, self.totol_pred_num, -1)
            pred_7 = self.pred_7(F_7).view(B, self.totol_pred_num, -1)
            # [B, C, WxH] -> [B, WxH, C]
            preds = torch.cat([pred_3, pred_4, pred_5, pred_6,pred_7], dim=-1).permute(0, 2, 1)
        else:
            F_3, F_4, F_5, F_6 = features[:]
            # get batch
            B = F_3.shape[0]
            pred_3 = self.pred_3(F_3).view(B, self.totol_pred_num, -1)
            pred_4 = self.pred_4(F_4).view(B, self.totol_pred_num, -1)
            pred_5 = self.pred_5(F_5).view(B, self.totol_pred_num, -1)
            pred_6 = self.pred_6(F_6).view(B, self.totol_pred_num, -1)
            # [B, C, WxH] -> [B, WxH, C]
            preds = torch.cat([pred_3, pred_4, pred_5, pred_6], dim=-1).permute(0, 2, 1)
        return preds
 

if __name__ == "__main__":
    device = torch.device("cuda")
    head = Head(20,256,256).to(device)
    F3 = torch.randn(1, 256, 40, 40).to(device)
    F4 = torch.randn(1, 256, 20, 20).to(device)
    F5 = torch.randn(1, 256, 10, 10).to(device)
    F6 = torch.randn(1, 256, 5, 5).to(device)
    
    out = head([F3,F4,F5,F6])
    for i in out:
        print(i.size())
    print(get_parameter_number(head))
