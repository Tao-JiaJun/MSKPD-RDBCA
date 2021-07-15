from model.backbone.resnet import resnet18, resnet50
from model.neck.FPN import FPN
from model.neck.RDB import RDBFPN
from model.backbone.mobilenetv2 import mobilenet_v2
def get_backbone(name, pretrained):
    if name == 'RESNET18':
        if pretrained:
            return resnet18(pretrained=True)
        else:
            return resnet18()
    elif name == 'RESNET50':
        if pretrained:
            return resnet50(pretrained=True)
        else:
            return resnet50()
    elif name == "MBNV2":
        return mobilenet_v2(pretrained=True)
    else:
        raise Exception('Backbone name error!')
def get_fpn(param):
    name = param["NAME"]
    if name == 'FPN':
        return FPN(C3_size=param["C3_CHANNEL"], 
                   C4_size=param["C4_CHANNEL"], 
                   C5_size=param["C5_CHANNEL"], 
                   feature_size=param["OUT_CHANNEL"])
    elif name == 'RDB':
        return RDBFPN(C3_size=param["C3_CHANNEL"], 
                      C4_size=param["C4_CHANNEL"], 
                      C5_size=param["C5_CHANNEL"], 
                      feature_size=param["OUT_CHANNEL"], 
                      use_p7=param["USE_P7"])
    else:
        raise Exception('Neck name error!')