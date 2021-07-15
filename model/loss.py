import torch
from utils.focal_loss import BCEFocalLoss,HeatmapLoss
from utils.iou import IoU
class Loss(object):
    def __init__(self, iou_type='iou'):
        self.cls_loss_func = HeatmapLoss()
        self.nks_loss_func = torch.nn.BCELoss(reduction='none')
        self.compute_iou = IoU(iou_type=iou_type)
    def __call__(self, pred, gt, num_cls):
        # preds
        pred_cls = torch.sigmoid(pred[:, :, : num_cls])
        pred_reg = torch.exp(    pred[:, :, num_cls:-1])
        pred_iou = torch.sigmoid(pred[:, :, -1])
        # gts
        gt_cls = gt[:, :, : num_cls].float()
        gt_reg = gt[:, :, num_cls:-1].float()
        gt_pos = gt[:, :, -1].float()
        N_pos = self.get_num_positvate(gt_pos)
    

        gt_iou_mask =  torch.ones_like(gt_pos)
        N_all = self.get_num_positvate(gt_iou_mask)
        
        batch_size = pred_cls.size(0)
        # loss of cls
        pred_cls_new = torch.pow(pred_cls, (2-pred_iou).unsqueeze(2))
        # gt_cls_new = torch.pow(gt_cls, (2-gt_pos).unsqueeze(2))
        cls_loss = torch.mean(torch.sum(torch.sum(self.cls_loss_func(pred_cls_new, gt_cls), dim=-1), dim=-1)  / N_pos)
        # loss of reg
        iou = self.compute_iou(pred_reg, gt_reg)
        reg_loss = torch.sum(torch.sum((1 - iou)  * gt_pos, dim=-1) / N_pos) / batch_size
        # loss of nks
        nks_loss = torch.sum(torch.sum(self.nks_loss_func(pred_iou, gt_pos), dim=-1) / N_all) / batch_size
        return cls_loss, reg_loss, nks_loss
    
    def get_num_positvate(self, gt_pos):
        N_pos = torch.sum(gt_pos, dim=-1)
        N_pos = torch.max(N_pos, torch.ones(N_pos.size(), device=N_pos.device))
        return N_pos
