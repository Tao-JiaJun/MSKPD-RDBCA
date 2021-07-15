import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import os
import numpy as np
import math
import sys
sys.path.append("./")
from model import get_backbone, get_fpn
from model.head import Head
from model.loss import Loss
from model.ground_truth_maker import GTMaker
class Detector(nn.Module):
    def __init__(self, 
                 device,
                 input_size,
                 num_cls,
                 strides,
                 scales,
                 cfg):
        super(Detector, self).__init__()
        self.backbone = get_backbone(name=cfg['BACKBONE']['NAME'],
                                     pretrained=cfg['BACKBONE']['PRETRAINED'])
        self.neck = get_fpn(param=cfg['NECK'])
        self.head = Head(num_cls=num_cls,
                         in_size=cfg['HEAD']['IN_CHANNEL'],
                         out_size=cfg['HEAD']['OUT_CHANNEL'],
                         use_p7=cfg['HEAD']['USE_P7'])
        self.input_size = input_size
        self.device = device
        self.conf_thresh = 0.05
        self.nms_thresh = 0.5
        self.scale = np.array([[self.input_size[1], self.input_size[0], self.input_size[1], self.input_size[0]]])
        self.topk = 100
        self.num_cls = num_cls
        self.strides = strides
        self.scales = scales
        self.use_nms = True
        self.location_weight = torch.tensor([[-1, -1, 1, 1]]).float().to(self.device)
        self.pixel_location = self.set_init().to(self.device)
        self.gt_maker = GTMaker(input_size, num_cls, strides, scales)
        self.loss = Loss()
    def forward(self, x, gt_list=None):
        features = self.backbone(x)
        features = self.neck(features)
        features = self.head(features)
        if self.training:
            gt_tensor = torch.from_numpy(self.gt_maker(gt_list=gt_list)).float().to(self.device)
            cls_loss, reg_loss, nks_loss = self.loss(features, gt_tensor, num_cls=self.num_cls)
            return cls_loss, reg_loss, nks_loss
        else:
            return self.evaluating(features)
    
    def evaluating(self, pred_head):
        with torch.no_grad():
            # batch size
            bbox_list = []
            score_list = []
            cls_list = []
            B = pred_head.shape[0]
            cls_pred = torch.sigmoid(pred_head[:, :, : self.num_cls])
            # [xmin, ymin, xmax, ymax] = [x,y,x,y] + [l,t,r,b] + [-1,-1,1,1]
            loc_pred = torch.exp(pred_head[:, :, self.num_cls:-1]) * self.location_weight + self.pixel_location
            nks_pred = torch.sigmoid(pred_head[:, :, -1]).unsqueeze(-1)
            cls_pred = torch.pow(cls_pred, (2-nks_pred))
            for i in range(B):
                p_cls = cls_pred[i, ...].unsqueeze(0)
                p_loc = loc_pred[i, ...]
                # select the top 100 scoring predictions
                topk_scores, topk_inds, topk_clses = self._topk(p_cls)
                # a confidence threshold of 0.05 to filter out predictions with low confidence
                topk_scores = topk_scores[0].cpu().numpy()
                keep = np.where(topk_scores >= self.conf_thresh)
      
                topk_scores = topk_scores[keep]
                topk_cls = topk_clses[0][keep].cpu().numpy()
                topk_bbox_pred = p_loc[topk_inds[0][keep]].cpu().numpy()
                # nms
                if self.use_nms:
                    
                    keep = np.zeros(len(topk_bbox_pred), dtype=np.int)
                    for i in range(self.num_cls):
                        inds = np.where(topk_cls == i)[0]
                        if len(inds) == 0:
                            continue
                        c_bboxes = topk_bbox_pred[inds]
                        c_scores = topk_scores[inds]
                        c_keep = self.nms(c_bboxes, c_scores)
                        keep[inds[c_keep]] = 1

                    keep = np.where(keep > 0)
                    topk_bbox_pred = topk_bbox_pred[keep]
                    topk_scores = topk_scores[keep]
                    topk_cls = topk_cls[keep]

                # adjust the value of the box so that it does not appear beyond the picture
                bboxes = self.clip_boxes(topk_bbox_pred, self.input_size) / self.scale
                bbox_list.append(bboxes)
                score_list.append(topk_scores)
                cls_list.append(topk_cls)
            return bbox_list, score_list, cls_list

    def _gather_feat(self, feat, ind, mask=None):
        dim  = feat.size(2)
        ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _topk(self, scores):
   
        B, HW, C  = scores.size()
        scores = scores.permute(0, 2, 1)
        topk_scores, topk_inds = torch.topk(scores, self.topk)
        
        topk_inds = topk_inds % (HW)
        
        topk_score, topk_ind = torch.topk(topk_scores.view(B, -1), self.topk)
        topk_clses = torch.true_divide(topk_ind, self.topk).int()
        topk_inds = self._gather_feat(topk_inds.view(B, -1, 1), topk_ind).view(B, self.topk)

        return topk_score, topk_inds, topk_clses
    def set_init(self):
        """Generate the pixel value of each feature layer corresponding to each detection point"""
        # Calculate the number of H*W
        total = sum([(self.input_size[0] // s) * (self.input_size[1] // s)
                     for s in self.strides])
        # set to 0
        pixel_location = torch.zeros(total, 4).to(self.device)
        start_index = 0
        for index in range(len(self.strides)):
            s = self.strides[index]
            # make a feature map size corresponding the scale
            ws = self.input_size[1] // s
            hs = self.input_size[0] // s
            for ys in range(hs):
                for xs in range(ws):
                    x_y = ys * ws + xs
                    index = x_y + start_index
                    x = xs * s + s / 2
                    y = ys * s + s / 2
                    pixel_location[index, :] = torch.tensor([[x, y, x, y]]).float()
            start_index += ws * hs
        return pixel_location

    def clip_boxes(self, boxes, im_shape):
        """
        Clip boxes to image boundaries.
        """
        if boxes.shape[0] == 0:
            return boxes

        # x1 >= 0
        boxes[:, 0::4] = np.maximum(np.minimum(
            boxes[:, 0::4], im_shape[1] - 1), 0)
        # y1 >= 0
        boxes[:, 1::4] = np.maximum(np.minimum(
            boxes[:, 1::4], im_shape[0] - 1), 0)
        # x2 < im_shape[1]
        boxes[:, 2::4] = np.maximum(np.minimum(
            boxes[:, 2::4], im_shape[1] - 1), 0)
        # y2 < im_shape[0]
        boxes[:, 3::4] = np.maximum(np.minimum(
            boxes[:, 3::4], im_shape[0] - 1), 0)
        return boxes

    def nms(self, dets, scores):
        """"Pure Python NMS baseline."""
        x1 = dets[:, 0]  # xmin
        y1 = dets[:, 1]  # ymin
        x2 = dets[:, 2]  # xmax
        y2 = dets[:, 3]  # ymax

        areas = (x2 - x1) * (y2 - y1)                 # the size of bbox
        # sort bounding boxes by decreasing order
        order = scores.argsort()[::-1]

        # store the final bounding boxes
        keep = []
        while order.size > 0:
            i = order[0]  # the index of the bbox with highest confidence
            keep.append(i)  # save it to keep
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            inter = w * h

            # Cross Area / (bbox + particular area - Cross Area)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            # reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep
