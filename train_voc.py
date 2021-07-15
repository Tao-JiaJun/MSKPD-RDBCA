import argparse
import collections
import math
import os
import time

import numpy as np
import torch
import torch.cuda as cuda
import torch.optim as optim
import torch.utils.data as data

from configs import *
from datasets import BaseTransform, detection_collate
from datasets.augmentations import SSDAugmentation
from datasets.config import MEANS, voc_384, voc_512
from datasets.voc import VOCDataset
from model.detector import Detector
from utils import get_device, get_parameter_number


def parse_args():
    parser = argparse.ArgumentParser(description='Detector')
    # optimizer
    parser.add_argument('--lr', default=1e-3, type=float, 
                        help='initial learning rate')
    # training strategy
    parser.add_argument('--use_cos', action='store_true', default=True,
                        help='use cos lr')
    parser.add_argument('--use_warm_up', action='store_true', default=True,
                        help='use warm up')
    # dataloader
    parser.add_argument('--num_workers', default=4, type=int, 
                        help='Number of workers used in dataLoader')
    # other
   
    parser.add_argument('--save_epoch', type=int, default=10,
                         help='save model')
    # cuda
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use cuda.')
    parser.add_argument('--cuda_device', default='0', type=str, 
                        help='CUDA_VISIBLE_DEVICES')
    # batch size
    parser.add_argument('--batch_size', default=16, type=int, 
                        help='Batch size for training')
    # resume
    parser.add_argument('-r', '--resume', action='store_true', default=False, 
                        help='keep training')
    parser.add_argument('--save_weights', default='./weights/MBNV2_RDB_384/', type=str, 
                        help='floder where save model weights')
    parser.add_argument('--model_weight', default='', type=str, 
                        help='model weight name')
    # config files
    parser.add_argument('--config_file', default=MBNV2_RDB_384, type=dict, 
                        help='floder where save model weights')
    return parser.parse_args()

def train_model():
    args = parse_args()
    if args.cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
        device = get_device()

    # model
    voc = voc_384
    print('==> Loading the model...', flush=True)
    model = Detector(device,
                     input_size=voc["input_size"],
                     num_cls=voc["num_cls"],
                     strides=voc["strides"],
                     scales=voc["scales"],
                     cfg=args.config_file)
    model.to(device)
    print(get_parameter_number(model))

    base_lr = args.lr
    tmp_lr = base_lr
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    # each part of loss weight
    iteration = 0
    start_epoch = 0
    path_to_save = args.save_weights
    os.makedirs(path_to_save, exist_ok=True)
    if args.resume: 
        model_path = os.path.join(path_to_save, args.model_weight)
        assert os.path.isfile(model_path)
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('Learning rate is {}.'.format(tmp_lr))
        print('Load checkpoint at epoch {}.'.format(start_epoch))

    print('==> Loading datasets...', flush=True)
    batch_size = args.batch_size
    # train set
    train_dataset = VOCDataset(root=voc['root'],
                               image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                               transform=SSDAugmentation(voc["input_size"], MEANS))
    train_dataloader = data.DataLoader(train_dataset,
                                       batch_size=batch_size,
                                       num_workers=args.num_workers,
                                       shuffle=True,
                                       collate_fn=detection_collate,
                                       pin_memory=True,
                                       drop_last=True)
    epoch_size = len(train_dataset) // batch_size
    print('The train dataset size:', len(train_dataset))
    print("--------------------Object Detection--------------------")
    print("Start to train...!")
    print("--------------------------------------------------------", flush=True)
    num_hint = 100
    loss_hint = collections.deque(maxlen=num_hint)
    cls_hint = collections.deque(maxlen=num_hint)
    reg_hint = collections.deque(maxlen=num_hint)
    nks_hint = collections.deque(maxlen=num_hint)
    epochs = voc["max_epoch"]
    for epoch in range(start_epoch, epochs):
        model.train()
        set_cos_lr(epoch, epochs, optimizer, base_lr, None, cos_lr=True)
        tik = time.time()
        for iter_i, (_, images, gt_list) in enumerate(train_dataloader):
            
            if args.use_warm_up:
                set_warm_up(epoch, epoch_size, iter_i, optimizer, base_lr)
            iteration += 1
            # reset gradient
            optimizer.zero_grad()
            # inference and compute loss
            cls_loss, reg_loss, nks_loss = model(images.float().to(device), gt_list)
            total_loss = cls_loss + reg_loss + nks_loss
            # backward
            total_loss.backward()
            # update parameters of net
            optimizer.step()
            # recorder
            loss_hint.append(total_loss.item())
            cls_hint.append(cls_loss.item())
            reg_hint.append(reg_loss.item())
            nks_hint.append(nks_loss.item())
        
            if (iter_i+1) % num_hint == 0:
                tmp_lr = get_lr(optimizer)
                time_diff = time.time() - tik
                print('[Epoch: %d/%d][Iter: %d/%d][cls: %3.4f][reg: %3.4f][nks: %3.4f][lr: %.7f][time: %3.4f]' %
                      (epoch+1, epochs, iter_i+1, epoch_size,
                       np.mean(cls_hint), np.mean(reg_hint),np.mean(nks_hint), tmp_lr, time_diff), flush=True)
                tik = time.time()
            # save model
            break
        if (epoch+1) % args.save_epoch == 0 or epoch > epochs - 5:
            print("===> save model", flush=True)
            tmp_lr = get_lr(optimizer)
            save_model(path_to_save, model, optimizer, epoch, tmp_lr)

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def set_cos_lr(epoch, epochs, optimizer, base_lr, lr_epoch, cos_lr=True):
    if cos_lr:
        # use cos lr
        start_epoch = 5
        end_epoch = 5
        if epoch > start_epoch and epoch <= epochs - end_epoch:
            # use cos lr
            tmp_lr = 0.00001 + 0.5*(base_lr-0.00001) * \
                (1+math.cos(math.pi*(epoch-start_epoch)*1. / (epochs-start_epoch-end_epoch)))
            set_lr(optimizer, tmp_lr)
        elif epoch > epochs - end_epoch:
            tmp_lr = 0.00001
            set_lr(optimizer, tmp_lr)
        # use step lr
    else:
        if epoch in lr_epoch:
            tmp_lr = get_lr(optimizer)
            tmp_lr = tmp_lr * 0.1
            set_lr(optimizer, tmp_lr)


def set_warm_up(epoch, epoch_size, iter_i, optimizer, base_lr):
    # WarmUp strategy for learning rate
    epoch_limit = 1
    if epoch < epoch_limit:
        tmp_lr = base_lr * pow((iter_i+epoch*epoch_size)
                                * 1. / (epoch_limit*epoch_size), 4)
        set_lr(optimizer, tmp_lr)
    elif epoch == epoch_limit and iter_i == 0:
        tmp_lr = base_lr
        set_lr(optimizer, tmp_lr)


def save_model(path_to_save, model, optimizer, epoch, lr):
    print("===> Saving model", flush=True)
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        "lr": lr,
        'optimizer': optimizer.state_dict(),
    }
    path = os.path.join(path_to_save, 'checkpoint_{:d}_{:.6f}.pth.tar'.format(epoch, lr))
    torch.save(checkpoint, path)


if __name__ == "__main__":
    train_model()
