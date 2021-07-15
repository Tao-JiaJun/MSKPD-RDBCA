import sys
import glob
import time
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from datasets import *
from utils import get_parameter_number
import torch.utils.data as data
from datasets.voc import VOC_CLASSES, VOCDataset
from datasets.config import voc_384,voc_512
from model.detector import Detector
from configs import *
from tqdm import tqdm

               
def eval_model(model, device, dataloader, num_images):
    num_images = num_images
    image_names = []
    scale_list = []
    detec_result = [[] for _ in range(len(VOC_CLASSES)) ]
    counter_images_per_class = {}
    detec_time = 0
    print("===> Detecting")
    with tqdm(total=len(dataloader)) as pbar:
        for i, (img_info, img, targets) in enumerate(dataloader):
            img = img.to(device)
            tic = time.time()
            bbox_list, score_list, cls_list = model(img)
            detec_time += time.time() - tic
            for j, (info, gts, bboxes, scores, cls_indexes) in enumerate(zip(img_info, targets, bbox_list, score_list, cls_list)):
                # get image info
                file_path = info[0]
                file_name = info[1]
                w = info[2]
                h = info[3]
    
                img_bbox = []
                scale = np.array([[w, h, w, h]])
                bboxes *= scale
                for bbox, score, cls_index in zip(bboxes, scores, cls_indexes):
                    cls_name = VOC_CLASSES[cls_index] 
                    box = "{:.1f} {:.1f} {:.1f} {:.1f}".format(bbox[0]+1,bbox[1]+1,bbox[2]+1,bbox[3]+1)
                    detec_result[cls_index].append({"file_path":file_path,"file_name":file_name,"score":score,"bbox":box})
                
                
                with open("./eval/input/detection-results/"+file_name+".txt", "w+") as new_f:
                    for bbox, score, cls_index in zip(bboxes, scores, cls_indexes):
                        cls_name = VOC_CLASSES[cls_index] 
                        new_f.write("%s %s %s %s %s %s\n" % (cls_name,score,str(int(bbox[0]+1)), str(int(bbox[1]+1)), str(int(bbox[2]+1)), str(int(bbox[3]+1))))
          
            pbar.update(1)
    FPS = num_images / detec_time
    return FPS, detec_time

if __name__ == '__main__':
    num_classes = len(VOC_CLASSES)
    if  torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        device = "cuda"
    else:
        device = "cpu"

    voc = voc_512
    model = Detector(device, input_size=voc['input_size'], num_cls=20, strides = voc['strides'], scales=voc['scales'], cfg=MBNV2_RDB_512)
    print('Let us test MSKPD-RDBCA on the VOC0712 dataset ......')

    # load params
    checkpoint = torch.load('./weights/MBNV2_RDB_512_79.00.pth',map_location=device)
    model.load_state_dict(checkpoint)
    # NOTE: In order to save upload space, we only have model parameters here. \
    # NOte: If you start training from the beginning, please use the following code
    # model.load_state_dict(checkpoint['model'])
   
    model = model.to(device)
    model.eval()
    print(get_parameter_number(model))
    print('Finished loading model!')
    # load data
    input = torch.randn(8, 3, 512, 512).to(device)
    output = model(input)
    print("warm up over")
    dataset = VOCDataset(root=voc["root"], 
                         image_sets=[('2007', 'test')],
                         transform=BaseTransform(model.input_size, (MEANS)),
                         is_test=True)
    batch_size = 1
    dataloader = data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=detection_collate)
    
    # evaluation
    tic = time.time()
    FPS, detec_time = eval_model(model, device, dataloader, len(dataset))
    total_time = time.time() - tic
    print("[FPS:%.3f][detec_time:%.2f][total_time:%.2f]" % (FPS, detec_time, total_time))
    print("now you can use 'python eval/get_map.py' to get mAP.")

