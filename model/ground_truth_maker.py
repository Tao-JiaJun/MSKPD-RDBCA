import numpy as np
import torch
import time
import cv2
class GTMaker(object):
    def __init__(self, input_size, num_cls, strides, scales):
        """
        # Author: 
        #   Taojj
        # Args: 
        #   input_size  : int 
        #   num_cls     : int  
        #   strides     : list 
        #   scales      : list
        """
        self.num_cls = num_cls
        self.strides = strides
        self.scales = scales
        self.w = input_size[0]
        self.h = input_size[1]
        self.feature_map_points = [(self.w // s) * (self.h // s) for s in self.strides]
        # Calculate the space needed for flattening
        self.total = sum(self.feature_map_points)
        # offsets
        self.regular_index = [0]
        for i,_ in enumerate(strides):
            self.regular_index.append(sum(self.feature_map_points[j] for j in range(i+1)))
        """
        # num_cls     = 20
        # l, t, r, b  =  4
        # pos         =  1
        # +           = 25
        """
        self.gt_num = num_cls + 4 + 1
        # Calculate how many detection points are in each scale
        self.w_detec_point = [self.get_step_pixel(self.w, s) for s in self.strides]
        self.h_detec_point = [self.get_step_pixel(self.h, s) for s in self.strides]
    def __call__(self, gt_list):
        batch_size = len(gt_list)
        gt_np = np.zeros([batch_size, self.total, self.gt_num])
        # handle the data in each batch
        for batch_i, gts in enumerate(gt_list):
            # handle the bbox and label of each image
            for gt in gts:
                box   = gt[:4]
                label = gt[-1]
                # Restore original size
                box = (np.array(box) * np.array([self.w,self.h,self.w,self.h])).tolist()
                # Check for dirty data
                if self.is_dirty_data(box):
                    continue
                # TODO 1 determine which scale interval the shortest side of this picture falls into (sec. 2.3)
                scale_i = self.fall_into_scale(box)
                num_w = len(self.h_detec_point[scale_i])
                # TODO2 center of the ground truth
                cx = (box[2]+box[0]) // 2
                cy = (box[3]+box[1]) // 2
                # TODO 3 Get offset and ltrb
                # get offset Number of rows * row + column
                ci = int(cx // self.strides[scale_i])
                cj = int(cy // self.strides[scale_i])
                offset = int(num_w * cj + ci)
                # get l, t, r, b
                x = self.w_detec_point[scale_i][ci]
                y = self.h_detec_point[scale_i][cj]
                l, t, r, b = self.get_ltrb(x, y, box)
                # TODO 4 Calculate the Gaussian radius (sec. 2.4)
                box_w_s = (box[2] - box[0]) / self.strides[scale_i]
                box_h_s = (box[3] - box[1]) / self.strides[scale_i] 
                gaussian_r = max(0,self.gaussian_radius([box_w_s, box_h_s]))
                diameter = 2*gaussian_r + 1
                # TODO 5 Calculate the starting offset of the feature layer
                # start offset
                start_index = self.regular_index[scale_i]
                # end offset
                end_index = self.regular_index[scale_i+1]
                # TODO 6 Mark the selected positive sample on the detection point (sec. 2.4)
                index = start_index + offset
                gt_np[batch_i, index, int(label)] = 1.0
                gt_np[batch_i, index, self.num_cls:self.num_cls + 4] = np.array([l, t, r, b])
                gt_np[batch_i, index, self.num_cls + 4] = 1.0
                # Generate heat map
                grid_x_mat, grid_y_mat = np.meshgrid(np.arange(num_w), np.arange(num_w))
                heatmap = np.exp(- (grid_x_mat - ci) ** 2 / (2*(diameter/3)**2) - (grid_y_mat - cj)**2 / (2*(diameter/3)**2)).reshape(-1)
                # Get previous records
                pre_v = gt_np[batch_i, start_index : end_index, int(label)]
                # Save the higher value of both
                gt_np[batch_i, start_index : end_index, int(label)] = np.maximum(heatmap, pre_v)
        return gt_np

    @staticmethod
    def gaussian_radius(det_size, min_overlap=0.5):
        height, width = det_size

        a1 = 1
        b1 = (height + width)
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2 #
        #r1 = (b1 - sq1) / (2 * a1)

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2
        #r2 = (b2 - sq2) / (2 * a2)

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        # r3 = (b3 + sq3) / 2
        r3 = (b3 + sq3) / (2 * a3)
        return min(r1, r2, r3)

    def fall_into_scale(self, box):
        xmin, ymin, xmax, ymax = box[:]
        M = min(xmax - xmin, ymax - ymin)
        scale = -1
        for i in range(len(self.scales) - 1):
            if self.scales[i] < M and M <= self.scales[i+1]:
                return i
        return scale

    def get_ltrb(self, x, y, box):
        xmin, ymin, xmax, ymax = box[:]
        l = max(0, x - xmin)
        r = max(0, xmax - x)
        t = max(0, y - ymin)
        b = max(0, ymax - y)
        return l, t, r, b


    def get_step_pixel(self, pixel, s):
        steps = pixel // s
        x = [i * s + s // 2 for i in range(steps)]
        return x

    @staticmethod
    def is_dirty_data(box):
        xmin, ymin, xmax, ymax = box[:]
        if xmax - xmin < 1e-10 or ymax - ymin < 1e-10:
            return True
        return False

def vis_heatmap(targets, size):
    # vis heatmap
    HW = targets.shape[0]
    h = int(np.sqrt(HW))
    heatmap = np.zeros([size,size])
    for c in range(0,20):
        tmp_map = targets[:, c].reshape(h, h)
        if sum(sum(tmp_map)) == 0:
            continue
        tmp_map = cv2.resize(tmp_map,(size, size))
        heatmap += tmp_map
    return heatmap
if __name__ == "__main__":
    import sys
    sys.path.append("./")
    from datasets.augmentations import SSDAugmentation
    from datasets.voc import VOCDataset
    
    def base_transform(image, size, mean):
        x = cv2.resize(image, (size[1], size[0])).astype(np.float32)
        x -= mean
        x = x.astype(np.float32)
        return x

    class BaseTransform:
        def __init__(self, size, mean):
            self.size = size
            self.mean = np.array(mean, dtype=np.float32)

        def __call__(self, image, boxes=None, labels=None):
            return base_transform(image, self.size, self.mean), boxes, labels
    strides = [8,16,32,64]
    scales  = [0, 48, 96, 192, 1e10]
    size = 640
    gt_maker = GTMaker([size,size], 20, strides, scales)
    # dataset
    
    VOC_ROOT = '../datasets/VOCdevkit/'
    dataset = VOCDataset(VOC_ROOT, [('2007', 'trainval')],
                            #transform=SSDAugmentation([size,size]))
                            transform=BaseTransform([size,size],(0,0,0)))
    class_color = [(np.random.randint(255),np.random.randint(255),np.random.randint(255)) for _ in range(4)]

    for i in range(len(dataset)):
        img_id, im, gt = dataset.pull_item(i)
        print(img_id)
        res = gt_maker([gt])
        img = im.permute(1,2,0).numpy()[:, :, (2, 1, 0)].astype(np.uint8).copy()
        for box in gt:
            xmin, ymin, xmax, ymax, _ = box
            xmin *= size
            ymin *= size
            xmax *= size
            ymax *= size
            i = gt_maker.fall_into_scale([xmin,ymin,xmax,ymax])
            img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_color[i], 2)

            
        points_list = []
        # for resolution of 320
        # fm1 = res[0][0:1600]
        # fm2 = res[0][1600:2000]
        # fm3 = res[0][2000:2100]
        # fm4 = res[0][2100:2125]
        # for resolution of 640
        fm1 = res[0][0:6400]
        fm2 = res[0][6400:8000]
        fm3 = res[0][8000:8400]
        fm4 = res[0][8400:8500]
        fms = [fm1,fm2,fm3,fm4]
        heatmap = np.zeros([size,size])
        for scale_i, fm in enumerate(fms):
            hmap = vis_heatmap(fm, size)
            heatmap += hmap
            for offset, r in enumerate(fm):
                if r[-1] == 1:
                    i = offset %  len(gt_maker.w_detec_point[scale_i])
                    j = offset // len(gt_maker.h_detec_point[scale_i])
                    x = gt_maker.w_detec_point[scale_i][i]
                    y = gt_maker.h_detec_point[scale_i][j]
                    points_list.append([(x,y),(255,255,255)])
        for point in points_list:
            cv2.circle(img, point[0], radius=2,color=point[1], thickness=8)
        print(len(points_list))
        heatmap[heatmap>=1.0] = 1.0
        # convert heat map to RGB format
        heatmap = np.uint8(255 * heatmap)  
        # apply the heat map to the original image
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        img_add = cv2.addWeighted(heatmap, 0.2, img, 0.5, 1)
  
        cv2.imshow('gt', img_add)
        cv2.waitKey(0)