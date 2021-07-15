"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

# note: if you used our download scripts, this should be right

VOC_ROOT = '../datasets/VOCdevkit/'

class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOCDataset(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, 
                 root,
                 image_sets,
                 transform=None,
                 target_transform=VOCAnnotationTransform(),
                 dataset_name='VOC0712',
                 is_test = False):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self.is_test = is_test
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'VOC' + year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        img_id, im, gt = self.pull_item(index)

        return img_id, im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]
        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape
        img_info = [img_id[0], img_id[1], width, height]
        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if not self.is_test and self.transform is not None:
           #zero padding
            if height > width:
                img_ = np.zeros([height, height, 3])
                img_[:, :, 0] = np.mean(img[:,:,0])
                img_[:, :, 1] = np.mean(img[:,:,1])
                img_[:, :, 2] = np.mean(img[:,:,2])
                delta_w = height - width
                left = delta_w // 2
                img_[:, left:left+width, :] = img
                offset = np.array([[ left / height, 0.,  left / height, 0.]])
                scale =  np.array([[width / height, 1., width / height, 1.]])
                width = height
            elif height < width:
                img_ = np.zeros([width, width, 3])
                img_[:, :, 0] = np.mean(img[:,:,0])
                img_[:, :, 1] = np.mean(img[:,:,1])
                img_[:, :, 2] = np.mean(img[:,:,2])
                delta_h = width - height
                top = delta_h // 2
                img_[top:top+height, :, :] = img
                offset = np.array([[0.,    top / width, 0.,    top / width]])
                scale =  np.array([[1., height / width, 1., height / width]])
                height = width
            else:
                img_ = img
                scale =  np.array([[1., 1., 1., 1.]])
                offset = np.zeros([1, 4])
            if len(target) == 0:
                target = np.zeros([1, 5])
            else:
                target = np.array(target)
                target[:, :4] = target[:, :4] * scale + offset
            img, boxes, labels = self.transform(img_, target[:, :4], target[:, 4])
            
        
            img_info = [img_id[0], img_id[1], width, height]
        else:
            if len(target) == 0:
                target = np.zeros([1, 5])
            else:
                target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
        # to rgb
        img = img[:, :, (2, 1, 0)]
        target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return img_info, torch.from_numpy(img).permute(2, 0, 1), target

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
if __name__ == "__main__":
    from augmentations import SSDAugmentation
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

    # dataset
    size = 640
    dataset = VOCDataset(VOC_ROOT, [('2007', 'trainval')],
                            #transform=SSDAugmentation([size,size]))
                            transform=BaseTransform([size,size],(0,0,0)))
    for i in range(len(dataset)):
        img_id, im, gt = dataset.pull_item(i)
        img = im.permute(1,2,0).numpy()[:, :, (2, 1, 0)].astype(np.uint8).copy()
        for box in gt:
            xmin, ymin, xmax, ymax, _ = box
            xmin *= size
            ymin *= size
            xmax *= size
            ymax *= size
            img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,0,255), 2)
        cv2.imshow('gt', img)
        cv2.waitKey(0)
