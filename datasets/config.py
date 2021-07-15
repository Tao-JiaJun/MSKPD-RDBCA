import os.path

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))
MEANS = (104, 117, 123)

voc_384 = {
    'root': '../datasets/VOCdevkit/',
    'num_cls': 20,
    'input_size': [384, 384],
    'strides': [8, 16, 32, 64],
    'scales': [0, 48, 96, 192, 1e10],
    'max_epoch': 180,
    'lr_epoch': (90, 120),#define but do not use it
    'name': 'VOC',
}
voc_512 = {
    'root': '../datasets/VOCdevkit/',
    'num_cls': 20,
    'input_size': [512, 512],
    'strides': [8, 16, 32, 64, 128],
    'scales': [0, 48, 96, 192, 384, 1e10],
    'max_epoch': 180,
    'lr_epoch': (90, 120),#define but do not use it
    'name': 'VOC',
}
