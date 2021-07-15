RES18_FPN_384 = {
    'BACKBONE':{
        'NAME': 'RESNET18',
        'PRETRAINED': True,
    },
    'NECK': {
        'NAME': 'FPN',
        'C3_CHANNEL': 128,
        'C4_CHANNEL': 256,
        'C5_CHANNEL': 512,
        'OUT_CHANNEL': 256,
        'USE_P7': False,
    },
    'HEAD':{
        'IN_CHANNEL': 256,
        'OUT_CHANNEL': 256,
        'USE_P7': False,
    }
}
RES18_FPN_512 = {
    'BACKBONE':{
        'NAME': 'RESNET18',
        'PRETRAINED': True,
    },
    'NECK': {
        'NAME': 'FPN',
        'C3_CHANNEL': 128,
        'C4_CHANNEL': 256,
        'C5_CHANNEL': 512,
        'OUT_CHANNEL': 128,
        'USE_P7': True,
    },
    'HEAD':{
        'IN_CHANNEL': 128,
        'OUT_CHANNEL': 128,
        'USE_P7': True,
    }
}
RES50_FPN_384 = {
    'BACKBONE':{
        'NAME': 'RESNET50',
        'PRETRAINED': True,
    },
    'NECK': {
        'NAME': 'FPN',
        'C3_CHANNEL': 512,
        'C4_CHANNEL': 1024,
        'C5_CHANNEL': 2048,
        'OUT_CHANNEL': 128,
        'USE_P7': False,
    },
    'HEAD':{
        'IN_CHANNEL': 128,
        'OUT_CHANNEL': 128,
        'USE_P7': False,
    }
}