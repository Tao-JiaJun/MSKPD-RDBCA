RES18_RDB_384 = {
    'BACKBONE':{
        'NAME': 'RESNET18',
        'PRETRAINED': True,
    },
    'NECK': {
        'NAME': 'RDB',
        'C3_CHANNEL': 128,
        'C4_CHANNEL': 256,
        'C5_CHANNEL': 512,
        'OUT_CHANNEL': 128,
        'USE_P7': False,
    },
    'HEAD':{
        'IN_CHANNEL': 128,
        'OUT_CHANNEL': 128,
        'USE_P7': False,
    }
}
RES18_RDB_512 = {
    'BACKBONE':{
        'NAME': 'RESNET18',
        'PRETRAINED': True,
    },
    'NECK': {
        'NAME': 'RDB',
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
RES50_RDB_384 = {
    'BACKBONE':{
        'NAME': 'RESNET50',
        'PRETRAINED': True,
    },
    'NECK': {
        'NAME': 'RDB',
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
RES50_RDB_512 = {
    'BACKBONE':{
        'NAME': 'RESNET50',
        'PRETRAINED': True,
    },
    'NECK': {
        'NAME': 'RDB',
        'C3_CHANNEL': 512,
        'C4_CHANNEL': 1024,
        'C5_CHANNEL': 2048,
        'OUT_CHANNEL': 128,
        'USE_P7': True,
    },
    'HEAD':{
        'IN_CHANNEL': 128,
        'OUT_CHANNEL': 128,
        'USE_P7': True,
    }
}


