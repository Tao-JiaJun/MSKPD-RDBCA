MBNV2_RDB_384 = {
    'BACKBONE':{
        'NAME': 'MBNV2',
        'PRETRAINED': True,
    },
    'NECK': {
        'NAME': 'RDB',
        'C3_CHANNEL': 32,
        'C4_CHANNEL': 96,
        'C5_CHANNEL': 160,
        'OUT_CHANNEL': 128,
        'USE_P7': False,
    },
    'HEAD':{
        'IN_CHANNEL': 128,
        'OUT_CHANNEL': 128,
        'USE_P7': False,
    }
}
MBNV2_RDB_512 = {
    'BACKBONE':{
        'NAME': 'MBNV2',
        'PRETRAINED': True,
    },
    'NECK': {
        'NAME': 'RDB',
        'C3_CHANNEL': 32,
        'C4_CHANNEL': 96,
        'C5_CHANNEL': 160,
        'OUT_CHANNEL': 128,
        'USE_P7': True,
    },
    'HEAD':{
        'IN_CHANNEL': 128,
        'OUT_CHANNEL': 128,
        'USE_P7': True,
    }
}
