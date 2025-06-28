import numpy as np

def dataset_cfg(dataset_name):

    config = {
        'GlaS':
            {
                'IN_CHANNELS': 3,
                'NUM_CLASSES': 2,
                'MEAN': [0.787803, 0.512017, 0.784938],
                'STD': [0.428206, 0.507778, 0.426366],
                'RESIZE':(256, 256),
                'PALETTE': list(np.array([
                    [0, 0, 0],
                    [255, 255, 255],
                ]).flatten())
            },
        'ISIC-2017':
            {
                'IN_CHANNELS': 3,
                'NUM_CLASSES': 2,
                'MEAN': [0.699002, 0.556046, 0.512134],
                'STD': [0.365650, 0.317347, 0.339400],
                'RESIZE':(256, 256),
                'PALETTE': list(np.array([
                    [0, 0, 0],
                    [255, 255, 255],
                ]).flatten())
            },
        'LA':
            {
                'IN_CHANNELS': 1,
                'NUM_CLASSES': 2,
                'PATCH_SIZE': (112, 112, 80),
                'FORMAT': '.h5',
            },
        'Tooth':
            {
                'IN_CHANNELS': 1,
                'NUM_CLASSES': 2,
                # 'PATCH_SIZE': (160, 160, 96),
                'PATCH_SIZE': (128, 128, 128),
                'FORMAT': '.h5',
            },
    }

    return config[dataset_name]
