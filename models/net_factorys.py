from models.networks_2d import unet, resunet_plusplus, trans_unet
from models.networks_3d import unet_3d, vnet_attention, unetr_pp
import sys


def get_network(network, in_channels, num_classes, attention_type='Identity', img_size=(256, 256)):
    model = None
    if network == 'unet':
        model = unet.UNet(in_channels, num_classes, attention_type)
    elif network == 'resunet++':
        model = resunet_plusplus.res_unet_plusplus(in_channels, num_classes, attention_type)
    elif network == 'transunet':
        model = trans_unet.get_trans_unet(num_classes, img_size=img_size, attention_type=attention_type)
    else:
        print('sorry, the network you input is not supported yet')
        sys.exit()

    return model


def get_network_3d(network, in_channels, num_classes, attention_type='Identity', img_size=(128, 128, 128)):
    model = None
    if network == 'unet3d':
        model = unet_3d.UNet3D(in_channels, num_classes, attention_type=attention_type)
    elif network == 'vnet':
        model = vnet_attention.VNet(in_channels, num_classes, attention_type=attention_type)
    elif network == 'unetr_pp':
        model = unetr_pp.UNETR_PP(in_channels, num_classes, attention_type=attention_type, img_size=img_size)
    else:
        print('sorry, the network you input is not supported yet')
        sys.exit()

    return model