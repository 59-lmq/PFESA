import torch
import torch.nn as nn
import os
import numpy as np
from collections import OrderedDict
from torch.nn import init

from models.Attention.attention_factorys import get_attention_module_3d


class GetAttention(nn.Module):
    def __init__(self, attention_type, out_channels):
        super(GetAttention, self).__init__()

        self.attention = get_attention_module_3d(attention_type, out_channels)

    def forward(self, x):
        if self.attention is not None:
            x = self.attention(x)
        return x


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def passthrough(x, **kwargs):
    return x


def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)


class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)

        self.bn1 = torch.nn.BatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, in_channels, elu):
        super(InputTransition, self).__init__()
        self.num_features = 16
        self.in_channels = in_channels

        self.conv1 = nn.Conv3d(self.in_channels, self.num_features, kernel_size=5, padding=2)

        self.bn1 = torch.nn.BatchNorm3d(self.num_features)

        self.relu1 = ELUCons(elu, self.num_features)

    def forward(self, x):
        out = self.conv1(x)  # 16, d, h, w
        repeat_rate = int(self.num_features / self.in_channels)  # 16 / 1 = 16
        out = self.bn1(out)  # 16, d, h, w
        x16 = x.repeat(1, repeat_rate, 1, 1, 1)  # 16, d, h, w
        return self.relu1(torch.add(out, x16))  # 16, d, h, w


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2 * inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)  # 32, d/2, h/2, w/2
        self.bn1 = torch.nn.BatchNorm3d(outChans)

        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)

        self.bn1 = torch.nn.BatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):
    def __init__(self, in_channels, classes, elu):
        super(OutputTransition, self).__init__()
        self.classes = classes
        self.conv1 = nn.Conv3d(in_channels, classes, kernel_size=5, padding=2)
        self.bn1 = torch.nn.BatchNorm3d(classes)

        self.conv2 = nn.Conv3d(classes, classes, kernel_size=1)
        self.relu1 = ELUCons(elu, classes)

    def forward(self, x):
        # convolve 32 down to channels as the desired classes
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        return out


class VNet(nn.Module):
    """
    Implementations based on the Vnet paper: https://arxiv.org/abs/1606.04797
    """

    def __init__(self, in_channels=1, classes=1, elu=True, attention_type='Identity'):
        super(VNet, self).__init__()
        self.classes = classes
        self.in_channels = in_channels

        self.in_tr = InputTransition(in_channels, elu=elu)  # 16, d, h, w
        self.down_tr32 = DownTransition(16, 1, elu)  # 32, d/2, h/2, w/2
        self.down_tr64 = DownTransition(32, 2, elu)  # 64, d/4, h/4, w/4
        self.down_tr128 = DownTransition(64, 3, elu, dropout=False)  # 128, d/8, h/8, w/8
        self.down_tr256 = DownTransition(128, 2, elu, dropout=False)
        self.up_tr256 = UpTransition(256, 256, 2, elu, dropout=False)
        self.up_tr128 = UpTransition(256, 128, 2, elu, dropout=False)
        self.up_tr64 = UpTransition(128, 64, 1, elu)
        self.up_tr32 = UpTransition(64, 32, 1, elu)
        self.out_tr = OutputTransition(32, classes, elu)

        self.attention_in_tr = GetAttention(attention_type, 16)
        self.attention_down_tr32 = GetAttention(attention_type, 32)
        self.attention_down_tr64 = GetAttention(attention_type, 64)
        self.attention_down_tr128 = GetAttention(attention_type, 128)


    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)

        # attention for skip connection
        out16 = self.attention_in_tr(out16)
        out32 = self.attention_down_tr32(out32)
        out64 = self.attention_down_tr64(out64)
        out128 = self.attention_down_tr128(out128)

        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out


def vnet(in_channels, num_classes):
    model = VNet(in_channels, num_classes)
    init_weights(model, 'kaiming')
    return model


if __name__ == '__main__':
    model = vnet(1, 2)
    model.eval()
    input = torch.rand(2, 1, 128, 128, 128)
    output = model(input)
    output = output.data.cpu().numpy()
    # print(output)
    print(output.shape)

    # 模型参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    # 模型是 xx M
    # 模型参数量是 xx M
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")

    from torchsummary import summary
    #
    # model = resnet34(num_classes=5)
    # # # load model weights
    # model_weight_path = r"xxxxx"  # "./resNet34.pth"
    # model.load_state_dict(torch.load(model_weight_path), False)
    print(summary(model.cuda(), (1, 96, 96, 80)))
