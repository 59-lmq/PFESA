import torch
import torch.nn as nn
from models.Attention.attention_factorys import get_attention_module_3d


def init_func(m, init_type, gain=0.02):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
            nn.init.normal_(m.weight.data, 0.0, gain)
        elif init_type == 'xavier':
            nn.init.xavier_normal_(m.weight.data, gain=gain)
        elif init_type == 'kaiming':
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            nn.init.orthogonal_(m.weight.data, gain=gain)
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, gain)
        nn.init.constant_(m.bias.data, 0.0)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, bias=True, bn=True, act=nn.ReLU(inplace=True)):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels,
                               kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn1 = nn.BatchNorm3d(out_channels) if bn else None
        self.act1 = act

        self.conv2 = nn.Conv3d(out_channels, out_channels,
                               kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn2 = nn.BatchNorm3d(out_channels) if bn else None
        self.act2 = act

    def forward(self, x):
        x = self.conv1(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        if self.bn2 is not None:
            x = self.bn2(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, bias=True, bn=True, act=nn.ReLU(inplace=True),
                 attention_type=None):
        super(EncoderBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels, kernel_size, stride, padding, bias, bn, act)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.attention = get_attention_module_3d(attention_type, out_channels)

    def forward(self, x):
        x = self.conv_block(x)
        pool = self.pool(x)
        # attention for skip connection
        if self.attention is not None:
            x = self.attention(x)
        return x, pool


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, output_padding=0, bias=True, bn=True, act=nn.ReLU(inplace=True),
                 attention_type=None):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2,
                                     padding=0, output_padding=output_padding, bias=bias)
        self.conv_block = ConvBlock(out_channels * 2, out_channels, kernel_size, stride, padding, bias, bn, act)

    def forward(self, x, skip_connection):
        x = self.up(x)
        x = torch.cat([x, skip_connection], dim=1)
        x = self.conv_block(x)
        return x


class UNet3D(nn.Module):
    def __init__(self, in_channels, class_nums, base_channels=64, depth=4,
                 kernel_size=3, stride=1, padding=1, output_padding=0,
                 bias=True, bn=True, act=nn.ReLU(inplace=True),
                 attention_type=None):
        super(UNet3D, self).__init__()

        # encoder part
        self.encoder1 = EncoderBlock(in_channels, base_channels,
                                     kernel_size, stride, padding, bias, bn, act, attention_type)
        self.encoder2 = EncoderBlock(base_channels, base_channels * 2,
                                     kernel_size, stride, padding, bias, bn, act, attention_type)
        self.encoder3 = EncoderBlock(base_channels * 2, base_channels * 4,
                                        kernel_size, stride, padding, bias, bn, act, attention_type)
        self.encoder4 = EncoderBlock(base_channels * 4, base_channels * 8,
                                        kernel_size, stride, padding, bias, bn, act, attention_type)

        # bottleneck part
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16,
                                    kernel_size, stride, padding, bias, bn, act)

        # decoder part
        self.decoder4 = DecoderBlock(base_channels * 16, base_channels * 8,
                                     kernel_size, stride, padding, output_padding, bias, bn, act, attention_type)
        self.decoder3 = DecoderBlock(base_channels * 8, base_channels * 4,
                                        kernel_size, stride, padding, output_padding, bias, bn, act, attention_type)
        self.decoder2 = DecoderBlock(base_channels * 4, base_channels * 2,
                                        kernel_size, stride, padding, output_padding, bias, bn, act, attention_type)
        self.decoder1 = DecoderBlock(base_channels * 2, base_channels,
                                        kernel_size, stride, padding, output_padding, bias, bn, act, attention_type)

        # output part
        self.last_conv = nn.Conv3d(base_channels, class_nums, kernel_size=1, stride=1, padding=0, bias=bias)

        self.init_weights(init_type='xavier')

    def forward(self, x):
        # encoder part
        x1, p1 = self.encoder1(x)
        x2, p2 = self.encoder2(p1)
        x3, p3 = self.encoder3(p2)
        x4, p4 = self.encoder4(p3)

        # bottleneck part
        x = self.bottleneck(p4)

        # decoder part
        x = self.decoder4(x, x4)
        x = self.decoder3(x, x3)
        x = self.decoder2(x, x2)
        x = self.decoder1(x, x1)

        # output part
        x = self.last_conv(x)
        return x

    def init_weights(self, init_type='xavier'):
        for m in self.modules():
            init_func(m, init_type)


if __name__ == '__main__':
    attention_list = ['Identity', 'SE', 'CBAM', 'ECA', 'SimAM', 'SIAM', 'EFFTA']
    for att_name in attention_list:
        model = UNet3D(in_channels=1, class_nums=2, base_channels=16, depth=4,
                       kernel_size=3, stride=1, padding=1, output_padding=0,
                       bias=True, bn=True, act=nn.ReLU(inplace=True),
                       attention_type=att_name)
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

        from ptflops import get_model_complexity_info

        # 定义输入尺寸
        input_res = (1, 128, 128, 128)  # 根据模型输入调整尺寸

        # 计算FLOPs和参数量
        macs, params = get_model_complexity_info(model, input_res, as_strings=True, print_per_layer_stat=False)

        print(f"FLOPs (MACs): {macs}")
        # print(f"参数量: {params}")
        print(f"{att_name} -- 参数量: {params}")

    # from torchstat import stat
    #
    # stat(model, (128, 128, 128))

    # from torchsummary import summary
    # #
    # # model = resnet34(num_classes=5)
    # # # # load model weights
    # # model_weight_path = r"xxxxx"  # "./resNet34.pth"
    # # model.load_state_dict(torch.load(model_weight_path), False)
    # print(summary(model.cuda(), (1, 96, 96, 80)))
    # # print(model, (3, 300, 300))


