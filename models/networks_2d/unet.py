import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Attention.attention_factorys import get_attention_module2d


# 卷积块
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out)


# 编码器部分
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, attention_type='Identity', **kwargs):
        super(Encoder, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.attention = get_attention_module2d(attention_type, out_channels, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        # attention for skip connection
        if self.attention is not None:
            x = self.attention(x)
        return x, p


# 解码器部分
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip_connection):
        x = self.up_conv(x)
        x = torch.cat([x, skip_connection], dim=1)  # 拼接跳跃连接
        x = self.conv(x)
        return x


# 中间瓶颈部分
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        return self.conv(x)


# UNet模型
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, attention_type='Identity', **kwargs):
        super(UNet, self).__init__()

        # 编码器
        self.encoder1 = Encoder(in_channels, 64, attention_type=attention_type, **kwargs)
        self.encoder2 = Encoder(64, 128, attention_type=attention_type, **kwargs)
        self.encoder3 = Encoder(128, 256, attention_type=attention_type, **kwargs)
        self.encoder4 = Encoder(256, 512, attention_type=attention_type, **kwargs)

        # 中间瓶颈
        self.bottleneck = Bottleneck(512, 1024)

        # 解码器
        self.decoder4 = Decoder(1024, 512)
        self.decoder3 = Decoder(512, 256)
        self.decoder2 = Decoder(256, 128)
        self.decoder1 = Decoder(128, 64)

        # 最后的卷积层
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # 编码器
        enc1, p1 = self.encoder1(x)
        enc2, p2 = self.encoder2(p1)
        enc3, p3 = self.encoder3(p2)
        enc4, p4 = self.encoder4(p3)

        # 中间瓶颈
        bottleneck_out = self.bottleneck(p4)

        # 解码器
        dec4 = self.decoder4(bottleneck_out, enc4)
        dec3 = self.decoder3(dec4, enc3)
        dec2 = self.decoder2(dec3, enc2)
        dec1 = self.decoder1(dec2, enc1)

        # 最后的卷积
        out = self.final_conv(dec1)
        return out


# 测试代码
if __name__ == "__main__":
    # 假设输入图像大小为 (batch_size, channels, height, width)
    attention_list = ['Identity', 'SE', 'CBAM', 'ECA', 'SimAM', 'SIAM', 'EFFTA']
    for att_name in attention_list:
        model = UNet(3, 2, attention_type=att_name)  # 输入3通道（RGB），输出1通道（分割）
        input_tensor = torch.randn(1, 3, 256, 256)  # 随机生成一个输入图像
        output_tensor = model(input_tensor)
        print(output_tensor.shape)  # 输出的形状

        # from utils.metrics import cal_params_flops
        # cal_params_flops(model, input_tensor)
        #
        # 模型参数
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters: {total_params}")
        # 模型是 xx M
        # 模型参数量是 xx M
        print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")

        from ptflops import get_model_complexity_info

        # 定义输入尺寸
        input_res = (3, 256, 256)  # 根据模型输入调整尺寸

        # 计算FLOPs和参数量
        macs, params = get_model_complexity_info(model, input_res, as_strings=True, print_per_layer_stat=False)

        print(f"FLOPs (MACs): {macs}")
        print(f"参数量: {params}")


    # from torchsummary import summary
    # #
    # # model = resnet34(num_classes=5)
    # # # # load model weights
    # # model_weight_path = r"xxxxx"  # "./resNet34.pth"
    # # model.load_state_dict(torch.load(model_weight_path), False)
    # print(summary(model.cuda(), (3, 256, 256)))

