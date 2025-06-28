import torch.nn as nn
import math


class ECA(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ECA, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)  # batch * channel * 1 * 1
        y = y.squeeze(-1).transpose(-1, -2)  # batch * channel * 1 -> batch * 1 * channel
        y = self.conv(y)  # batch * 1 * channel
        y = y.transpose(-1, -2).unsqueeze(-1)  # batch * channel * 1 -> batch * channel * 1 * 1
        # y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        out = x * y.expand_as(x)
        return out


class ECA3D(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ECA3D, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-2, -3)
        # print(y.shape)
        y = self.conv(y)
        y = y.transpose(-2, -3).unsqueeze(-1)
        # y = self.conv(y.squeeze(-1).transpose(-1, -2).transpose(-2, -3)).transpose(-1, -2).transpose(-2, -3).unsqueeze(-1)
        y = self.sigmoid(y)
        out = x * y.expand_as(x)
        return out


if __name__ == '__main__':
    import torch

    # eca = ECA(64)
    eca3d = ECA3D(16)

    # x = torch.randn(1, 64, 56, 56)
    x3d = torch.randn(2, 16, 112, 112, 80)

    # out = eca(x)
    out3d = eca3d(x3d)

    # print(out.shape)
    print(out3d.shape)