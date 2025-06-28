import torch
import torch.nn as nn
import torch.fft as fft


class PFESA(nn.Module):
    """
    PFESA: Parameter-Free Edge Structure Attention
    """
    def __init__(self, base_ratio=0.1):
        super(PFESA, self).__init__()
        self.activation = nn.Sigmoid()  # 归一化注意力权重
        self.base_ratio = base_ratio  # 基础截止频率
        self.eps = 1e-5

    def _edge_attention(self, x):

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)  # batch, c, h, w
        x_var = x.var(dim=[2, 3], keepdim=True)  # batch, c, 1, 1
        y = x_minus_mu_square / (x_var + self.eps)  # batch, c, h, w

        return y

    def _structure_attention(self, x):

        energy_low = torch.pow(x, 2)  # batch, c, h, w
        energy_mu = torch.mean(energy_low, dim=[2, 3], keepdim=True)  # batch, c, 1, 1
        energy_var = torch.var(energy_low, dim=[2, 3], keepdim=True)  # batch, c, 1, 1
        y = (energy_low - energy_mu) / (energy_var + self.eps)  # batch, c, h, w

        y = self.activation(y)

        return y

    def forward(self, x):
        b, c, h, w = x.size()

        # **傅里叶变换**
        x_freq = fft.fftn(x, dim=(-2, -1))  # 只对 H, W 维度进行 FFT
        # 将 freq 移动到中心
        x_freq = fft.fftshift(x_freq, dim=(-2, -1))

        # **低频 & 高频掩码**
        low_freq_mask = self._create_low_freq_mask(h, w, device=x_freq.device)  # 改进版低频掩码
        high_freq_mask = 1 - low_freq_mask  # 高频掩码

        low_freq = x_freq * low_freq_mask
        high_freq = x_freq * high_freq_mask

        # **逆傅里叶变换**
        low_freq = torch.abs(fft.ifftn(low_freq, dim=(-2, -1)))  # batch, c, h, w
        high_freq = torch.abs(fft.ifftn(high_freq, dim=(-2, -1)))  # batch, c, h, w

        # low_edge_att = self._edge_attention(low_freq)
        low_edge_att = self._structure_attention(low_freq)
        high_edge_att = self._edge_attention(high_freq)

        out_att = low_edge_att + high_edge_att

        out_att = self.activation(out_att)

        return out_att * x

    def _create_low_freq_mask(self, h, w, device='cpu'):
        """ 创建适应输入尺寸的低频掩码 """
        mask_ratio = self.base_ratio * min(h, w) / max(h, w)
        y = torch.linspace(-1, 1, h, device=device)
        x = torch.linspace(-1, 1, w, device=device)
        Y, X = torch.meshgrid(y, x, indexing='ij')
        mask = torch.exp(-(Y ** 2 + X ** 2) / (2 * mask_ratio ** 2))
        return mask


class PFESA3D(nn.Module):
    def __init__(self, base_ratio=0.1):
        super(PFESA3D, self).__init__()
        self.activation = nn.Sigmoid()  # 归一化注意力权重
        self.base_ratio = base_ratio
        self.eps = 1e-5

    def _edge_attention(self, x):

        x_minus_mu_square = (x - x.mean(dim=[2, 3, 4], keepdim=True)).pow(2)
        x_var = x.var(dim=[2, 3, 4], keepdim=True)
        y = x_minus_mu_square / (x_var + self.eps)

        return y

    def _structure_attention(self, x):

        energy_low = torch.pow(x, 2)  # batch, c, h, w
        energy_mu = torch.mean(energy_low, dim=[2, 3, 4], keepdim=True)  # batch, c, 1, 1
        energy_var = torch.var(energy_low, dim=[2, 3, 4], keepdim=True)  # batch, c, 1, 1
        y = (energy_low - energy_mu) / (energy_var + self.eps)  # batch, c, h, w

        y = self.activation(y)

        return y

    def forward(self, x):
        b, c, d, h, w = x.size()

        # **傅里叶变换**
        x_freq = fft.fftn(x, dim=(-3, -2, -1))  # 只对 H, W 维度进行 FFT
        # 将 freq 移动到中心
        x_freq = fft.fftshift(x_freq, dim=(-3, -2, -1))

        # **低频 & 高频掩码**
        low_freq_mask = self._create_low_freq_mask(d, h, w, device=x_freq.device)  # 改进版低频掩码
        high_freq_mask = 1 - low_freq_mask  # 高频掩码

        low_freq = x_freq * low_freq_mask
        high_freq = x_freq * high_freq_mask

        # **逆傅里叶变换**
        low_freq = torch.abs(fft.ifftn(low_freq, dim=(-3, -2, -1)))  # batch, c, h, w
        high_freq = torch.abs(fft.ifftn(high_freq, dim=(-3, -2, -1)))  # batch, c, h, w

        # low_edge_att = self._edge_attention(low_freq)
        low_edge_att = self._structure_attention(low_freq)
        high_edge_att = self._edge_attention(high_freq)

        out_att = low_edge_att + high_edge_att

        out_att = self.activation(out_att)

        return out_att * x

    def _create_low_freq_mask(self, d, h, w, device='cpu'):
        """ 创建适应输入尺寸的低频掩码 """
        mask_ratio = self.base_ratio * min(d, h, w) / max(d, h, w)
        z = torch.linspace(-1, 1, d, device=device)
        y = torch.linspace(-1, 1, h, device=device)
        x = torch.linspace(-1, 1, w, device=device)
        Z,Y, X = torch.meshgrid(z, y, x, indexing='ij')
        mask = torch.exp(-(Z ** 2 + Y ** 2 + X ** 2) / (2 * mask_ratio ** 2))
        return mask