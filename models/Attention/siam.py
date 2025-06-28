# Understand from SIAM: A parameter-free, Spatial Intersection Attention Module

import torch
import torch.nn as nn


class ReLU1(nn.Hardtanh):
    def __init__(self, inplace=False):
        super(ReLU1, self).__init__(0, 1, inplace=inplace)

    def forward(self, x):
        return super(ReLU1, self).forward(x)


def channel_avg_pooling(x, mean_dim: int or list = 1):
    return x.mean(dim=mean_dim, keepdim=True)


class SIAM(torch.nn.Module):
    def __init__(self):
        super(SIAM, self).__init__()
        # self.activation = ReLU1(inplace=True)
        self.activation = ReLU1()
        # self.activation = nn.ReLU6(inplace=True)

    def forward(self, x):
        # Feature: C * D * H * W
        b, c, d, h, w = x.size()
        # 3-D feature voxel: D * H * W
        mean_3d_feature_map = channel_avg_pooling(x)  # batch, 1, d, h, w
        mean_3d_feature_map = mean_3d_feature_map.squeeze(1)  # batch, d, h, w

        # 2-D query map: H * W for squeezing D
        query_d = channel_avg_pooling(mean_3d_feature_map, mean_dim=1)  # batch, 1, h, w
        # 2-D query map: D * W for squeezing H
        query_h = channel_avg_pooling(mean_3d_feature_map, mean_dim=2)  # batch, d, 1, w
        # 2-D query map: D * H for squeezing W
        query_w = channel_avg_pooling(mean_3d_feature_map, mean_dim=3)  # batch, d, h, 1

        # feature_d: batch, d*c, h, w
        feature_d = x.permute(0, 2, 1, 3, 4).reshape(b, d * c, h, w)

        # feature_h: batch, h*c, d, w
        feature_h = x.permute(0, 3, 1, 2, 4).reshape(b, h * c, d, w)

        # feature_w: batch, w*c, d, h
        feature_w = x.permute(0, 4, 1, 2, 3).reshape(b, w * c, d, h)

        # permute
        # # query_d not change : batch, 1, h, w
        # query_d = query_d.permute(0, 2, 3, 1).reshape(b, h * w, 1)
        # query_h: batch, d, 1, w -> batch, 1, d, w
        query_h = query_h.permute(0, 2, 1, 3)
        # query_w: batch, d, h, 1 -> batch, 1, d, h
        query_w = query_w.permute(0, 3, 1, 2)

        # expand
        # query_d: batch, 1, h, w -> batch, d*c, h, w
        query_d = query_d.expand(b, d * c, h, w)
        # query_h: batch, 1, d, w -> batch, h*c, d, w
        query_h = query_h.expand(b, h * c, d, w)
        # query_w: batch, 1, d, h -> batch, w*c, d, h
        query_w = query_w.expand(b, w * c, d, h)

        # map_feature_query_d: dot product of feature_d and query_d / d
        map_feature_query_d = torch.mul(feature_d, query_d) / d
        map_feature_query_h = torch.mul(feature_h, query_h) / h
        map_feature_query_w = torch.mul(feature_w, query_w) / w

        # reshape to original shape
        # map_feature_query_d: DC * H * W -> C * D * H * W
        map_feature_query_d = map_feature_query_d.reshape(b, c, d, h, w)

        # map_feature_query_h: HC * H * W -> C * D * H * W
        map_feature_query_h = map_feature_query_h.reshape(b, c, d, h, w)

        # map_feature_query_w: WC * H * W -> C * D * H * W
        map_feature_query_w = map_feature_query_w.reshape(b, c, d, h, w)

        map_feature_query_d = self.activation(map_feature_query_d)
        map_feature_query_h = self.activation(map_feature_query_h)
        map_feature_query_w = self.activation(map_feature_query_w)

        out = map_feature_query_d + map_feature_query_h + map_feature_query_w
        # return out
        return out * x


class SIAM2D(torch.nn.Module):
    def __init__(self):
        super(SIAM2D, self).__init__()
        self.activation = ReLU1(inplace=True)

    def forward(self, x):
        # Feature: C * H * W
        b, c, h, w = x.size()
        # 2-D feature map: H * W
        z = channel_avg_pooling(x)  # batch, 1, h, w
        z = z.squeeze(1)  # batch, h, w

        # 1-D query: 1 * W for squeezing H
        a_col = channel_avg_pooling(z, mean_dim=1)  # batch, 1, w
        # 1-D query: H * 1 for squeezing W
        a_row = channel_avg_pooling(z, mean_dim=2)  # batch, h, 1

        # Feature Column: HC * W
        f_col = x.permute(0, 2, 1, 3).reshape(b, h * c, w)

        # Feature Row: WC * H
        f_row = x.permute(0, 3, 1, 2).reshape(b, w * c, h)

        a_col_t = a_col.permute(0, 2, 1)  # batch, w, 1

        # map_col = F_col * A_col_T / h
        # map_col: HC * W dot W * 1 -> HC * 1
        m_col = torch.matmul(f_col, a_col_t) / h
        m_row = torch.matmul(f_row, a_row) / w

        # map_col: HC * 1 -> C * H * 1
        m_col = m_col.reshape(b, c, h, 1)
        m_row = m_row.reshape(b, c, 1, w)

        m_col = self.activation(m_col)
        m_row = self.activation(m_row)

        out = m_col + m_row
        return out * x
        # return out


def test_siam():
    batch = 1
    channel = 1
    height = 2
    width = 2
    depth = 2

    # 设置随机种子
    torch.manual_seed(0)
    input_x = torch.randn(batch, channel, height, width, depth)
    print(input_x)
    print(input_x.size())

    siam = SIAM()
    out_tensor = siam(input_x)
    print(out_tensor.size())
    print(out_tensor)


def test_siam2d():
    batch = 2
    channel = 2
    height = 3
    width = 3

    # 设置随机种子
    torch.manual_seed(0)
    input_x = torch.randn(batch, channel, height, width)
    print(input_x)
    print(input_x.size())

    siam = SIAM2D()
    out_tensor = siam(input_x)
    print(out_tensor.size())
    print(out_tensor)

if __name__ == "__main__":
    test_siam()