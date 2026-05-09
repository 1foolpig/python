import torch
import torch.nn as nn
import torch.nn.functional as F

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class MCAAttention(nn.Module):
    def __init__(self, in_channels):
        super(MCAAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.tanh = nn.Tanh()
    def forward(self, x):
        # 输入形状: (batch_size, channels, height, width)
        batch_size, channels, height, width = x.size()
        # 池化操作
        avg_x = F.adaptive_avg_pool2d(x, (height, 1))  # (batch_size, channels, height, 1)
        avg_y = F.adaptive_avg_pool2d(x, (1, width))  # (batch_size, channels, 1, width)
        # 逐通道点乘
        attention_map = avg_x * avg_y  # (batch_size, channels, height, width)
        # 卷积和激活
        weight = self.tanh(self.conv(attention_map))  # (batch_size, channels, height, width)
        output = x * weight  # (batch_size, channels, height, width)
        # 打印最终输出特征图
        # print("最终输出特征图形状:", output.shape)
        # print("最终输出特征图内容:", output)
        return output
#MCA注意力机制
class MCA(nn.Module):
    def __init__(self, inp, reduction=32):
        super(MCA, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out