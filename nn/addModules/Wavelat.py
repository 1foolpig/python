import torch
import torch.nn.functional as F
import pywt
import torch.nn as nn
class SimpleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)

    def forward(self, x):
        if x.device != self.conv.weight.device:
            x = x.to(self.conv.weight.device)
        return self.conv(x)


# 小波滤波器创建函数
def create_wavelet_filter(wave, in_size, out_size, device='cuda', type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type, device=device)  # 指定 device
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type, device=device)  # 指定 device
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0).to(device)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type, device=device).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type, device=device).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0).to(device)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters

def wavelet_transform(x, dec_filters):
    b, c, h, w = x.shape
    pad = (dec_filters.shape[2] // 2 - 1, dec_filters.shape[3] // 2 - 1)
    x = F.conv2d(x, dec_filters, stride=2, groups=c, padding=pad)
    return x.reshape(b, c, 4, h // 2, w // 2)

def inverse_wavelet_transform(x, rec_filters):
    b, c, _, h_half, w_half = x.shape
    pad = (rec_filters.shape[2] // 2 - 1, rec_filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, rec_filters, stride=2, groups=c, padding=pad)
    return x

def custom_wavelet_process(x, dec_filters, rec_filters, conv_layer):
    # 进行小波变换
    x_wavelet = wavelet_transform(x, dec_filters)

    # 将小波变换的结果分解成四个 b * c * (h/2) * (w/2) 的张量
    b, c, _, h_half, w_half = x_wavelet.shape

    # 提取四个张量并调整形状
    ll = x_wavelet[:, :, 0, :, :].reshape(b, c, h_half, w_half)  # 低频部分
    lh = x_wavelet[:, :, 1, :, :].reshape(b, c, h_half, w_half)  # 高频部分
    hl = x_wavelet[:, :, 2, :, :].reshape(b, c, h_half, w_half)  # 高频部分
    hh = x_wavelet[:, :, 3, :, :].reshape(b, c, h_half, w_half)  # 高频部分
    # 对每个张量进行 3x3 卷积
    ll_convolved = conv_layer(ll.to(conv_layer.conv.weight.device))  # 对低频部分的卷积
    lh_convolved = conv_layer(lh.to(conv_layer.conv.weight.device))  # 对低频-高频部分的卷积
    hl_convolved = conv_layer(hl.to(conv_layer.conv.weight.device))  # 对高频-低频部分的卷积
    hh_convolved = conv_layer(hh.to(conv_layer.conv.weight.device))  # 对高频部分的卷积

    combined = torch.stack([ll_convolved, lh_convolved, hl_convolved, hh_convolved], dim=2)

    # 进行逆小波变换
    x_reconstructed = inverse_wavelet_transform(combined, rec_filters)

    return x_reconstructed

class ImprovedMCAAttention(nn.Module):
    """
    改进的坐标注意力机制：引入瓶颈压缩与Sigmoid激励
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mip = max(8, in_channels // reduction)

        # 瓶颈层：先压缩通道提取核心特征
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()  # 使用SiLU增强非线性表达 [cite: 60]

        # 分离卷积：分别学习水平和垂直方向的权重
        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # 坐标池化 [cite: 64, 90]
        x_h = F.adaptive_avg_pool2d(x, (h, 1))
        x_w = F.adaptive_avg_pool2d(x, (1, w)).permute(0, 1, 3, 2)

        # 特征聚合
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))

        # 分解并恢复维度
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # 生成注意力权重
        a_h = self.sigmoid(self.conv_h(x_h))
        a_w = self.sigmoid(self.conv_w(x_w))

        # 残差式输出，确保信息流稳定
        return identity * a_h * a_w


class MCA_Wave(nn.Module):
    """
    输入形式适配 YOLOv8: [-1, 1, MCA_Wave, [ch_out]]
    """

    def __init__(self, in_channels, wave='haar'):
        super(MCA_Wave, self).__init__()
        # 使用组卷积（Groups=in_channels）处理小波分量，降低参数量同时保持通道独立性
        self.conv_wave = nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size=3,
                                   padding=1, groups=in_channels * 4, bias=False)
        self.bn_wave = nn.BatchNorm2d(in_channels * 4)

        self.mca_attention = ImprovedMCAAttention(in_channels)

        # 滤波器注册为 Buffer
        dec_filters, rec_filters = create_wavelet_filter(wave, in_channels, in_channels, device='cpu')
        self.register_buffer('dec_filters', dec_filters)
        self.register_buffer('rec_filters', rec_filters)

    def forward(self, x):
        # 1. 原始残差分支
        identity = x

        # 2. 小波路径处理
        b, c, h, w = x.shape
        x_wavelet = wavelet_transform(x, self.dec_filters)
        # 调整为卷积格式 (b, c*4, h/2, w/2)
        x_wavelet = x_wavelet.reshape(b, c * 4, h // 2, w // 2)
        x_wavelet = self.bn_wave(self.conv_wave(x_wavelet))

        # 3. 逆变换恢复
        x_reconstructed = inverse_wavelet_transform(x_wavelet.reshape(b, c, 4, h // 2, w // 2), self.rec_filters)

        # 4. 注意力分支与残差融合 [cite: 46, 51]
        out = self.mca_attention(x_reconstructed)

        return out + identity  # 引入残差连接，大幅提升训练稳定性