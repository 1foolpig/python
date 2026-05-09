"""
第一个模块ChanelAttention：
1.将输入F_in（B,C,H,W）分别进行局部平均池化和全局平均池化，以8*8作为一个窗口（如果不能整除则补齐或删除），得到F_arv和F_max，大小为（B,C,H/8,W/8）
2.然后通过相同的两个1*1卷积层,这两个卷积层第一个将通道数降为d，形状变为(B,d,H/8,W/8),再经过silu激活函数，第二个将通道数升回C，变为(B,C,H/8,W/8)作为输出

第二个模块WindowAttention：

1.将F_arv和F_max分别作为输入展平为(B,C,(1/64)*H*W),即为（B,C,N）,其中B为批次，C为通道数，H,W分别为第一个模块的宽和高
2.分别通过线性变换矩阵分别得到Q、K、V三个空间，大小为（B,N,d_model）,这里的d_model为通道数，你可以自行设置常见的大小
3.分别使用窗口自注意力，将(B,N,d_model)变成(B*nW, ws*ws, d_model)大小，其中ws是window_size，nW为窗口数量
4.设置头数H，将Q,K,V分成d_k=d_model/H的维度，即Q_h、K_h、V_h都为(B*nW,H,ws*ws,d_k)
5.对Q,K,V进行线性变换，然后S=Q ⊙ K^T,得到大小为（B*nW,H,ws*ws,ws*ws）的S
6.S'=S/sqrt(d_k)，再用softmax(S')对其归一化
7.与经过线性变换的V点乘得到O_W（B*nW,H,ws*ws,d_k）,然后拼接回（B*nW,ws*ws,d_model）大小
8.最后经过1*1的conv,最终输入都变为（B,C,H/8,W/8）大小，F_arv输出为F_a,F_max输出为F_m
9.将两个F_a、F_m相加作为最终返回，F_out大小为（B,C,H/8,W/8）

第三个模块：
1.将第二个模块的输出的F_out（B,C,H/8,W/8）通过F'_out=sigmoid(F_out)
2.分别将F'_out进行全局平均池化，大小变为（B,C,1,1）和上采样，大小变为(B,C,H,W)
3.将全局平均池化的结果与最开始的第一个模块的输入F_in相乘（通过广播），再将上采样的与F_in相乘
4.将两个相乘后的结果相加作为最终返回

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt

class ChannelAttention(nn.Module):
    def __init__(self, C, reduction_d=8, window_size=8, wavelet_type='db1'):
        super().__init__()
        self.C = C
        self.d = reduction_d
        self.ws = window_size
        self.wavelet_type = wavelet_type

        self.conv1 = nn.Conv2d(C, self.d, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(self.d, C, kernel_size=1, bias=False)
        self.act = nn.SiLU()

        # 新增全局池化 + 两层卷积分支，用于替代原来的 F_avg 分支
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # 输出 (B,C,1,1)
        self.global_conv1 = nn.Conv2d(C, self.d, kernel_size=1, bias=False)
        self.global_conv2 = nn.Conv2d(self.d, C, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        dec_filters, _ = self.create_wavelet_filter(self.wavelet_type, self.C, self.C, torch.float)
        self.register_buffer('dec_filters', dec_filters)

    # 保留 create_wavelet_filter、wavelet_transform 方法不变
    def create_wavelet_filter(self, wave, in_size, out_size, dtype=torch.float):
        w = pywt.Wavelet(wave)
        dec_hi = torch.tensor(w.dec_hi[::-1], dtype=dtype)
        dec_lo = torch.tensor(w.dec_lo[::-1], dtype=dtype)
        dec_filters = torch.stack([
            dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
            dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
            dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
            dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1),
        ], dim=0)
        dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)
        return dec_filters, None

    def wavelet_transform(self, x, filters):
        b, c, h, w = x.shape
        pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
        # groups=c 表示对每个通道做单独卷积
        x_wt = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
        x_wt = x_wt.view(b, c, 4, h // 2, w // 2)
        return x_wt
    def forward(self, x):
        B, C, H, W = x.shape
        H_cut = (H // self.ws) * self.ws
        W_cut = (W // self.ws) * self.ws
        x_cropped = x[:, :, :H_cut, :W_cut]
        # F_max仍然保持原先流程
        F_max = F.max_pool2d(x_cropped, kernel_size=self.ws, stride=self.ws)
        F_max_wt = self.wavelet_transform(F_max, self.dec_filters)
        F_max_low = F_max_wt[:, :, 0, :, :]

        # F_max 走之前的 conv1+act+conv2，保持不变
        def conv_block(t):
            t = self.conv1(t)
            t = self.act(t)
            t = self.conv2(t)
            return t

        F_max_conv = conv_block(F_max_low)

        # F_avg分支改为全局池化 + 2层conv + sigmoid的通道权重
        F_avg_glob = self.global_avg_pool(x)  # (B, C, 1, 1)
        F_avg_global = self.global_conv1(F_avg_glob)
        F_avg_global = self.act(F_avg_global)
        F_avg_global = self.global_conv2(F_avg_global)
        F_avg_global = self.sigmoid(F_avg_global)  # 通道权重 (B,C,1,1)

        return F_avg_global, F_max_conv, H_cut // 2, W_cut // 2



class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (wh, ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, original_size=None):
        """
        x: (B, H, W, C) 输入特征图（可能已padding）
        original_size: (H_orig, W_orig) 原始尺寸，用于裁剪
        """
        B, H, W, C = x.shape
        wh, ww = self.window_size

        # 自动padding到能被窗口大小整除
        H_pad = ((H + wh - 1) // wh) * wh
        W_pad = ((W + ww - 1) // ww) * ww

        need_pad = (H != H_pad) or (W != W_pad)
        if need_pad:
            x = F.pad(x, (0, 0, 0, W_pad - W, 0, H_pad - H))  # padding格式: (left, right, top, bottom)

        B, H_padded, W_padded, C = x.shape
        num_windows_h = H_padded // wh
        num_windows_w = W_padded // ww

        # 划分窗口
        x = x.view(B, num_windows_h, wh, num_windows_w, ww, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B * num_windows_h * num_windows_w, wh * ww, C)

        # qkv 线性变换
        qkv = self.qkv(x)
        qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = out.transpose(1, 2).reshape(x.shape[0], x.shape[1], C)
        out = self.proj(out)

        # 重组回原图大小
        out = out.view(B, num_windows_h, num_windows_w, wh, ww, C)
        out = out.permute(0, 1, 3, 2, 4, 5).contiguous()
        out = out.view(B, H_padded, W_padded, C)

        # 如果之前做了padding，现在裁剪回原始尺寸
        if need_pad and original_size is not None:
            H_orig, W_orig = original_size
            out = out[:, :H_orig, :W_orig, :]
        elif need_pad:
            out = out[:, :H, :W, :]

        return out


class HybridAttention(nn.Module):
    def __init__(self, C, reduction_d=8, pool_window_size=8, attn_window_size=4, num_heads=4):
        super().__init__()
        self.C = C
        self.pool_window_size = pool_window_size

        # ChannelAttention 模块
        self.channel_attn = ChannelAttention(C, reduction_d, pool_window_size)
        self.window_attn_max = WindowAttention(dim=C, window_size=(attn_window_size, attn_window_size),
                                               num_heads=num_heads)

    def forward(self, X_in):
        B, C, H_orig, W_orig = X_in.shape
        F_avg_global, F_max_conv, H_cut_div2, W_cut_div2 = self.channel_attn(X_in)
        F_max_hwc = F_max_conv.permute(0, 2, 3, 1)
        out_max = self.window_attn_max(F_max_hwc, original_size=(H_cut_div2, W_cut_div2))
        out_max = out_max.permute(0, 3, 1, 2)  # (B,C,H,W)
        # Sigmoid归一化
        out_max = torch.sigmoid(out_max)
        # 将 F_avg_global (B,C,1,1) 直接和输入相乘（广播）
        result1 = X_in * F_avg_global
        # 将 out_max (B,C,H',W') 上采样到原始大小，再和输入相乘
        out_max_upsampled = F.interpolate(out_max, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
        result2 = X_in * out_max_upsampled
        # 相加作为最终输出
        final_output = result1 + result2
        return final_output


# 测试代码
if __name__ == "__main__":
    # 测试不同尺寸
    test_cases = [
        (2, 64, 480, 480),
        (2, 64, 480, 326),
        (1, 32, 256, 193),
        (4, 128, 256,313),
    ]

    for B, C, H, W in test_cases:
        X_in = torch.randn(B, C, H, W)
        model = HybridAttention(C=C, reduction_d=8, pool_window_size=8, attn_window_size=4, num_heads=4)
        output = model(X_in)

        print(f"输入形状: {X_in.shape} -> 输出形状: {output.shape}")
        assert output.shape == X_in.shape, "输出形状应该与输入相同"

    print("所有测试通过！")
