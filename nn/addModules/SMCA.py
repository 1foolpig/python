"""
将空间注意力机制与通道注意力机制串联
第一个模块：
1.首先对输入的F（B,C,H,W）,其中B为批次，C为通道数。H,W分别为高和宽，将其作全局平均池化和全局最大池化，分别得到F_arv和F_max，大小为（B,C,1,1）
2.然后通过相同的两个1*1卷积层,这两个卷积层第一个将通道数降为d，形状变为(B,d,1,1),再经过tanh激活函数，第二个将通道数升回C，变为(B,C,1,1)
3.再经过sigmoid得到权重文件W_C
4.对输入F点乘权重文件W_C得到输出F_cam

第二个模块：
1.将输入F展平为(B,C,H*W),即为（B,C,N）
2.通过线性变换矩阵分别得到Q、K、V三个空间，大小为（B,N,d_model）,这里的d_model你可以自行设置常见的大小
3.使用窗口注意力，将(B,N,d_model)变成(B*nW, ws*ws, d_model)大小，其中ws是window_size，nW为窗口数量
4.设置头数H，将Q,K,V分成d_k=d_model/H的维度，即Q_h、K_h、V_h都为(B*nW,H,ws*ws,d_k)
5.对Q,K,V进行线性变换，然后S=Q ⊙ K^T,得到大小为（B*nW,H,ws*ws,ws*ws）的S
6.S'=S/sqrt(d_k)，再用softmax(S')对其归一化
7.与经过线性变换的V点乘得到O_W（B*nW,H,ws*ws,d_k）,然后拼接回（B*nW,ws*ws,d_model）大小
8.最后经过1*1的conv,最终变为（B,C,H,W）大小

第三个模块：
1.将第一个模块的输出F_cam，经过小波变换后（小波变换使用pywt中的函数）作为第二个模块的输入,第二个模块的输出为F_out
2.输入F通过小波变换和逆变换得到F_wt
3.让F_wt与F_out加权相加，其中计算方式如下：
形状与参数
weight ∈ R^{2×C}，实现为 self.weight = nn.Parameter(torch.randn(2, C))。
计算方式
先对 dim=0 进行 softmax，得到 w ∈ R^{2×C}，再用 w 的两行分别乘以 F_wt、F_out。
F_fused[c] = w[0, c] * F_wt[c] + w[1, c] * F_out[c]，其中 c 表示通道。

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms


class ChannelAttention(nn.Module):
    """通道注意力模块"""

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        d = max(in_channels // reduction, 1)
        self.conv1 = nn.Conv2d(in_channels, d, 1, bias=False)
        self.conv2 = nn.Conv2d(d, in_channels, 1, bias=False)

    def forward(self, x):
        # x: (B, C, H, W)
        F_avg = F.adaptive_avg_pool2d(x, 1)  # (B, C, 1, 1)
        F_max = F.adaptive_max_pool2d(x, 1)  # (B, C, 1, 1)

        # 共享卷积层
        avg_out = self.conv2(torch.tanh(self.conv1(F_avg)))  # (B, C, 1, 1)
        max_out = self.conv2(torch.tanh(self.conv1(F_max)))  # (B, C, 1, 1)

        # 相加后sigmoid
        W_C = torch.sigmoid(avg_out + max_out)  # (B, C, 1, 1)

        # 点乘
        F_cam = x * W_C  # (B, C, H, W)
        return F_cam


class WindowSpatialAttention(nn.Module):
    """窗口空间注意力模块"""

    def __init__(self, in_channels, d_model=64, num_heads=8, window_size=8):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"

        self.in_channels = in_channels
        self.d_model = d_model
        self.num_heads = num_heads
        self.window_size = window_size
        self.d_k = d_model // num_heads

        # Q, K, V线性变换
        self.q_linear = nn.Linear(in_channels, d_model)
        self.k_linear = nn.Linear(in_channels, d_model)
        self.v_linear = nn.Linear(in_channels, d_model)

        # 输出投影
        self.out_conv = nn.Conv2d(d_model, in_channels, 1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.window_size

        # 1. 展平 (B, C, H*W) -> (B, N, C)
        N = H * W
        x_flat = x.view(B, C, N).permute(0, 2, 1)  # (B, N, C)

        # 2. 线性变换得到Q, K, V
        Q = self.q_linear(x_flat)  # (B, N, d_model)
        K = self.k_linear(x_flat)  # (B, N, d_model)
        V = self.v_linear(x_flat)  # (B, N, d_model)

        # 3. 窗口划分
        # 先reshape回空间维度
        Q = Q.permute(0, 2, 1).view(B, self.d_model, H, W)  # (B, d_model, H, W)
        K = K.permute(0, 2, 1).view(B, self.d_model, H, W)
        V = V.permute(0, 2, 1).view(B, self.d_model, H, W)

        # Pad到window_size的整数倍
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            Q = F.pad(Q, (0, pad_w, 0, pad_h))
            K = F.pad(K, (0, pad_w, 0, pad_h))
            V = F.pad(V, (0, pad_w, 0, pad_h))

        _, _, H_pad, W_pad = Q.shape
        nH, nW = H_pad // ws, W_pad // ws
        nW_total = nH * nW

        # 窗口划分: (B, d_model, H_pad, W_pad) -> (B*nW, d_model, ws, ws)
        Q = Q.view(B, self.d_model, nH, ws, nW, ws).permute(0, 2, 4, 1, 3, 5).contiguous()
        Q = Q.view(B * nW_total, self.d_model, ws * ws).permute(0, 2, 1)  # (B*nW, ws*ws, d_model)

        K = K.view(B, self.d_model, nH, ws, nW, ws).permute(0, 2, 4, 1, 3, 5).contiguous()
        K = K.view(B * nW_total, self.d_model, ws * ws).permute(0, 2, 1)

        V = V.view(B, self.d_model, nH, ws, nW, ws).permute(0, 2, 4, 1, 3, 5).contiguous()
        V = V.view(B * nW_total, self.d_model, ws * ws).permute(0, 2, 1)

        # 4. 多头分割: (B*nW, ws*ws, d_model) -> (B*nW, H, ws*ws, d_k)
        Q = Q.view(B * nW_total, ws * ws, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        K = K.view(B * nW_total, ws * ws, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        V = V.view(B * nW_total, ws * ws, self.num_heads, self.d_k).permute(0, 2, 1, 3)

        # 5. 注意力计算: S = Q @ K^T
        S = torch.matmul(Q, K.transpose(-2, -1))  # (B*nW, H, ws*ws, ws*ws)

        # 6. 缩放和softmax
        S_scaled = S / (self.d_k ** 0.5)
        attn = F.softmax(S_scaled, dim=-1)  # (B*nW, H, ws*ws, ws*ws)

        # 保存注意力图用于可视化
        self.attn_map = attn.detach()

        # 7. 与V点乘
        O_W = torch.matmul(attn, V)  # (B*nW, H, ws*ws, d_k)

        # 拼接多头
        O_W = O_W.permute(0, 2, 1, 3).contiguous().view(B * nW_total, ws * ws, self.d_model)

        # 8. 窗口合并回空间维度
        O_W = O_W.permute(0, 2, 1).view(B, nH, nW, self.d_model, ws, ws)
        O_W = O_W.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, self.d_model, H_pad, W_pad)

        # 去除padding
        if pad_h > 0 or pad_w > 0:
            O_W = O_W[:, :, :H, :W]

        # 1x1卷积投影回原通道数
        F_out = self.out_conv(O_W)  # (B, C, H, W)
        return F_out


class WaveletFusion(nn.Module):
    """小波融合模块"""

    def __init__(self, in_channels, wavelet='haar'):
        super().__init__()
        self.wavelet = wavelet
        self.weight = nn.Parameter(torch.randn(2, in_channels))

    def dwt2d(self, x):
        """2D离散小波变换"""
        B, C, H, W = x.shape
        dtype = x.dtype  # 记录原始dtype
        x_np = x.float().detach().cpu().numpy()  # 转float32给numpy

        coeffs_list = []
        for b in range(B):
            batch_coeffs = []
            for c in range(C):
                coeffs = pywt.dwt2(x_np[b, c], self.wavelet)
                cA, (cH, cV, cD) = coeffs
                h, w = cA.shape
                combined = np.zeros((h * 2, w * 2), dtype=np.float32)
                combined[:h, :w] = cA
                combined[:h, w:] = cH
                combined[h:, :w] = cV
                combined[h:, w:] = cD
                batch_coeffs.append(combined)
            coeffs_list.append(np.stack(batch_coeffs))

        result = torch.from_numpy(np.stack(coeffs_list)).to(device=x.device, dtype=dtype)  # 恢复原始dtype
        return result

    def idwt2d(self, x):
        """2D逆小波变换"""
        B, C, H, W = x.shape
        dtype = x.dtype  # 记录原始dtype
        x_np = x.float().detach().cpu().numpy()  # 转float32给numpy

        recon_list = []
        for b in range(B):
            batch_recon = []
            for c in range(C):
                h, w = H // 2, W // 2
                cA = x_np[b, c, :h, :w]
                cH = x_np[b, c, :h, w:]
                cV = x_np[b, c, h:, :w]
                cD = x_np[b, c, h:, w:]

                coeffs = (cA, (cH, cV, cD))
                recon = pywt.idwt2(coeffs, self.wavelet)
                batch_recon.append(recon.astype(np.float32))  # 确保float32
            recon_list.append(np.stack(batch_recon))

        result = torch.from_numpy(np.stack(recon_list)).to(device=x.device, dtype=dtype)  # 恢复原始dtype
        return result

    def forward(self, F, F_cam):
        """
        F: 原始输入 (B, C, H, W)
        F_cam: 通道注意力输出 (B, C, H, W)
        """
        B, C, H, W = F.shape

        F_dwt = self.dwt2d(F)
        F_wt = self.idwt2d(F_dwt)

        # 确保尺寸一致
        if F_wt.shape[-2:] != F.shape[-2:]:  # 修正了错误的 F.interpolate 写法
            F_wt = torch.nn.functional.interpolate(F_wt, size=(H, W), mode='bilinear', align_corners=False)

        return F_wt

    def fuse(self, F_wt, F_out):
        """
        加权融合F_wt和F_out
        F_wt: (B, C, H, W)
        F_out: (B, C, H, W)
        """
        w = torch.nn.functional.softmax(self.weight, dim=0)  # 修正了 F.softmax 命名冲突

        w0 = w[0].view(1, -1, 1, 1)
        w1 = w[1].view(1, -1, 1, 1)
        F_fused = w0 * F_wt + w1 * F_out
        return F_fused


class HybridAttention(nn.Module):
    """混合注意力机制：通道注意力 + 空间注意力 + 小波融合"""

    def __init__(self, in_channels, reduction=16, d_model=64, num_heads=8, window_size=8):
        super().__init__()
        self.channel_attn = ChannelAttention(in_channels, reduction)
        self.spatial_attn = WindowSpatialAttention(in_channels, d_model, num_heads, window_size)
        self.wavelet_fuse = WaveletFusion(in_channels)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        # 第一模块：通道注意力
        F_cam= self.channel_attn(x)  # (B, C, H, W), (B, C, 1, 1)

        # 第三模块准备：F经过小波变换
        F_wt = self.wavelet_fuse(x, F_cam)  # (B, C, H, W)

        # F_cam经过小波变换后送入空间注意力
        F_cam_dwt = self.wavelet_fuse.dwt2d(F_cam)

        # 第二模块：空间注意力
        F_out = self.spatial_attn(F_cam_dwt)  # (B, C, H, W), (B, 1, H, W)

        # 第三模块：加权融合
        output = self.wavelet_fuse.fuse(F_wt, F_out)  # (B, C, H, W)

        return output


def visualize_attention(image_path, model, save_path='attention_visualization.png'):
    """
    可视化注意力机制热力图

    Args:
        image_path: 输入图片路径
        model: HybridAttention模型
        save_path: 保存路径
    """
    # 读取图片
    img = Image.open(image_path).convert('RGB')
    original_size = img.size

    # 预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0)  # (1, 3, 224, 224)

    # 通过卷积层将3通道转换为模型需要的通道数
    in_channels = model.channel_attn.conv1.in_channels
    conv_adapter = nn.Conv2d(3, in_channels, 1).eval()
    with torch.no_grad():
        x = conv_adapter(img_tensor)  # (1, C, 224, 224)

    # 前向传播
    model.eval()
    with torch.no_grad():
        output, attn_dict = model(x)

    # 提取注意力图
    channel_weights = attn_dict['channel_weights'].squeeze().cpu().numpy()  # (C,)
    spatial_attn = attn_dict['spatial_attn'].squeeze().cpu().numpy()  # (H, W)
    fusion_weights = attn_dict['fusion_weights'].cpu().numpy()  # (2, C)

    # 创建可视化
    fig = plt.figure(figsize=(20, 10))

    # 1. 原始图片
    ax1 = plt.subplot(2, 4, 1)
    img_np = np.array(img.resize((224, 224)))
    ax1.imshow(img_np)
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # 2. 通道注意力权重分布
    ax2 = plt.subplot(2, 4, 2)
    ax2.bar(range(len(channel_weights)), channel_weights, color='steelblue', alpha=0.7)
    ax2.set_title('Channel Attention Weights', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Channel Index')
    ax2.set_ylabel('Weight Value')
    ax2.grid(True, alpha=0.3)

    # 3. 空间注意力热力图
    ax3 = plt.subplot(2, 4, 3)
    spatial_heatmap = cv2.resize(spatial_attn, (224, 224))
    im3 = ax3.imshow(spatial_heatmap, cmap='jet', alpha=0.8)
    ax3.set_title('Spatial Attention Heatmap', fontsize=14, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)

    # 4. 空间注意力叠加在原图上
    ax4 = plt.subplot(2, 4, 4)
    ax4.imshow(img_np)
    ax4.imshow(spatial_heatmap, cmap='jet', alpha=0.5)
    ax4.set_title('Spatial Attention Overlay', fontsize=14, fontweight='bold')
    ax4.axis('off')

    # 5. 融合权重对比
    ax5 = plt.subplot(2, 4, 5)
    fusion_w = fusion_weights.mean(axis=1)  # 对所有通道求平均
    ax5.bar(['Wavelet Branch', 'Attention Branch'], fusion_w, color=['coral', 'lightgreen'])
    ax5.set_title('Fusion Weights', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Average Weight')
    ax5.grid(True, alpha=0.3)

    # 6. 输出特征的均值热力图
    ax6 = plt.subplot(2, 4, 6)
    output_mean = output.mean(dim=1).squeeze().cpu().numpy()  # (H, W)
    output_heatmap = cv2.resize(output_mean, (224, 224))
    im6 = ax6.imshow(output_heatmap, cmap='viridis')
    ax6.set_title('Output Feature Heatmap', fontsize=14, fontweight='bold')
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, fraction=0.046)

    # 7. 通道注意力Top-10通道
    ax7 = plt.subplot(2, 4, 7)
    top_indices = np.argsort(channel_weights)[-10:]
    top_values = channel_weights[top_indices]
    ax7.barh(range(10), top_values, color='purple', alpha=0.7)
    ax7.set_yticks(range(10))
    ax7.set_yticklabels([f'Ch {i}' for i in top_indices])
    ax7.set_title('Top-10 Channel Weights', fontsize=14, fontweight='bold')
    ax7.set_xlabel('Weight Value')
    ax7.grid(True, alpha=0.3)

    # 8. 融合权重每个通道的分布
    ax8 = plt.subplot(2, 4, 8)
    x_pos = np.arange(min(20, fusion_weights.shape[1]))
    width = 0.35
    ax8.bar(x_pos - width / 2, fusion_weights[0, :len(x_pos)], width, label='Wavelet', alpha=0.7)
    ax8.bar(x_pos + width / 2, fusion_weights[1, :len(x_pos)], width, label='Attention', alpha=0.7)
    ax8.set_title('Fusion Weights per Channel (First 20)', fontsize=14, fontweight='bold')
    ax8.set_xlabel('Channel Index')
    ax8.set_ylabel('Weight Value')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"可视化结果已保存到: {save_path}")
    plt.show()

    # 打印统计信息
    print("\n=== 注意力机制统计信息 ===")
    print(f"通道注意力权重范围: [{channel_weights.min():.4f}, {channel_weights.max():.4f}]")
    print(f"通道注意力权重均值: {channel_weights.mean():.4f}")
    print(f"空间注意力范围: [{spatial_attn.min():.4f}, {spatial_attn.max():.4f}]")
    print(f"空间注意力均值: {spatial_attn.mean():.4f}")
    print(f"融合权重 - 小波分支: {fusion_w[0]:.4f}")
    print(f"融合权重 - 注意力分支: {fusion_w[1]:.4f}")


# 测试代码
# if __name__ == "__main__":
#     # 实例化模型
#     model = HybridAttention(
#         in_channels=128,
#         reduction=8,
#         d_model=32,
#         num_heads=4,
#         window_size=8
#     )
#
#     # 可视化注意力机制（请替换为你的图片路径）
#     image_path = "F:\\python\\YOLOv8-main\\test1.jpg"  # 替换为你的图片路径
#     visualize_attention(image_path, model, save_path='F:\\python\\YOLOv8-main\\predict')
