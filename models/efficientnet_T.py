import torch
import torch.nn as nn
import math
from functools import partial
import torch.nn.functional as F


__all__ = [f'efficientnet_b{i}_t' for i in range(8)]


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class MultiHeadAttention(nn.Module):
    """多头注意力机制（MHA）"""

    def __init__(self, in_channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        assert self.head_dim * num_heads == in_channels, "in_channels must be divisible by num_heads"

        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        batch_size, C, H, W = x.size()

        # 生成 query, key, value
        q = self.query(x).view(batch_size, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)  # (B, H, HW, D)
        k = self.key(x).view(batch_size, self.num_heads, self.head_dim, H * W)  # (B, H, D, HW)
        v = self.value(x).view(batch_size, self.num_heads, self.head_dim, H * W)  # (B, H, D, HW)

        # 计算注意力权重
        attention = F.softmax(torch.matmul(q, k), dim=-1)  # (B, H, HW, HW)

        # 加权求和
        out = torch.matmul(attention, v.permute(0, 1, 3, 2))  # (B, H, HW, D)
        out = out.permute(0, 1, 3, 2).contiguous().view(batch_size, C, H, W)  # Reshape to (B, C, H, W)
        out = self.out_conv(out)

        return self.gamma * out + x


class MBConvBlock(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, kernel_size=3,
                 replace_se=False):  # 新增replace_se参数控制替换
        super().__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and inp == oup

        hidden_dim = int(inp * expand_ratio)

        layers = []
        # Expansion phase
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(inp, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU()
            ])

        # Depthwise convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride,
                      (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            SiLU()
        ])

        # 仅当replace_se为True时替换为MHA
        if replace_se:
            layers.append(MultiHeadAttention(hidden_dim))

        # Projection phase
        layers.extend([
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            nn.BatchNorm2d(oup),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class EfficientNet_T(nn.Module):
    def __init__(self, phi_value, num_classes=1000, include_top=True,
                 replace_stages=[2, 3, 4]):  # 默认替换stage2-4的中间层
        super().__init__()
        self.include_top = include_top

        # 原始EfficientNet配置
        width_coeff = [1.0, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0]
        depth_coeff = [1.0, 1.1, 1.2, 1.4, 1.8, 2.2, 2.6, 3.1]
        dropout_rate = [0.2, 0.3, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5]
        last_channel = [1280, 1280, 1408, 1536, 1792, 2048, 2304, 2560]

        stage_configs = [
            [1, 16, 1, 1, 3],
            [6, 24, 2, 2, 3],
            [6, 40, 2, 2, 5],
            [6, 80, 3, 2, 3],
            [6, 112, 3, 1, 5],
            [6, 192, 4, 2, 5],
            [6, 320, 1, 1, 3]
        ]

        phi = phi_value
        width_mult = width_coeff[phi]
        depth_mult = depth_coeff[phi]
        self.dropout_rate = dropout_rate[phi]
        self.last_channel = _make_divisible(last_channel[phi] * width_mult, 8)

        # Stem
        out_channels = _make_divisible(32 * width_mult, 8)
        self.stem = nn.Sequential(
            nn.Conv2d(3, out_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            SiLU()
        )
        in_channels = out_channels

        # 构建中间层
        self.stages = nn.ModuleList()
        self.feature_channels = []

        for stage_idx, (t, c, n, s, k) in enumerate(stage_configs):
            output_channels = _make_divisible(c * width_mult, 8)
            layers = []
            for i in range(int(n * depth_mult)):
                stride = s if i == 0 else 1
                # 仅在指定阶段的中间层替换为MHA
                replace_se = (stage_idx in replace_stages) and (i == int(n * depth_mult) // 2)
                layers.append(
                    MBConvBlock(in_channels, output_channels,
                                stride, t, kernel_size=k,
                                replace_se=replace_se)
                )
                in_channels = output_channels
            self.stages.append(nn.Sequential(*layers))

            # 保持原始特征采集点
            if stage_idx in [0, 1, 2, 4, 6]:
                self.feature_channels.append(output_channels)

        self.feature_channels.append(self.last_channel)

        # Head
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, self.last_channel, 1, bias=False),
            nn.BatchNorm2d(self.last_channel),
            SiLU()
        )

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.dropout = nn.Dropout(self.dropout_rate)
            self.classifier = nn.Linear(self.last_channel, num_classes)

        self._initialize_weights()

    # 保持前向传播和特征采集点不变
    def forward(self, x, is_feat=False, preact=False):
        features = []
        x = self.stem(x)
        features.append(x)  # f0

        x = self.stages[0](x)
        x = self.stages[1](x)
        features.append(x)  # f1

        x = self.stages[2](x)
        features.append(x)  # f2

        x = self.stages[3](x)
        features.append(x)  # f3

        x = self.stages[4](x)
        x = self.stages[5](x)
        features.append(x)  # f4

        x = self.stages[6](x)
        x = self.head(x)
        features.append(x)  # f5

        if self.include_top:
            x = self.avgpool(x).flatten(1)
            x = self.dropout(x)
            x = self.classifier(x)

        if is_feat:
            return features, x
        else:
            return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


def _gen_efficientnet_t(phi, num_classes=1000, **kwargs):
    return EfficientNet_T(phi_value=phi, num_classes=num_classes, **kwargs)


# 标准模型定义
efficientnet_b0_t = partial(_gen_efficientnet_t, 0)
efficientnet_b1_t = partial(_gen_efficientnet_t, 1)
efficientnet_b2_t = partial(_gen_efficientnet_t, 2)
efficientnet_b3_t = partial(_gen_efficientnet_t, 3)
efficientnet_b4_t = partial(_gen_efficientnet_t, 4)
efficientnet_b5_t = partial(_gen_efficientnet_t, 5)
efficientnet_b6_t = partial(_gen_efficientnet_t, 6)
efficientnet_b7_t = partial(_gen_efficientnet_t, 7)

if __name__ == '__main__':
    # 测试CIFAR版本显存占用
    x = torch.randn(32, 3, 32, 32)
    net = efficientnet_b2_t(num_classes=100)

    feats, logit = net(x, is_feat=True)
    print("\nModified EfficientNet-T Feature Shapes:")
    for i, feat in enumerate(feats):
        print(f"f{i}: {feat.shape}")

    num_params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"Total parameters: {num_params:.2f}M")
