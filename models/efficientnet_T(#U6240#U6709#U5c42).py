"""
Modified EfficientNet with Self-Attention (EfficientNet-T)
Replaced SE blocks with Self-Attention modules
"""

import torch
import torch.nn as nn
import math
from functools import partial

__all__ = ['efficientnet_b0_t', 'efficientnet_b1_t', 'efficientnet_b2_t',
           'efficientnet_b3_t', 'efficientnet_b4_t', 'efficientnet_b5_t',
           'efficientnet_b6_t', 'efficientnet_b7_t']


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


class SelfAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),
            SiLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1),
            nn.Sigmoid()
        )

        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        ca = self.channel_att(nn.functional.adaptive_avg_pool2d(x, 1))

        # Spatial attention
        sa = self.spatial_att(x)

        # Combined attention
        return x * ca * sa


class MBConvBlock(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, kernel_size=3, use_attention=True):
        super().__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and inp == oup
        self.use_attention = use_attention

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

        # Self-Attention
        if self.use_attention:
            layers.append(SelfAttention(hidden_dim))

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
    def __init__(self, phi_value, num_classes=1000, include_top=True):
        super().__init__()
        self.include_top = include_top
        width_coeff = [1.0, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0]
        depth_coeff = [1.0, 1.1, 1.2, 1.4, 1.8, 2.2, 2.6, 3.1]
        dropout_rate = [0.2, 0.3, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5]
        last_channel = [1280, 1280, 1408, 1536, 1792, 2048, 2304, 2560]

        stage_configs = [
            # t  c  n  s  k
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

        # Build MBConv blocks
        self.stages = nn.ModuleList()
        self.feature_channels = []

        for idx, (t, c, n, s, k) in enumerate(stage_configs):
            output_channels = _make_divisible(c * width_mult, 8)
            layers = []
            for i in range(int(n * depth_mult)):
                stride = s if i == 0 else 1
                use_att = (idx in [2, 3, 4, 5])  # Apply attention to middle stages
                layers.append(
                    MBConvBlock(in_channels, output_channels,
                                stride, t, kernel_size=k,
                                use_attention=use_att)
                )
                in_channels = output_channels
            self.stages.append(nn.Sequential(*layers))
            if idx in [0, 1, 2, 4, 6]:
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

    def get_feat_modules(self):
        feat_m = nn.ModuleList()
        feat_m.append(self.stem)
        feat_m.extend(self.stages)
        feat_m.append(self.head)
        return feat_m

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


def _efficientnet_t(phi, num_classes):
    return EfficientNet_T(phi_value=phi, num_classes=num_classes)


# Model definitions with '_t' suffix
def efficientnet_b0_t(num_classes=1000): return _efficientnet_t(0, num_classes)


def efficientnet_b1_t(num_classes=1000): return _efficientnet_t(1, num_classes)


def efficientnet_b2_t(num_classes=1000): return _efficientnet_t(2, num_classes)


def efficientnet_b3_t(num_classes=1000): return _efficientnet_t(3, num_classes)


def efficientnet_b4_t(num_classes=1000): return _efficientnet_t(4, num_classes)


def efficientnet_b5_t(num_classes=1000): return _efficientnet_t(5, num_classes)


def efficientnet_b6_t(num_classes=1000): return _efficientnet_t(6, num_classes)


def efficientnet_b7_t(num_classes=1000): return _efficientnet_t(7, num_classes)


if __name__ == '__main__':
    x = torch.randn(2, 3, 224, 224)
    net = efficientnet_b0_t(num_classes=1000)

    feats, logit = net(x, is_feat=True)
    print("\nModified EfficientNet-T Feature Shapes:")
    for i, feat in enumerate(feats):
        print(f"f{i}: {feat.shape}")
    print("Output shape:", logit.shape)

    num_params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"Total parameters: {num_params:.2f}M")
