"""
EfficientNet implementation with aligned intermediate feature extraction
"""

import torch
import torch.nn as nn
import math
from functools import partial

__all__ = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
           'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7']

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

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, _make_divisible(channel // reduction, 8)),
            SiLU(),
            nn.Linear(_make_divisible(channel // reduction, 8), channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class MBConvBlock(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, kernel_size=3, se_ratio=0.25):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and inp == oup
        self.use_se = se_ratio is not None and se_ratio > 0

        hidden_dim = int(inp * expand_ratio)
        reduced_dim = max(1, int(inp * se_ratio)) if self.use_se else 0

        layers = []
        # expand
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(inp, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU()
            ])
        # depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride,
                      (kernel_size-1)//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            SiLU()
        ])
        # se
        if self.use_se:
            layers.append(SELayer(hidden_dim, reduction=reduced_dim))
        # project
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

class EfficientNet(nn.Module):
    def __init__(self, phi_value, num_classes=1000, include_top=True):
        super(EfficientNet, self).__init__()
        self.include_top = include_top
        width_coefficient = [1.0, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0]
        depth_coefficient = [1.0, 1.1, 1.2, 1.4, 1.8, 2.2, 2.6, 3.1]
        dropout_rate = [0.2, 0.3, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5]
        last_channel = [1280, 1280, 1408, 1536, 1792, 2048, 2304, 2560]
        stage_configs = [
            # t  c  n  s  k
            [1, 16, 1, 1, 3],  # stage1
            [6, 24, 2, 2, 3],  # stage2
            [6, 40, 2, 2, 5],  # stage3
            [6, 80, 3, 2, 3],  # stage4
            [6, 112, 3, 1, 5], # stage5
            [6, 192, 4, 2, 5], # stage6
            [6, 320, 1, 1, 3]  # stage7
        ]

        phi = phi_value
        width_mult = width_coefficient[phi]
        depth_mult = depth_coefficient[phi]
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

        # Build MBConv blocks with feature collection points
        self.stages = nn.ModuleList()
        self.feature_indices = [0, 1, 2, 4, 6, 7]  # Align with MobileNetV2's 6 features
        current_stage = 0
        self.feature_channels = []

        # Store feature channels for each collection point
        for idx, (t, c, n, s, k) in enumerate(stage_configs):
            output_channels = _make_divisible(c * width_mult, 8)
            if idx in [0, 1, 2, 4, 6]:  # Feature collection stages
                self.feature_channels.append(output_channels)
            layers = []
            for i in range(int(n * depth_mult)):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                layers.append(MBConvBlock(in_channels, output_channels,
                                        stride, t, kernel_size=k))
                in_channels = output_channels
            self.stages.append(nn.Sequential(*layers))
            current_stage += 1

        # Final feature channel
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
        # Stem
        x = self.stem(x)
        features.append(x)  # f0

        # Stage 1-2 (feature 1)
        x = self.stages[0](x)
        x = self.stages[1](x)
        features.append(x)  # f1

        # Stage 3 (feature 2)
        x = self.stages[2](x)
        features.append(x)  # f2

        # Stage 4 (feature 3)
        x = self.stages[3](x)
        features.append(x)  # f3

        # Stage 5-6 (feature 4)
        x = self.stages[4](x)
        x = self.stages[5](x)
        features.append(x)  # f4

        # Final stages and head (feature 5)
        x = self.stages[6](x)
        x = self.head(x)
        features.append(x)  # f5

        if self.include_top:
            x = self.avgpool(x).flatten(1)  # Flatten for classifier
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

def _efficientnet(phi, num_classes):
    return EfficientNet(phi_value=phi, num_classes=num_classes)

# Model definitions
def efficientnet_b0(num_classes=1000): return _efficientnet(0, num_classes)
def efficientnet_b1(num_classes=1000): return _efficientnet(1, num_classes)
def efficientnet_b2(num_classes=1000): return _efficientnet(2, num_classes)
def efficientnet_b3(num_classes=1000): return _efficientnet(3, num_classes)
def efficientnet_b4(num_classes=1000): return _efficientnet(4, num_classes)
def efficientnet_b5(num_classes=1000): return _efficientnet(5, num_classes)
def efficientnet_b6(num_classes=1000): return _efficientnet(6, num_classes)
def efficientnet_b7(num_classes=1000): return _efficientnet(7, num_classes)



if __name__ == '__main__':
    x = torch.randn(2, 3, 224, 224)
    net = efficientnet_b0(num_classes=1000)

    feats, logit = net(x, is_feat=True)
    print("\nFeature shapes:")
    for i, feat in enumerate(feats):
        print(f"f{i}: {feat.shape}")
    print("Output shape:", logit.shape)

    num_params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"Total parameters: {num_params:.2f}M")