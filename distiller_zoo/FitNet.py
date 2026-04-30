from __future__ import print_function

import torch
import torch.nn as nn


class HintLoss(nn.Module):
    """
    Fitnets: hints for thin deep nets, ICLR 2015
    自动适配学生模型和教师模型的维度
    """

    def __init__(self):
        super(HintLoss, self).__init__()
        self.crit = nn.MSELoss()  # 使用 MSE 损失

    def forward(self, f_s, f_t):
        """
        Args:
            f_s (torch.Tensor): 学生模型的特征图，形状为 [batch_size, ch_s, h, w]
            f_t (torch.Tensor): 教师模型的特征图，形状为 [batch_size, ch_t, h, w]
        Returns:
            torch.Tensor: 损失值
        """
        if f_s.shape[1] != f_t.shape[1]:
            # 如果通道数不一致，使用 1x1 卷积将学生模型的通道数调整到与教师模型一致
            self.conv = nn.Conv2d(f_s.shape[1], f_t.shape[1], kernel_size=1, stride=1, padding=0).to(f_s.device)
            f_s = self.conv(f_s)

        if f_s.shape[2:] != f_t.shape[2:]:
            # 如果空间维度不一致，使用插值将学生模型的特征图调整到与教师模型一致
            f_s = torch.nn.functional.interpolate(f_s, size=f_t.shape[2:], mode='bilinear', align_corners=False)

        # 计算 MSE 损失
        loss = self.crit(f_s, f_t)
        return loss
