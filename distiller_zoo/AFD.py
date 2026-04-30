import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class nn_bn_relu(nn.Module):
    def __init__(self, nin, nout):
        super(nn_bn_relu, self).__init__()
        self.linear = nn.Linear(nin, nout)
        self.bn = nn.BatchNorm1d(nout)
        self.relu = nn.ReLU(False)

    def forward(self, x, relu=True):
        if relu:
            return self.relu(self.bn(self.linear(x)))
        return self.bn(self.linear(x))

class AFD(nn.Module):
    def __init__(self, s_shapes, t_shapes, qk_dim=128):
        super(AFD, self).__init__()
        self.attention = Attention(s_shapes, t_shapes, qk_dim)

    def forward(self, g_s, g_t):
        loss = self.attention(g_s, g_t)
        return sum(loss)

class Attention(nn.Module):
    def __init__(self, s_shapes, t_shapes, qk_dim):
        super(Attention, self).__init__()
        self.qk_dim = qk_dim
        # 计算唯一的教师特征图形状
        unique_t_shapes = []
        for shape in t_shapes:
            if shape not in unique_t_shapes:
                unique_t_shapes.append(shape)
        self.n_t = [t_shapes.count(shape) for shape in unique_t_shapes]

        self.linear_trans_s = LinearTransformStudent(s_shapes, t_shapes, qk_dim)
        self.linear_trans_t = LinearTransformTeacher(t_shapes, qk_dim)

        self.p_t = nn.Parameter(torch.Tensor(len(t_shapes), qk_dim))
        self.p_s = nn.Parameter(torch.Tensor(len(s_shapes), qk_dim))
        torch.nn.init.xavier_normal_(self.p_t)
        torch.nn.init.xavier_normal_(self.p_s)

    def forward(self, g_s, g_t):
        bilinear_key, h_hat_s_all = self.linear_trans_s(g_s)
        query, h_t_all = self.linear_trans_t(g_t)

        p_logit = torch.matmul(self.p_t, self.p_s.t())

        logit = torch.add(torch.einsum('bstq,btq->bts', bilinear_key, query), p_logit) / np.sqrt(self.qk_dim)
        atts = F.softmax(logit, dim=2)  # b x t x s
        loss = []

        for i, (n, h_t) in enumerate(zip(self.n_t, h_t_all)):
            h_hat_s = h_hat_s_all[n]
            diff = self.cal_diff(h_hat_s, h_t, atts[:, i])
            loss.append(diff)
        return loss

    def cal_diff(self, v_s, v_t, att):
        # 检查维度
        if v_s.shape[2] != v_t.shape[1]:
            # 将v_s重新采样到与v_t相同的空间维度
            bs, n_s, dim_s = v_s.shape
            _, dim_t = v_t.shape
            if dim_s > dim_t:
                # 降采样
                v_s_resized = v_s[:, :, :dim_t]
            else:
                # 上采样
                v_s_resized = torch.zeros(bs, n_s, dim_t, device=v_s.device)
                v_s_resized[:, :, :dim_s] = v_s
            diff = (v_s_resized - v_t.unsqueeze(1)).pow(2).mean(2)
        else:
            diff = (v_s - v_t.unsqueeze(1)).pow(2).mean(2)
        
        diff = torch.mul(diff, att).sum(1).mean()
        return diff

class LinearTransformTeacher(nn.Module):
    def __init__(self, t_shapes, qk_dim):
        super(LinearTransformTeacher, self).__init__()
        self.query_layer = nn.ModuleList([nn_bn_relu(t_shape[1], qk_dim) for t_shape in t_shapes])

    def forward(self, g_t):
        bs = g_t[0].size(0)
        channel_mean = [f_t.mean(3).mean(2) for f_t in g_t]
        spatial_mean = [f_t.pow(2).mean(1).view(bs, -1) for f_t in g_t]
        query = torch.stack([query_layer(f_t, relu=False) for f_t, query_layer in zip(channel_mean, self.query_layer)],
                            dim=1)
        value = [F.normalize(f_s, dim=1) for f_s in spatial_mean]
        return query, value

class LinearTransformStudent(nn.Module):
    def __init__(self, s_shapes, t_shapes, qk_dim):
        super(LinearTransformStudent, self).__init__()
        self.t = len(t_shapes)
        self.s = len(s_shapes)
        self.qk_dim = qk_dim
        self.relu = nn.ReLU(inplace=False)

        # 获取唯一的教师特征图形状
        unique_t_shapes = []
        for shape in t_shapes:
            if shape not in unique_t_shapes:
                unique_t_shapes.append(shape)
        
        self.samplers = nn.ModuleList([Sample(t_shape) for t_shape in unique_t_shapes])
        self.key_layer = nn.ModuleList([nn_bn_relu(s_shape[1], qk_dim) for s_shape in s_shapes])
        self.bilinear = nn_bn_relu(qk_dim, qk_dim * len(t_shapes))

    def forward(self, g_s):
        bs = g_s[0].size(0)
        channel_mean = [f_s.mean(3).mean(2) for f_s in g_s]
        spatial_mean = [sampler(g_s, bs) for sampler in self.samplers]

        key = torch.stack([key_layer(f_s) for key_layer, f_s in zip(self.key_layer, channel_mean)],
                                     dim=1).view(bs * self.s, -1)  # Bs x h
        bilinear_key = self.bilinear(key, relu=False).view(bs, self.s, self.t, -1)
        value = [F.normalize(s_m, dim=2) for s_m in spatial_mean]
        return bilinear_key, value

class Sample(nn.Module):
    def __init__(self, t_shape):
        super(Sample, self).__init__()
        t_N, t_C, t_H, t_W = t_shape
        self.sample = nn.AdaptiveAvgPool2d((t_H, t_W))

    def forward(self, g_s, bs):
        g_s = torch.stack([self.sample(f_s.pow(2).mean(1, keepdim=True)).view(bs, -1) for f_s in g_s], dim=1)
        return g_s