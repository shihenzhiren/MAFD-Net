import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class ChannelSelector(nn.Module):
    def __init__(self, in_channels, reduction=16, temperature=1.0):
        super(ChannelSelector, self).__init__()
        self.temperature = temperature
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 双路特征压缩
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
        )
        
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
        )
        
        # 特征融合与输出
        self.fc_fuse = nn.Sequential(
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # 平均池化与最大池化
        avg_out = self.fc1(self.avg_pool(x))
        max_out = self.fc2(self.max_pool(x))
        
        # 特征融合
        fuse = avg_out + max_out
        
        # 通道自适应权重
        channel_weights = self.fc_fuse(fuse)
        
        # 应用温度参数调整
        if self.temperature != 1.0:
            channel_weights = torch.pow(channel_weights, 1.0 / self.temperature)
            channel_weights = channel_weights / channel_weights.sum(dim=1, keepdim=True)
        
        # 加权特征
        return x * channel_weights, channel_weights

class DynamicFeatureSelector(nn.Module):
    def __init__(self, t_dims, s_dims, select_layers=5):
        super(DynamicFeatureSelector, self).__init__()
        self.t_dims = t_dims  # 教师模型各层输出维度
        self.s_dims = s_dims  # 学生模型各层输出维度
        self.n_t_feats = len(t_dims)
        self.n_s_feats = len(s_dims)
        
        # 构建层选择机制，为每个教师层动态选择最匹配的学生层
        self.layer_adaptation = nn.ModuleList()
        for t_idx in range(self.n_t_feats):
            t_dim = t_dims[t_idx][1]  # 通道维度
            t_h, t_w = t_dims[t_idx][2], t_dims[t_idx][3]  # 空间维度
            
            # 为每个教师层构建与所有学生层的映射网络
            layer_adapters = nn.ModuleList()
            for s_idx in range(self.n_s_feats):
                s_dim = s_dims[s_idx][1]  # 学生层通道维度
                # 构建简单的映射网络，包含通道适配和空间适配
                adapter = nn.Sequential(
                    nn.Conv2d(s_dim, t_dim, 1, bias=False),
                    nn.BatchNorm2d(t_dim),
                    nn.AdaptiveAvgPool2d((t_h, t_w))  # 添加空间适配层
                )
                layer_adapters.append(adapter)
            
            self.layer_adaptation.append(layer_adapters)
        
        # 层选择参数（可学习的）
        self.layer_weights = nn.Parameter(torch.ones(self.n_t_feats, self.n_s_feats))
        self.temperature = 0.5  # 用于softmax的温度参数
        self.select_layers = select_layers
    
    def forward(self, t_feats, s_feats):
        # 添加维度断言
        assert len(t_feats) == self.n_t_feats, f"预期教师特征数为{self.n_t_feats}，实际为{len(t_feats)}"
        assert len(s_feats) == self.n_s_feats, f"预期学生特征数为{self.n_s_feats}，实际为{len(s_feats)}"
        
        # 计算层间的匹配概率
        attn_weights = F.softmax(self.layer_weights / self.temperature, dim=1)
        
        # 为每个教师层选择最匹配的学生层
        adapted_features = []
        selected_pairs = []
        
        # 选择概率最高的匹配对
        for t_idx in range(self.n_t_feats):
            # 获取目标教师特征的形状
            t_feat = t_feats[t_idx]
            t_bs, t_c, t_h, t_w = t_feat.shape
            
            # 获取当前教师层的所有匹配权重
            weights = attn_weights[t_idx]
            
            # 选择前k个权重最高的学生层
            topk_weights, topk_indices = torch.topk(weights, k=min(self.select_layers, self.n_s_feats))
            
            # 构建适配特征
            adapted_feat = None
            for i, s_idx in enumerate(topk_indices):
                s_feat = s_feats[s_idx]
                weight = topk_weights[i]
                # 应用适配器
                adapted = self.layer_adaptation[t_idx][s_idx](s_feat)
                
                # 确保所有适配的特征具有相同的形状
                if adapted_feat is None:
                    adapted_feat = weight * adapted
                else:
                    adapted_feat = adapted_feat + weight * adapted
                
                # 记录选择的层对
                selected_pairs.append((t_idx, s_idx.item(), weight.item()))
            
            adapted_features.append(adapted_feat)
            
        return adapted_features, selected_pairs

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
    def __init__(self, s_shapes, t_shapes, qk_dim=128, use_channel_selection=True, use_dynamic_selection=True):
        super(AFD, self).__init__()
        self.s_shapes = s_shapes
        self.t_shapes = t_shapes
        self.qk_dim = qk_dim
        self.use_channel_selection = use_channel_selection
        self.use_dynamic_selection = use_dynamic_selection
        
        # 创建通道选择器
        if use_channel_selection:
            self.channel_selectors_s = nn.ModuleList([
                ChannelSelector(shape[1]) for shape in s_shapes
            ])
            self.channel_selectors_t = nn.ModuleList([
                ChannelSelector(shape[1]) for shape in t_shapes
            ])
        
        # 创建动态特征选择器
        if use_dynamic_selection:
            self.feature_selector = DynamicFeatureSelector(t_shapes, s_shapes)
        
        # 多尺度注意力机制
        self.attention = MultiScaleAttention(qk_dim)

    def forward(self, g_s, g_t):
        # 获取特征和权重信息
        s_feats = g_s
        t_feats = g_t
        
        # 应用通道选择器
        s_weights = []
        t_weights = []
        
        if self.use_channel_selection:
            for i, feat in enumerate(s_feats):
                s_feats[i], weight = self.channel_selectors_s[i](feat)
                s_weights.append(weight)
            
            for i, feat in enumerate(t_feats):
                t_feats[i], weight = self.channel_selectors_t[i](feat)
                t_weights.append(weight)
        
        # 应用动态特征选择
        if self.use_dynamic_selection:
            adapted_s_feats, selected_pairs = self.feature_selector(t_feats, s_feats)
            
            # 确保特征列表长度匹配，使用0填充
            if len(adapted_s_feats) != len(t_feats):
                print(f"警告：适配后学生特征数量({len(adapted_s_feats)})与教师特征数量({len(t_feats)})不匹配")
                # 如果适配后的特征数量比教师特征少，添加空特征
                if len(adapted_s_feats) < len(t_feats):
                    for i in range(len(adapted_s_feats), len(t_feats)):
                        # 创建与教师特征相同形状的零张量
                        zero_feat = torch.zeros_like(t_feats[i])
                        adapted_s_feats.append(zero_feat)
                # 如果适配后的特征数量比教师特征多，截断
                else:
                    adapted_s_feats = adapted_s_feats[:len(t_feats)]
            
            # 使用适配后的学生特征与教师特征进行注意力蒸馏
            loss = self.attention(adapted_s_feats, t_feats)
        else:
            # 直接使用原始特征进行注意力蒸馏
            # 确保特征列表长度匹配
            min_len = min(len(s_feats), len(t_feats))
            loss = self.attention(s_feats[:min_len], t_feats[:min_len])
        
        return loss

class MultiScaleAttention(nn.Module):
    def __init__(self, qk_dim=128):
        super(MultiScaleAttention, self).__init__()
        self.qk_dim = qk_dim
        self.projs_dict = nn.ModuleDict()  # 用于存储不同尺寸的投影层
        self.adapters = nn.ModuleDict()    # 用于存储不同通道数的适配器
        
    def _get_proj(self, projs_dict, in_channels):
        """获取或创建特征投影层"""
        key = str(in_channels)
        if key not in projs_dict:
            projs_dict[key] = nn.Conv2d(in_channels, self.qk_dim, 1)
        return projs_dict[key]
    
    def _get_adapter(self, s_c, t_c):
        """获取或创建通道适配器"""
        key = f"{s_c}_{t_c}"
        if key not in self.adapters:
            self.adapters[key] = nn.Conv2d(s_c, t_c, 1)
        return self.adapters[key]
    
    def forward(self, s_feats, t_feats):
        total_loss = 0
        
        # 确保有相同数量的特征
        assert len(s_feats) == len(t_feats), f"学生特征数量({len(s_feats)})与教师特征数量({len(t_feats)})不匹配"
        
        for t_idx, t_feat in enumerate(t_feats):
            # 获取相应的学生特征
            s_feat = s_feats[t_idx]
            
            # 获取形状信息
            t_bs, t_c, t_h, t_w = t_feat.shape
            s_bs, s_c, s_h, s_w = s_feat.shape
            
            # 空间尺寸自适应
            if s_h != t_h or s_w != t_w:
                s_feat = F.adaptive_avg_pool2d(s_feat, (t_h, t_w))
            
            # 通道适配
            if s_c != t_c:
                s_feat = self._get_adapter(s_c, t_c)(s_feat)
            
            # 计算特征差异
            loss = torch.mean((s_feat - t_feat) ** 2)
            
            # 添加到总损失
            total_loss += loss
        
        return total_loss / len(t_feats)

class Attention(nn.Module):
    def __init__(self, s_shapes, t_shapes, qk_dim=128):
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
        return sum(loss)

    def cal_diff(self, v_s, v_t, att):
        diff = (v_s - v_t.unsqueeze(1)).pow(2).mean(2)
        diff = torch.mul(diff, att).sum(1).mean()
        return diff

class LinearTransformTeacher(nn.Module):
    def __init__(self, t_shapes, qk_dim):
        super(LinearTransformTeacher, self).__init__()
        self.query_layer = nn.ModuleList([nn_bn_relu(t_shape[1], qk_dim) for t_shape in t_shapes])
        
        # 添加辅助模块来处理空间维度
        self.adapt_dims = nn.ModuleList()
        for t_shape in t_shapes:
            h, w = t_shape[2], t_shape[3]
            # 根据特征图大小决定采用不同的池化策略
            if h > 16 or w > 16:
                pool = nn.AdaptiveAvgPool2d((16, 16))
            else:
                pool = nn.Identity()
            self.adapt_dims.append(pool)

    def forward(self, g_t):
        bs = g_t[0].size(0)
        
        # 优化的特征提取
        channel_mean = []
        spatial_mean = []
        
        for i, f_t in enumerate(g_t):
            # 应用空间适配
            f_t_adapted = self.adapt_dims[i](f_t)
            
            # 通道统计特征
            cm = f_t_adapted.mean(3).mean(2)
            channel_mean.append(cm)
            
            # 空间统计特征
            sm = f_t_adapted.pow(2).mean(1).view(bs, -1)
            spatial_mean.append(sm)
        
        # 构建查询向量
        query = torch.stack([query_layer(f_t, relu=False) for f_t, query_layer in zip(channel_mean, self.query_layer)],
                          dim=1)
        
        # 归一化处理
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
        
        # 辅助模块处理空间维度
        self.adapt_dims = nn.ModuleList()
        for s_shape in s_shapes:
            h, w = s_shape[2], s_shape[3]
            if h > 16 or w > 16:
                pool = nn.AdaptiveAvgPool2d((16, 16))
            else:
                pool = nn.Identity()
            self.adapt_dims.append(pool)
        
        self.samplers = nn.ModuleList([Sample(t_shape) for t_shape in unique_t_shapes])
        self.key_layer = nn.ModuleList([nn_bn_relu(s_shape[1], qk_dim) for s_shape in s_shapes])
        self.bilinear = nn_bn_relu(qk_dim, qk_dim * len(t_shapes))

    def forward(self, g_s_list):
        bs = g_s_list[0].size(0)
        
        # 优化的特征提取与处理
        channel_mean = []
        
        for i, f_s in enumerate(g_s_list):
            # 应用空间适配
            f_s_adapted = self.adapt_dims[i](f_s)
            
            # 计算通道平均
            cm = f_s_adapted.mean(3).mean(2)
            channel_mean.append(cm)
        
        # 计算空间特征
        spatial_mean = [sampler(g_s_list, bs) for sampler in self.samplers]
        
        # 构建键向量
        key = torch.stack([key_layer(f_s) for key_layer, f_s in zip(self.key_layer, channel_mean)],
                        dim=1).view(bs * self.s, -1)  # Bs x h
        
        # 双线性变换
        bilinear_key = self.bilinear(key, relu=False).view(bs, self.s, self.t, -1)
        
        # 归一化处理
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