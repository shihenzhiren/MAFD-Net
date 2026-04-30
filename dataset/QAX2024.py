from __future__ import print_function
import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import torch
import torch.nn.functional as F

# 定义全局常量
QAX2024_MEAN = (0.0304, 0.0304, 0.0304)
QAX2024_STD = (0.1637, 0.1637, 0.1637)

def get_data_folder():
    data_folder = './data/QAX2024/'
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)
    return data_folder

class QAX2024BackCompat(datasets.ImageFolder):
    """强制返回索引 (img, target, index)"""
    def __getitem__(self, index):
        img, target = super().__getitem__(index)  # 原始返回 (img, target)
        return img, target, index  # 添加索引

class QAX2024Instance(QAX2024BackCompat):
    """直接继承父类"""
    pass

def get_QAX2024_dataloaders(batch_size=64, num_workers=8):
    # 数据标准化参数
    mean = QAX2024_MEAN
    std = QAX2024_STD
    
    # 训练集转换
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # 验证集转换
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # 创建数据集（使用修正后的类）
    train_dataset = QAX2024BackCompat(
        root='data/QAX2024/train',
        transform=train_transform
    )
    val_dataset = QAX2024BackCompat(
        root='data/QAX2024/test',
        transform=val_transform
    )
    
    # 打印数据集大小
    print(f"QAX2024训练集大小: {len(train_dataset)}")
    print(f"QAX2024验证集大小: {len(val_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

# 其余代码（QAX2024InstanceSample等）保持不变
class QAX2024InstanceSample(datasets.ImageFolder):
    """QAX2024数据集的实例级采样版本
    支持CRD对比学习和基于教师知识的采样
    """
    def __init__(self, root, transform=None, target_transform=None,
                 k=4096, mode='exact', is_sample=True, percent=1.0):
        super().__init__(root=root, transform=transform, 
                        target_transform=target_transform)
        
        self.k = k
        self.mode = mode
        self.is_sample = is_sample
        
        # 存储教师模型的类别相似度信息
        self.class_similarity = None  # [num_classes, num_classes]
        
        # 构建类别索引
        self.class_indices = {}
        for i, (_, label) in enumerate(self.samples):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(i)
        
    def update_class_similarity(self, teacher_outputs):
        """更新类别间的相似度矩阵"""
        with torch.no_grad():
            probs = F.softmax(teacher_outputs, dim=1)  # [N, num_classes]
            self.class_similarity = torch.matmul(probs.t(), probs)  # [num_classes, num_classes]
            
    def __getitem__(self, index):
        if not self.is_sample:
            return super().__getitem__(index)
            
        path, target = self.samples[index]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
            
        if self.class_similarity is not None:
            # 根据类别相似度选择负样本
            sim_scores = self.class_similarity[target].clone()  # [num_classes]
            # 排除自身类别
            sim_scores[target] = 0
            # 确保有非零值用于采样
            if sim_scores.sum() == 0:
                sim_scores = torch.ones_like(sim_scores)
                sim_scores[target] = 0
            # 按相似度采样负样本类别
            try:
                neg_classes = torch.multinomial(sim_scores, min(self.k, len(self.class_indices)-1), replacement=True)
            except:
                # 出错时回退到均匀采样
                valid_classes = [c for c in range(len(sim_scores)) if c != target]
                neg_classes = torch.tensor(np.random.choice(valid_classes, min(self.k, len(valid_classes)), replace=True))
            
            # 从选中的类别中随机选择样本
            neg_idx = []
            for cls_idx in neg_classes:
                cls_idx_item = cls_idx.item()
                if cls_idx_item in self.class_indices and len(self.class_indices[cls_idx_item]) > 0:
                    neg_idx.append(np.random.choice(self.class_indices[cls_idx_item]))
                else:
                    # 如果该类别没有样本，从所有样本中随机选择
                    neg_idx.append(np.random.randint(0, len(self.samples)))
            
            # 确保负样本数量符合预期
            while len(neg_idx) < self.k:
                neg_idx.append(np.random.randint(0, len(self.samples)))
                
            neg_idx = np.array(neg_idx)
        else:
            # 原始的随机采样，确保不超出范围
            neg_idx = np.random.randint(
                0, 
                len(self.samples), 
                size=self.k
            )
            
        sample_idx = np.hstack((np.asarray([index]), neg_idx))
        return image, target, index, sample_idx

def get_QAX2024_dataloaders_sample(batch_size=128, num_workers=8, k=4096,
                              mode='exact', is_sample=True, percent=1.0):
    """
    返回支持CRD的QAX2024数据加载器
    Args:
        batch_size: 批次大小
        num_workers: 工作进程数
        k: 负样本数量
        mode: 采样模式 ('exact' 或 'relax')
        is_sample: 是否进行采样
        percent: 负样本比例
    Returns:
        train_loader: 训练集加载器
        val_loader: 验证集加载器
        n_data: 训练集大小
    """
    data_folder = get_data_folder()
    
    # 数据增强
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(QAX2024_MEAN, QAX2024_STD)
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(QAX2024_MEAN, QAX2024_STD)
    ])

    # 创建数据集
    train_set = QAX2024InstanceSample(
        root=os.path.join(data_folder, 'train'),
        transform=train_transform,
        k=k,
        mode=mode,
        is_sample=is_sample,
        percent=percent
    )
    n_data = len(train_set)

    val_set = QAX2024InstanceSample(
        root=os.path.join(data_folder, 'test'),
        transform=test_transform,
        k=k,
        mode=mode,
        is_sample=is_sample,
        percent=percent
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size//2,
        shuffle=False,
        num_workers=num_workers//2
    )

    return train_loader, val_loader, n_data

def compute_mean_std(dataloader):
    """计算数据集的均值和标准差"""
    mean = 0.
    std = 0.
    nb_samples = 0.
    
    for data, _ in dataloader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples
    
    mean /= nb_samples
    std /= nb_samples
    
    return mean, std

# 以下代码可以在需要计算数据集均值和方差时使用
"""
# 创建不带归一化的数据加载器
temp_transform = transforms.Compose([
    transforms.ToTensor(),
])
temp_dataset = datasets.ImageFolder(root='./data/QAX2024/train', transform=temp_transform)
temp_loader = DataLoader(temp_dataset, batch_size=64, num_workers=4, shuffle=False)

# 计算均值和方差
mean, std = compute_mean_std(temp_loader)
print(f"计算得到的均值: {mean}")
print(f"计算得到的方差: {std}")
""" 