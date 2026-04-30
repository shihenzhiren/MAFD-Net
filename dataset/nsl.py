from __future__ import print_function

import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import torch
import torch.nn.functional as F

"""
mean = {
    'nsl': (0.0304, 0.0304, 0.0304),  # 根据实际数据集计算的均值
}

std = {
    'nsl': (0.1637, 0.1637, 0.1637),  # 根据实际数据集计算的方差
}
"""

# 定义全局常量
NSL_MEAN = (0.0304, 0.0304, 0.0304)  # 计算得到的NSL数据集均值
NSL_STD = (0.1637, 0.1637, 0.1637)   # 计算得到的NSL数据集方差

def get_data_folder():
    """
    return the path to store the data
    """
    data_folder = './data/nsl/'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder


class NSLBackCompat(datasets.ImageFolder):
    """
    NSLInstance+Sample Dataset
    """

    @property
    def train_labels(self):
        return self.targets

    @property
    def test_labels(self):
        return self.targets

    @property
    def train_data(self):
        return self.imgs

    @property
    def test_data(self):
        return self.imgs


class NSLInstance(NSLBackCompat):
    """NSLInstance Dataset.
    """

    def __getitem__(self, index):

        img, target = self.imgs[index][0], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

def get_nsl_dataloaders(batch_size=128, num_workers=8, is_instance=False):
    """
    NSL Dataset with separate train and test directories
    """
    data_folder = get_data_folder()
    train_folder = os.path.join(data_folder, 'train')
    test_folder = os.path.join(data_folder, 'test')

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(NSL_MEAN, NSL_STD),  # 使用计算得到的均值和方差
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(NSL_MEAN, NSL_STD),  # 使用计算得到的均值和方差
    ])

    if is_instance:
        train_set = NSLInstance(root=train_folder,
                                transform=train_transform)
        n_data = len(train_set)
    else:
        train_set = datasets.ImageFolder(root=train_folder,
                                         transform=train_transform)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = datasets.ImageFolder(root=test_folder,
                                    transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size / 2),
                             shuffle=False,
                             num_workers=int(num_workers / 2))

    if is_instance:
        return train_loader, test_loader, n_data
    else:
        return train_loader, test_loader

class NSLInstanceSample(datasets.ImageFolder):
    """NSL数据集的实例级采样版本
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
            sim_scores = self.class_similarity[target]  # [num_classes]
            # 排除自身类别
            sim_scores[target] = 0
            # 按相似度采样负样本类别
            neg_classes = torch.multinomial(sim_scores, self.k, replacement=True)
            # 从选中的类别中随机选择样本
            neg_idx = []
            for cls_idx in neg_classes:
                samples_in_class = self.class_indices[cls_idx.item()]
                neg_idx.append(np.random.choice(samples_in_class))
            neg_idx = np.array(neg_idx)
        else:
            # 原始的随机采样
            neg_idx = np.random.choice(self.class_indices[target], self.k, replace=True)
            
        sample_idx = np.hstack((np.asarray([index]), neg_idx))
        return image, target, index, sample_idx

def get_nsl_dataloaders_sample(batch_size=128, num_workers=8, k=4096,
                              mode='exact', is_sample=True, percent=1.0):
    """
    返回支持CRD的NSL数据加载器
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
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(NSL_MEAN, NSL_STD)  # 使用计算得到的均值和方差
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(NSL_MEAN, NSL_STD)  # 使用计算得到的均值和方差
    ])

    # 创建数据集
    train_set = NSLInstanceSample(
        root=os.path.join(data_folder, 'train'),
        transform=train_transform,
        k=k,
        mode=mode,
        is_sample=is_sample,
        percent=percent
    )
    n_data = len(train_set)

    val_set = NSLInstanceSample(
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

# 创建不带归一化的数据加载器
temp_transform = transforms.Compose([
    transforms.ToTensor(),
])
temp_dataset = datasets.ImageFolder(root='./data/nsl/train', transform=temp_transform)
temp_loader = DataLoader(temp_dataset, batch_size=64, num_workers=4, shuffle=False)

# 计算均值和方差
mean, std = compute_mean_std(temp_loader)
print(f"计算得到的均值: {mean}")
print(f"计算得到的方差: {std}")


