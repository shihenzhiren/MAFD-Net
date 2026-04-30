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
    'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
    'cifar100': (0.2675, 0.2565, 0.2761),
}
"""


def get_data_folder():
    """
    return the path to store the data
    """
    data_folder = './data/'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder

class CIFAR100BackCompat(datasets.CIFAR100):
    """
    CIFAR100Instance+Sample Dataset
    """

    @property
    def train_labels(self):
        return self.targets

    @property
    def test_labels(self):
        return self.targets

    @property
    def train_data(self):
        return self.data

    @property
    def test_data(self):
        return self.data

class CIFAR100Instance(CIFAR100BackCompat):
    """CIFAR100Instance Dataset.
    """
    def __getitem__(self, index):
        
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


def get_cifar100_dataloaders(batch_size=128, num_workers=8, is_instance=False):
    """
    cifar 100
    """
    data_folder = get_data_folder()

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    if is_instance:
        train_set = CIFAR100Instance(root=data_folder,
                                     download=True,
                                     train=True,
                                     transform=train_transform)
        n_data = len(train_set)
    else:
        train_set = datasets.CIFAR100(root=data_folder,
                                      download=True,
                                      train=True,
                                      transform=train_transform)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = datasets.CIFAR100(root=data_folder,
                                 download=True,
                                 train=False,
                                 transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))

    if is_instance:
        return train_loader, test_loader, n_data
    else:
        return train_loader, test_loader


class CIFAR100InstanceSample(CIFAR100BackCompat):
    """
    CIFAR100Instance+Sample Dataset
    """
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, k=4096, mode='exact', is_sample=True, percent=1.0):
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)
        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        # 构建类别索引
        num_classes = 100
        num_samples = len(self.data)
        label = self.targets
        
        # 存储每个类别的样本索引
        self.class_indices = {}
        for i in range(num_samples):
            if label[i] not in self.class_indices:
                self.class_indices[label[i]] = []
            self.class_indices[label[i]].append(i)
            
        # 存储教师模型的类别相似度信息
        self.class_similarity = None  # [num_classes, num_classes]
        
        # 原有的正负样本索引构建（保持不变）
        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

    def update_class_similarity(self, teacher_outputs):
        """更新类别间的相似度矩阵"""
        with torch.no_grad():
            probs = F.softmax(teacher_outputs, dim=1)  # [N, num_classes]
            self.class_similarity = torch.matmul(probs.t(), probs)  # [num_classes, num_classes]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.is_sample:
            return img, target, index
        
        # 使用改进的采样策略
        if self.class_similarity is not None:
            # 根据类别相似度选择负样本
            sim_scores = self.class_similarity[target]  # [num_classes]
            sim_scores[target] = 0  # 排除自身类别
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
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            
        if self.mode == 'exact':
            pos_idx = index
        elif self.mode == 'relax':
            pos_idx = np.random.choice(self.cls_positive[target], 1)[0]
            
        sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
        return img, target, index, sample_idx

def get_cifar100_dataloaders_sample(batch_size=128, num_workers=8, k=4096, mode='exact',
                                    is_sample=True, percent=1.0):
    """
    cifar 100
    """
    data_folder = get_data_folder()

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_set = CIFAR100InstanceSample(root=data_folder,
                                       download=True,
                                       train=True,
                                       transform=train_transform,
                                       k=k,
                                       mode=mode,
                                       is_sample=is_sample,
                                       percent=percent)
    n_data = len(train_set)
    
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = datasets.CIFAR100(root=data_folder,
                                 download=True,
                                 train=False,
                                 transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))

    return train_loader, test_loader, n_data
