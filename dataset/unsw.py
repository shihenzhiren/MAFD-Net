from __future__ import print_function
import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import torch
import torch.nn.functional as F

# UNSW-NB15 数据集统计参数（16x16 图像）
# 可通过 compute_mean_std() 重新计算
UNSW_MEAN = (0.5, 0.5, 0.5)
UNSW_STD  = (0.5, 0.5, 0.5)

# 数据集目录结构
#   data/unsw/train_multi/{0.0, 1.0, ..., 9.0}/image_*.png
#   data/unsw/test_multi/ {0.0, 1.0, ..., 9.0}/image_*.png
UNSW_TRAIN_DIR = 'data/unsw/train_multi'
UNSW_TEST_DIR  = 'data/unsw/test_multi'

# 训练集大小（用于 CRD）
UNSW_N_DATA = 175341


def get_data_folder():
    data_folder = './data/unsw/'
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)
    return data_folder


# ---------------------------------------------------------------------------
# 基础数据集类（返回 (img, target, index)）
# ---------------------------------------------------------------------------

class UNSWBackCompat(datasets.ImageFolder):
    """返回 (img, target, index)，方便蒸馏训练中统一处理数据格式。"""

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index


class UNSWInstance(UNSWBackCompat):
    """直接继承 UNSWBackCompat，保持接口一致。"""
    pass


# ---------------------------------------------------------------------------
# 普通数据加载器（不含 CRD 采样）
# ---------------------------------------------------------------------------

def get_unsw_dataloaders(batch_size=64, num_workers=8):
    """
    返回 UNSW 数据集的 train_loader 和 val_loader。

    数据已是 16×16 PNG 图像；此处 Resize 到 64×64 以兼容
    EfficientNet/ResNet 等大模型。
    """
    mean, std = UNSW_MEAN, UNSW_STD

    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_dataset = UNSWBackCompat(
        root=UNSW_TRAIN_DIR,
        transform=train_transform,
    )
    val_dataset = UNSWBackCompat(
        root=UNSW_TEST_DIR,
        transform=val_transform,
    )

    print(f"UNSW 训练集大小: {len(train_dataset)}")
    print(f"UNSW 验证集大小: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# CRD 采样数据集类
# ---------------------------------------------------------------------------

class UNSWInstanceSample(datasets.ImageFolder):
    """
    UNSW 数据集的实例级采样版本，支持 CRD 对比学习。

    __getitem__ 返回 (image, target, index, sample_idx)，
    其中 sample_idx 包含当前样本索引 + k 个负样本索引。
    """

    def __init__(self, root, transform=None, target_transform=None,
                 k=4096, mode='exact', is_sample=True, percent=1.0):
        super().__init__(root=root, transform=transform,
                         target_transform=target_transform)

        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        # 教师软标签驱动的类别相似度矩阵（可选）
        self.class_similarity = None  # [num_classes, num_classes]

        # 构建类别 → 样本索引映射
        self.class_indices = {}
        for i, (_, label) in enumerate(self.samples):
            self.class_indices.setdefault(label, []).append(i)

    def update_class_similarity(self, teacher_outputs):
        """用教师 logits 更新类别间相似度矩阵（可选调用）。"""
        with torch.no_grad():
            probs = F.softmax(teacher_outputs, dim=1)
            self.class_similarity = torch.matmul(probs.t(), probs)

    def __getitem__(self, index):
        if not self.is_sample:
            # 不采样时退化为普通格式 (img, target, index)
            img, target = datasets.ImageFolder.__getitem__(self, index)
            return img, target, index

        path, target = self.samples[index]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # ---------- 负样本采样 ----------
        if self.class_similarity is not None:
            sim_scores = self.class_similarity[target].clone()
            sim_scores[target] = 0.0
            if sim_scores.sum() == 0:
                sim_scores = torch.ones_like(sim_scores)
                sim_scores[target] = 0.0
            try:
                neg_classes = torch.multinomial(
                    sim_scores,
                    min(self.k, len(self.class_indices) - 1),
                    replacement=True,
                )
            except Exception:
                valid_classes = [c for c in range(len(sim_scores)) if c != target]
                neg_classes = torch.tensor(
                    np.random.choice(valid_classes,
                                     min(self.k, len(valid_classes)),
                                     replace=True)
                )

            neg_idx = []
            for cls_idx in neg_classes:
                c = cls_idx.item()
                if c in self.class_indices and len(self.class_indices[c]) > 0:
                    neg_idx.append(np.random.choice(self.class_indices[c]))
                else:
                    neg_idx.append(np.random.randint(0, len(self.samples)))

            while len(neg_idx) < self.k:
                neg_idx.append(np.random.randint(0, len(self.samples)))
            neg_idx = np.array(neg_idx)
        else:
            # 均匀随机采样
            neg_idx = np.random.randint(0, len(self.samples), size=self.k)

        sample_idx = np.hstack((np.asarray([index]), neg_idx))
        return image, target, index, sample_idx


# ---------------------------------------------------------------------------
# CRD 数据加载器
# ---------------------------------------------------------------------------

def get_unsw_dataloaders_sample(batch_size=128, num_workers=8, k=4096,
                                mode='exact', is_sample=True, percent=1.0):
    """
    返回支持 CRD 的 UNSW 数据加载器。

    Returns:
        train_loader, val_loader, n_data
    """
    mean, std = UNSW_MEAN, UNSW_STD

    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = UNSWInstanceSample(
        root=UNSW_TRAIN_DIR,
        transform=train_transform,
        k=k,
        mode=mode,
        is_sample=is_sample,
        percent=percent,
    )
    n_data = len(train_set)

    val_set = UNSWInstanceSample(
        root=UNSW_TEST_DIR,
        transform=test_transform,
        k=k,
        mode=mode,
        is_sample=is_sample,
        percent=percent,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size // 2,
        shuffle=False,
        num_workers=num_workers // 2,
    )

    return train_loader, val_loader, n_data


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def compute_mean_std(dataloader):
    """计算数据集的逐通道均值和标准差。"""
    mean = torch.zeros(3)
    std  = torch.zeros(3)
    nb_samples = 0

    for data, *_ in dataloader:
        bs = data.size(0)
        data = data.view(bs, data.size(1), -1)   # (B, C, H*W)
        mean += data.mean(2).sum(0)
        std  += data.std(2).sum(0)
        nb_samples += bs

    mean /= nb_samples
    std  /= nb_samples
    return mean, std
