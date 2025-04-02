# dataset.py
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np


class ThermalDataset(Dataset):
    """
    热误差预测数据集
    特征：热成像图片（1024x600 RGB）
    标签：热伸长误差（浮点数值）
    """

    def __init__(self, dataframe, transform=None, mode='train'):
        """
        参数:
            dataframe: 包含'image_path'和'error'列的DataFrame
            transform: 数据增强/预处理
            mode: 数据集模式 (train/val/test)
        """
        self.df = dataframe
        self.transform = transform
        self.mode = mode

        # 自适应填充尺寸 (保持1024x600原始比例)
        self.pad_size = (1024, 600)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_path']
        error = self.df.iloc[idx]['error']

        # 加载原始图像
        img = Image.open(img_path).convert('RGB')

        # 转换为numpy数组进行数值处理
        img_array = np.array(img)

        # 保持原始温度映射关系（关键步骤）
        # 由于颜色条已经编码温度范围，我们保持原始数值
        img = Image.fromarray(img_array)

        # 应用预处理
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(error, dtype=torch.float32)


def prepare_data(root_dir='data', test_size=0.2, random_state=42):
    """
    准备数据集DataFrame
    返回:
        DataFrame: 包含所有仿真轮次的数据路径和误差值
    """
    all_data = []

    # 遍历所有仿真轮次目录
    for sim_dir in os.listdir(root_dir):
        sim_path = os.path.join(root_dir, sim_dir)
        if not os.path.isdir(sim_path):
            continue

        # 读取当前轮次的CSV文件
        csv_path = os.path.join(sim_path, 'thermal_errors.csv')
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)

        # 添加完整图片路径
        df['image_path'] = df['Image'].apply(
            lambda x: os.path.join(sim_path, 'thermal_images', x))

        all_data.append(df)

    # 合并所有数据
    full_df = pd.concat(all_data, ignore_index=True)

    # 拆分训练集和测试集（保持工况分布）
    train_df, val_df = train_test_split(
        full_df,
        test_size=test_size,
        random_state=random_state,
        stratify=pd.cut(full_df['Thermal_Error(nm)'], bins=10)  # 分层抽样
    )

    return train_df, val_df


def get_data_loaders(train_df, val_df, batch_size=32):
    """
    获取数据加载器
    包含针对热成像特性的数据增强策略
    """
    # 训练集数据增强（保持温度数值有效性）
    train_transform = transforms.Compose([
        transforms.Pad((0, 0, 0, 0)),  # 保持原始尺寸1024x600
        transforms.RandomAffine(
            degrees=5,
            translate=(0.05, 0),
            scale=(0.95, 1.05)
        ),  # 小幅仿射变换
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0,
            hue=0
        ),  # 避免色相变化
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 验证集预处理
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集
    train_dataset = ThermalDataset(
        train_df,
        transform=train_transform,
        mode='train'
    )

    val_dataset = ThermalDataset(
        val_df,
        transform=val_transform,
        mode='val'
    )

    # 创建DataLoader（优化内存使用）
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader


# 数据验证测试
if __name__ == "__main__":
    # 示例使用
    train_df, val_df = prepare_data()
    print(f"训练样本数: {len(train_df)}, 验证样本数: {len(val_df)}")

    # 获取数据加载器
    train_loader, val_loader = get_data_loaders(train_df, val_df, batch_size=8)

    # 检查一个批次的数据
    for images, errors in train_loader:
        print("训练批次维度:")
        print(f"图像尺寸: {images.shape}")  # 应为 [batch, 3, 600, 1024]
        print(f"误差值范围: {errors.min().item():.2f} ~ {errors.max().item():.2f} nm")
        break