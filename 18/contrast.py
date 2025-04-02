"""
热误差预测模型训练脚本 (编码修复最终版)
功能：自动编码检测、日志记录、最佳模型保存、多配置比较
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
import torchvision.transforms as transforms
import pandas as pd
import argparse
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, List
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
import logging
from charset_normalizer import from_path  # 新增编码检测库

plt.switch_backend('Agg')

# -------------------- 日志配置 --------------------
def setup_logger(config_name: str) -> logging.Logger:
    """创建日志系统（覆盖模式）"""
    log_dir = os.path.join("logs", config_name)
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(config_name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # 文件handler（覆盖写入）
        file_handler = logging.FileHandler(
            os.path.join(log_dir, "training.log"),
            mode='w',
            encoding='utf-8'  # 明确指定日志文件编码
        )
        file_handler.setFormatter(formatter)

        # 控制台handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

# -------------------- 数据加载模块 --------------------
class ThermalDataset(Dataset):
    """增强型数据集加载器"""
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self._validate_paths()

    def _validate_paths(self):
        """预验证前10个样本路径"""
        sample_paths = self.df['image_path'].head(10).tolist()
        for path in sample_paths:
            if not os.path.exists(path):
                print(f"路径验证警告: {path} 不存在")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_path']
        label = self.df.iloc[idx]['target']

        try:
            # 统一路径处理
            img_path = os.path.normpath(img_path).replace('\\', '/')
            image = Image.open(img_path).convert('RGB')

            if self.transform:
                image = self.transform(image)

            return image, float(label)
        except Exception as e:
            print(f"加载失败: {img_path}, 错误: {str(e)}")
            return torch.zeros(3, 224, 224), 0.0

def prepare_data(root_dir: str) -> pd.DataFrame:
    """数据准备（含自动编码检测）"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = os.path.join(base_dir, 'data')
    data_root = os.path.normpath(data_root).replace('\\', '/')
    print(f"数据根目录: {data_root}")

    csv_files = []
    for dirpath, _, filenames in os.walk(data_root):
        if 'thermal_errors.csv' in filenames:
            csv_path = os.path.join(dirpath, 'thermal_errors.csv').replace('\\', '/')
            csv_files.append(csv_path)

    dfs = []
    for csv_file in csv_files:
        try:
            # 自动检测文件编码
            detection_result = from_path(csv_file).best()
            detected_encoding = detection_result.encoding if detection_result else 'utf-8'

            # 读取CSV时使用检测到的编码
            df = pd.read_csv(csv_file, encoding=detected_encoding)

            # 智能列检测
            img_col = next((c for c in df.columns if 'image' in c.lower()), 'Image')
            error_col = next((c for c in df.columns if 'error' in c.lower()), 'Thermal_Error')

            # 修复路径拼接
            csv_dir = os.path.dirname(csv_file)
            df['image_path'] = df[img_col].apply(
                lambda x: os.path.join(csv_dir, 'thermal_images', str(x))
                    .replace('\\', '/')
                    .replace('//', '/')
            )

            # 过滤无效路径
            df = df[df['image_path'].apply(os.path.exists)]
            if not df.empty:
                dfs.append(df.assign(target=df[error_col]))

        except Exception as e:
            print(f"处理 {csv_file} 出错: {str(e)}")
            continue

    if not dfs:
        raise ValueError("没有找到任何有效数据")

    full_df = pd.concat(dfs, ignore_index=True)

    # 数据集划分（8:2）
    train_df = full_df.sample(frac=0.8, random_state=42)
    val_df = full_df.drop(train_df.index)

    return pd.concat([
        train_df.assign(dataset='train'),
        val_df.assign(dataset='val')
    ])

# -------------------- 模型定义 --------------------
class ThermalErrorModel(nn.Module):
    """增强型预测模型"""
    def __init__(self):
        super().__init__()
        base_model = resnet18(weights=None)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])

        # 增强回归头
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.regressor(features).squeeze()

# -------------------- 训练引擎 --------------------
def get_data_loaders(df: pd.DataFrame, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """数据加载器工厂"""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return (
        DataLoader(
            ThermalDataset(df[df['dataset'] == 'train'], train_transform),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        ),
        DataLoader(
            ThermalDataset(df[df['dataset'] == 'val'], val_transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    )

def train(config: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    """最终优化训练流程"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"\n{'='*60} 训练初始化 {'='*60}")
    logger.info(f"配置名称: {config['name']}")
    logger.info(f"设备: {device} | 混合精度: {config.get('use_amp', False)}")

    # 创建模型保存目录
    model_dir = os.path.join("saved_models", config['name'])
    os.makedirs(model_dir, exist_ok=True)

    # 数据准备
    df = prepare_data(config['data_root'])
    train_loader, val_loader = get_data_loaders(df, config['batch_size'])

    # 模型初始化
    model = ThermalErrorModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)
    criterion = nn.SmoothL1Loss()
    scaler = GradScaler(enabled=config.get('use_amp', False))

    best_mae = float('inf')
    best_model_path = ""

    for epoch in range(config['epochs']):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"训练 Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            with autocast(enabled=config.get('use_amp', False)):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if config.get('use_amp', False):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * images.size(0)

        # 验证阶段
        model.eval()
        val_loss, predictions, targets = 0.0, [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                predictions.extend(outputs.cpu().numpy())
                targets.extend(labels.cpu().numpy())

        # 计算指标
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)
        mae = np.mean(np.abs(np.array(predictions) - np.array(targets)))

        # 保存最佳模型
        if mae < best_mae:
            # 删除旧的最佳模型
            if os.path.exists(best_model_path):
                os.remove(best_model_path)

            best_mae = mae
            best_model_path = os.path.join(model_dir, f"best_model_epoch{epoch+1}_mae{mae:.2f}.pth")
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"发现新最佳模型 ➔ 保存到: {best_model_path}")

        # 学习率调整
        scheduler.step(avg_val_loss)

        # 记录日志
        logger.info(f"\nEpoch {epoch+1}/{config['epochs']}")
        logger.info(f"训练损失: {avg_train_loss:.2f}nm | 验证损失: {avg_val_loss:.2f}nm")
        logger.info(f"当前MAE: {mae:.2f}nm | 最佳MAE: {best_mae:.2f}nm")
        logger.info("-" * 80)

    return {
        'config_name': config['name'],
        'best_mae': best_mae,
        'final_epoch': config['epochs'],
        'model_path': best_model_path
    }

# -------------------- 主程序 --------------------
def main():
    """主控制流程"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data').replace('\\', '/')

    # 参数配置组
    configs = [
        {
            'name': 'base',
            'data_root': data_path,
            'batch_size': 32,
            'lr': 1e-4,
            'epochs': 100,
            'weight_decay': 1e-4,
            'use_amp': False
        },
        {
            'name': 'large_batch',
            'data_root': data_path,
            'batch_size': 128,
            'lr': 4e-4,
            'epochs': 80,
            'weight_decay': 1e-4,
            'use_amp': True
        },
        {
            'name': 'low_lr',
            'data_root': data_path,
            'batch_size': 64,
            'lr': 5e-5,
            'epochs': 120,
            'weight_decay': 1e-4,
            'use_amp': False
        },
        {
            'name': 'amp_optimized',
            'data_root': data_path,
            'batch_size': 256,
            'lr': 2e-4,
            'epochs': 100,
            'weight_decay': 1e-4,
            'use_amp': True
        }
    ]

    parser = argparse.ArgumentParser(description='热误差模型训练平台')
    parser.add_argument('--config', nargs='+', choices=[c['name'] for c in configs],
                       default=['base'], help='选择训练配置')
    args = parser.parse_args()

    results = []
    for cfg in configs:
        if cfg['name'] in args.config:
            try:
                # 初始化日志
                logger = setup_logger(cfg['name'])

                # 执行训练
                result = train(cfg, logger)
                results.append(result)

                # 最终报告
                logger.info(f"\n{'='*60} 训练完成 {'='*60}")
                logger.info(f"最佳MAE: {result['best_mae']:.2f}nm")
                logger.info(f"模型保存路径: {result['model_path']}")
            except Exception as e:
                logger.error(f"训练异常终止: {str(e)}")
                continue

    # 生成最终报告
    if results:
        report = pd.DataFrame(results)
        report.to_csv("training_summary.csv", index=False)
        print("\n训练总结报告已生成: training_summary.csv")

if __name__ == "__main__":
    main()