import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import argparse
import os
import pandas as pd
import numpy as np
import logging
from charset_normalizer import from_path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights


# -------------------- 日志配置 --------------------
def setup_logger(config_name: str) -> logging.Logger:
    """创建日志系统（覆盖模式）"""
    log_dir = os.path.join("logs", config_name)
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(config_name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(
            os.path.join(log_dir, "training.log"),
            mode='w',
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


# -------------------- 数据加载模块 --------------------
class ThermalDataset(Dataset):
    """增强型数据集加载器"""

    def __init__(self, df, transform=None, augment=False):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.augment = augment
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
            img_path = os.path.normpath(img_path).replace('\\', '/')
            image = Image.open(img_path).convert('RGB')

            if self.transform:
                image = self.transform(image)

            # 确保返回的是float32类型，而不是Python的float（在PyTorch中会被解释为double/float64）
            return image, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            print(f"加载失败: {img_path}, 错误: {str(e)}")
            # 返回零张量作为回退机制，确保标签也是float32类型
            return torch.zeros(3, 224, 224), torch.tensor(0.0, dtype=torch.float32)


def prepare_data(root_dir: str) -> pd.DataFrame:
    """数据准备（含自动编码检测）"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = os.path.join(base_dir, 'data')
    data_root = os.path.normpath(data_root).replace('\\', '/')

    csv_files = []
    for dirpath, _, filenames in os.walk(data_root):
        if 'thermal_errors.csv' in filenames:
            csv_path = os.path.join(dirpath, 'thermal_errors.csv').replace('\\', '/')
            csv_files.append(csv_path)

    dfs = []
    for csv_file in csv_files:
        try:
            detection_result = from_path(csv_file).best()
            detected_encoding = detection_result.encoding if detection_result else 'utf-8'
            df = pd.read_csv(csv_file, encoding=detected_encoding)

            img_col = next((c for c in df.columns if 'image' in c.lower()), 'Image')
            error_col = next((c for c in df.columns if 'error' in c.lower()), 'Thermal_Error')

            csv_dir = os.path.dirname(csv_file)
            df['image_path'] = df[img_col].apply(
                lambda x: os.path.join(csv_dir, 'thermal_images', str(x)).replace('\\', '/')
            )

            df = df[df['image_path'].apply(os.path.exists)]
            if not df.empty:
                dfs.append(df.assign(target=df[error_col]))
        except Exception as e:
            print(f"处理 {csv_file} 出错: {str(e)}")
            continue

    if not dfs:
        raise ValueError("没有找到任何有效数据")

    full_df = pd.concat(dfs, ignore_index=True)

    # 按照热误差值排序并分层抽样以确保分布一致性
    full_df = full_df.sort_values('target')
    indices = np.arange(len(full_df))
    train_indices = indices[::5]  # 每5个样本中的第1个作为验证集
    train_indices = np.concatenate([indices[:5][1:], train_indices])  # 确保不漏掉最开始的样本
    val_indices = np.setdiff1d(indices, train_indices)

    train_df = full_df.iloc[train_indices]
    val_df = full_df.iloc[val_indices]

    return pd.concat([
        train_df.assign(dataset='train'),
        val_df.assign(dataset='val')
    ])


def get_data_loaders(df: pd.DataFrame, batch_size: int, image_size: int = 224) -> tuple:
    """数据加载器工厂"""
    # 更强的数据增强
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.1)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return (
        DataLoader(
            ThermalDataset(df[df['dataset'] == 'train'], train_transform, augment=True),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        ),
        DataLoader(
            ThermalDataset(df[df['dataset'] == 'val'], val_transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    )


# -------------------- 高级注意力机制 --------------------
class ChannelAttention(nn.Module):
    """通道注意力机制"""

    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享MLP
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    """空间注意力机制"""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 沿着通道维度计算平均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # 拼接结果
        out = torch.cat([avg_out, max_out], dim=1)

        # 应用卷积和sigmoid
        out = self.conv(out)

        # 应用注意力
        return x * self.sigmoid(out)


class CBAM(nn.Module):
    """结合通道和空间注意力的CBAM模块"""

    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # 先应用通道注意力，再应用空间注意力
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# -------------------- 模型定义 --------------------
class ThermalErrorModel(nn.Module):
    """高精度热误差预测模型"""

    def __init__(self, img_size=224):
        super().__init__()
        # 使用torchvision中预训练的ResNet50模型作为骨干网络
        # 使用内置的预训练权重加载方式，不需要从外部下载
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # 提取各层特征
        self.layer0 = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool
        )
        self.layer1 = self.backbone.layer1  # 256 channels
        self.layer2 = self.backbone.layer2  # 512 channels
        self.layer3 = self.backbone.layer3  # 1024 channels
        self.layer4 = self.backbone.layer4  # 2048 channels

        # 添加注意力机制
        self.cbam1 = CBAM(256)
        self.cbam2 = CBAM(512)
        self.cbam3 = CBAM(1024)
        self.cbam4 = CBAM(2048)

        # 自适应特征融合
        total_channels = 256 + 512 + 1024 + 2048
        self.fusion = nn.Sequential(
            nn.Conv2d(total_channels, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            CBAM(512)
        )

        # 高级回归器
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

        # 辅助输出头 - 用于深度监督
        self.aux_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )

        # 初始化新增层的权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in [self.fusion, self.regressor, self.aux_head,
                  self.cbam1, self.cbam2, self.cbam3, self.cbam4]:
            if isinstance(m, nn.Sequential):
                for module in m:
                    if isinstance(module, (nn.Conv2d, nn.Linear)):
                        nn.init.kaiming_normal_(module.weight, mode='fan_out')
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)
                    elif isinstance(module, nn.BatchNorm2d):
                        nn.init.ones_(module.weight)
                        nn.init.zeros_(module.bias)
            elif isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # 多尺度特征提取
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # 应用注意力机制
        f1 = self.cbam1(x1)
        f2 = self.cbam2(x2)
        f3 = self.cbam3(x3)
        f4 = self.cbam4(x4)

        # 特征调整到相同的空间维度
        target_size = f4.shape[2:]  # 使用最小特征图的尺寸

        f1 = F.interpolate(f1, size=target_size, mode='bilinear', align_corners=False)
        f2 = F.interpolate(f2, size=target_size, mode='bilinear', align_corners=False)
        f3 = F.interpolate(f3, size=target_size, mode='bilinear', align_corners=False)

        # 特征融合
        fused = torch.cat([f1, f2, f3, f4], dim=1)
        fused = self.fusion(fused)

        # 主要预测
        main_out = self.regressor(fused).squeeze()

        # 训练时的辅助预测（深度监督）
        if self.training:
            aux_out = self.aux_head(f4).squeeze()
            return main_out, aux_out

        return main_out


# -------------------- 损失函数 --------------------
class CombinedLoss(nn.Module):
    """组合损失函数，支持多任务学习"""

    def __init__(self, alpha=0.6, beta=0.4, aux_weight=0.4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.aux_weight = aux_weight
        self.smooth_l1 = nn.SmoothL1Loss()
        self.mse = nn.MSELoss()

    def forward(self, preds, targets):
        # 确保输入是float32类型
        if isinstance(targets, torch.Tensor) and targets.dtype != torch.float32:
            targets = targets.float()

        if isinstance(preds, tuple):
            # 训练模式，有主输出和辅助输出
            main_pred, aux_pred = preds

            # 确保main_pred和aux_pred都是float32
            if main_pred.dtype != torch.float32:
                main_pred = main_pred.float()
            if aux_pred.dtype != torch.float32:
                aux_pred = aux_pred.float()

            main_loss = self.alpha * self.smooth_l1(main_pred, targets) + self.beta * self.mse(main_pred, targets)
            aux_loss = self.alpha * self.smooth_l1(aux_pred, targets) + self.beta * self.mse(aux_pred, targets)
            return main_loss + self.aux_weight * aux_loss
        else:
            # 评估模式，只有主输出
            if preds.dtype != torch.float32:
                preds = preds.float()

            return self.alpha * self.smooth_l1(preds, targets) + self.beta * self.mse(preds, targets)


# -------------------- 训练引擎 --------------------
def train(config: dict, logger: logging.Logger) -> dict:
    """优化后的训练流程"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"\n{'=' * 60} 训练初始化 {'=' * 60}")
    logger.info(f"配置名称: {config['name']}")
    logger.info(f"设备: {device} | 混合精度: {config.get('use_amp', True)}")

    model_dir = os.path.join("saved_models", config['name'])
    os.makedirs(model_dir, exist_ok=True)

    # 数据准备
    logger.info("数据准备中...")
    df = prepare_data(config['data_root'])
    train_loader, val_loader = get_data_loaders(df, config['batch_size'], config['image_size'])
    logger.info(f"数据集大小: 训练={len(train_loader.dataset)}, 验证={len(val_loader.dataset)}")

    # 模型初始化
    model = ThermalErrorModel(img_size=config['image_size']).to(device)

    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )

    # 余弦退火学习率调度器
    T_max = config['epochs']
    eta_min = config['lr'] / 100
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    # 损失函数
    criterion = CombinedLoss(alpha=0.6, beta=0.4, aux_weight=0.3)

    # 混合精度训练
    scaler = torch.amp.GradScaler(enabled=config['use_amp'])

    best_mae = float('inf')
    best_model_path = ""
    early_stop_counter = 0
    early_stop_patience = 10

    # 训练循环
    for epoch in range(config['epochs']):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_mae = 0.0

        progress_bar = tqdm(train_loader, desc=f"训练 Epoch {epoch + 1}/{config['epochs']}")
        for images, labels in progress_bar:
            images = images.to(device, non_blocking=True)
            # 确保标签是float32类型
            labels = labels.to(device, dtype=torch.float32, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type='cuda', enabled=config['use_amp']):
                outputs = model(images)  # 训练模式下返回(main_out, aux_out)
                loss = criterion(outputs, labels)

                # 计算MAE用于监控（仅使用主输出）
                main_outputs = outputs[0] if isinstance(outputs, tuple) else outputs
                mae = F.l1_loss(main_outputs, labels)

            if config['use_amp']:
                scaler.scale(loss).backward()
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_mae += mae.item() * images.size(0)

            # 更新进度条信息
            progress_bar.set_postfix({
                'loss': loss.item(),
                'mae': mae.item(),
                'lr': optimizer.param_groups[0]['lr']
            })

        # 验证阶段
        model.eval()
        val_loss, total_mae = 0.0, 0.0
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=config['use_amp']):
            for images, labels in tqdm(val_loader, desc=f"验证 Epoch {epoch + 1}"):
                images = images.to(device)
                # 确保标签是float32类型
                labels = labels.to(device, dtype=torch.float32)

                # 验证模式下只返回主输出
                outputs = model(images)
                loss = criterion(outputs, labels)
                mae = F.l1_loss(outputs, labels)

                val_loss += loss.item() * images.size(0)
                total_mae += mae.item() * images.size(0)

        # 计算平均损失和MAE
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_mae = train_mae / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_mae = total_mae / len(val_loader.dataset)

        # 更新学习率
        scheduler.step()

        # 记录训练状态
        logger.info(
            f"Epoch {epoch + 1}/{config['epochs']} | "
            f"训练损失: {avg_train_loss:.4f} | "
            f"训练MAE: {avg_train_mae:.4f} | "
            f"验证损失: {avg_val_loss:.4f} | "
            f"验证MAE: {avg_mae:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # 保存最佳模型
        if avg_mae < best_mae:
            if os.path.exists(best_model_path):
                os.remove(best_model_path)
            best_mae = avg_mae
            best_model_path = os.path.join(model_dir, f"best_epoch{epoch + 1}_mae{avg_mae:.4f}.pth")

            # 保存完整的训练状态
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_mae': best_mae,
                'config': config
            }, best_model_path)

            logger.info(f"✓ 发现新最佳模型! MAE改进: {best_mae:.4f}nm → 已保存到: {best_model_path}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            logger.info(f"未改进的轮次: {early_stop_counter}/{early_stop_patience}")

            # 每5个epoch保存一次检查点
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(model_dir, f"checkpoint_epoch{epoch + 1}.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'mae': avg_mae,
                    'config': config
                }, checkpoint_path)
                logger.info(f"✓ 已保存检查点: {checkpoint_path}")

        # 提前停止
        if early_stop_counter >= early_stop_patience:
            logger.info(f"⚠ {early_stop_patience}轮未见改进, 提前停止训练!")
            break

    return {'config': config['name'], 'best_mae': best_mae, 'model_path': best_model_path}


# -------------------- 模型评估 --------------------
def evaluate_model(model_path: str, df: pd.DataFrame, batch_size: int, image_size: int = 224):
    """评估已训练模型的性能"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    model = ThermalErrorModel(img_size=image_size).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 准备数据加载器
    _, val_loader = get_data_loaders(df, batch_size, image_size)

    # 评估指标
    total_mae = 0.0
    total_mse = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
        for images, labels in tqdm(val_loader, desc="评估中"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            # 记录预测和真实值
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

            # 计算指标
            mae = F.l1_loss(outputs, labels, reduction='sum')
            mse = F.mse_loss(outputs, labels, reduction='sum')

            total_mae += mae.item()
            total_mse += mse.item()

    # 计算最终指标
    avg_mae = total_mae / len(val_loader.dataset)
    avg_mse = total_mse / len(val_loader.dataset)
    rmse = np.sqrt(avg_mse)

    # 计算R²
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # 检查极端误差
    abs_errors = np.abs(all_targets - all_preds)
    percentile_90 = np.percentile(abs_errors, 90)
    percentile_95 = np.percentile(abs_errors, 95)
    max_error = np.max(abs_errors)

    results = {
        'MAE': avg_mae,
        'RMSE': rmse,
        'R²': r_squared,
        '90th_Percentile_Error': percentile_90,
        '95th_Percentile_Error': percentile_95,
        'Max_Error': max_error
    }

    return results


# -------------------- 主程序 --------------------
def main():
    """主控制流程"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data').replace('\\', '/')

    configs = [
        {
            'name': 'resnet_base',
            'data_root': data_path,
            'batch_size': 32,
            'image_size': 224,
            'lr': 1e-4,
            'epochs': 100,
            'weight_decay': 1e-4,
            'use_amp': False
        },
        {
            'name': 'resnet_large_batch',
            'data_root': data_path,
            'batch_size': 128,
            'image_size': 224,
            'lr': 4e-4,
            'epochs': 80,
            'weight_decay': 1e-4,
            'use_amp': True
        },
        {
            'name': 'resnet_low_lr',
            'data_root': data_path,
            'batch_size': 64,
            'image_size': 224,
            'lr': 5e-5,
            'epochs': 120,
            'weight_decay': 1e-4,
            'use_amp': False
        },
        {
            'name': 'resnet_amp_optimized',
            'data_root': data_path,
            'batch_size': 256,
            'image_size': 224,
            'lr': 2e-4,
            'epochs': 100,
            'weight_decay': 1e-4,
            'use_amp': True
        }
    ]

    parser = argparse.ArgumentParser(description='机床热误差预测平台')
    parser.add_argument('--config', nargs='+', choices=[c['name'] for c in configs],
                        default=['resnet_high_precision'], help='选择训练配置')
    parser.add_argument('--evaluate', action='store_true', help='评估模式')
    parser.add_argument('--model_path', type=str, help='评估模式下的模型路径')
    args = parser.parse_args()

    if args.evaluate and args.model_path:
        # 评估模式
        df = prepare_data(configs[0]['data_root'])
        results = evaluate_model(args.model_path, df, batch_size=16, image_size=224)

        print("\n" + "=" * 60)
        print("模型评估结果:")
        print("=" * 60)
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
        print("=" * 60)

        # 将结果保存到CSV
        pd.DataFrame([results]).to_csv("evaluation_results.csv", index=False)
        print("评估结果已保存到 evaluation_results.csv")
    else:
        # 训练模式
        results = []
        for cfg in configs:
            if cfg['name'] in args.config:
                try:
                    logger = setup_logger(cfg['name'])
                    result = train(cfg, logger)
                    results.append(result)

                    logger.info(f"\n{'=' * 60} 训练完成 {'=' * 60}")
                    logger.info(f"最佳MAE: {result['best_mae']:.4f}nm")
                    logger.info(f"模型保存路径: {result['model_path']}")
                except Exception as e:
                    logger = setup_logger(cfg['name']) if 'logger' not in locals() else logger
                    logger.error(f"训练异常终止: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue

        if results:
            report = pd.DataFrame(results)
            report.to_csv("training_summary.csv", index=False)
            print("\n训练总结报告已生成: training_summary.csv")


if __name__ == "__main__":
    main()