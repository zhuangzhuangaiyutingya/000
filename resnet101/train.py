import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet101
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import argparse
import os
import pandas as pd
import logging
from charset_normalizer import from_path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

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
    train_df = full_df.sample(frac=0.8, random_state=42)
    val_df = full_df.drop(train_df.index)
    return pd.concat([train_df.assign(dataset='train'), val_df.assign(dataset='val')])

def get_data_loaders(df: pd.DataFrame, batch_size: int) -> tuple:
    """数据加载器工厂"""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
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

# -------------------- 模型定义 --------------------
class ChannelAttention(nn.Module):
    """通道注意力模块（参考CBAM）"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ThermalErrorModel(nn.Module):
    """改进版热误差预测模型（基于ResNet-resnet101）"""
    def __init__(self):
        super().__init__()
        base_model = resnet101(pretrained=True)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            base_model.bn1,
            base_model.relu,
            base_model.maxpool
        )

        self.layer1 = self._make_layer(base_model.layer1, 256)
        self.layer2 = self._make_layer(base_model.layer2, 512)
        self.layer3 = self._make_layer(base_model.layer3, 1024)
        self.layer4 = self._make_layer(base_model.layer4, 2048)

        self.ca1 = ChannelAttention(256)
        self.ca2 = ChannelAttention(512)
        self.ca3 = ChannelAttention(1024)

        self.fusion = nn.Sequential(
            nn.Conv2d(3840, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def _make_layer(self, base_layer, channels):
        layers = list(base_layer.children())
        return nn.Sequential(*layers, ChannelAttention(channels))

    def forward(self, x):
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        f1 = nn.functional.interpolate(f1, size=(7, 7), mode='bilinear')
        f2 = nn.functional.interpolate(f2, size=(7, 7), mode='bilinear')
        f3 = nn.functional.interpolate(f3, size=(7, 7), mode='bilinear')
        f4 = f4  # 已经是7x7
        fused = torch.cat([f1, f2, f3, f4], dim=1)
        fused = self.fusion(fused)
        return self.regressor(fused).squeeze()

# -------------------- 训练引擎 --------------------
def train(config: dict, logger: logging.Logger) -> dict:
    """优化后的训练流程"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"\n{'=' * 60} 训练初始化 {'=' * 60}")
    logger.info(f"配置名称: {config['name']}")
    logger.info(f"设备: {device} | 混合精度: {config.get('use_amp', False)}")

    model_dir = os.path.join("saved_models", config['name'])
    os.makedirs(model_dir, exist_ok=True)

    df = prepare_data(config['data_root'])
    train_loader, val_loader = get_data_loaders(df, config['batch_size'])

    model = ThermalErrorModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)
    criterion = nn.SmoothL1Loss()
    scaler = torch.amp.GradScaler('cuda', enabled=config['use_amp'])

    best_mae = float('inf')
    best_model_path = ""

    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"训练 Epoch {epoch + 1}"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=config['use_amp']):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if config['use_amp']:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * images.size(0)

        model.eval()
        val_loss, total_mae = 0.0, 0.0
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=config['use_amp']):
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                mae = nn.L1Loss()(outputs, labels)
                val_loss += loss.item() * images.size(0)
                total_mae += mae.item() * images.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_mae = total_mae / len(val_loader.dataset)

        scheduler.step(avg_val_loss)

        if avg_mae < best_mae:
            if os.path.exists(best_model_path):
                os.remove(best_model_path)
            best_mae = avg_mae
            best_model_path = os.path.join(model_dir, f"best_epoch{epoch + 1}_mae{avg_mae:.2f}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_mae': best_mae
            }, best_model_path)
            logger.info(f"发现新最佳模型 ➔ 保存到: {best_model_path}")

        logger.info(f"Epoch {epoch + 1}/{config['epochs']} | 训练损失: {avg_train_loss:.2f} | 验证损失: {avg_val_loss:.2f} | MAE: {avg_mae:.2f}")

    return {'config': config['name'], 'best_mae': best_mae, 'model_path': best_model_path}

# -------------------- 主程序 --------------------
def main():
    """主控制流程"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data').replace('\\', '/')

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
                logger = setup_logger(cfg['name'])
                result = train(cfg, logger)
                results.append(result)

                logger.info(f"\n{'='*60} 训练完成 {'='*60}")
                logger.info(f"最佳MAE: {result['best_mae']:.2f}nm")
                logger.info(f"模型保存路径: {result['model_path']}")
            except Exception as e:
                logger.error(f"训练异常终止: {str(e)}")
                continue

    if results:
        report = pd.DataFrame(results)
        report.to_csv("training_summary.csv", index=False)
        print("\n训练总结报告已生成: training_summary.csv")

if __name__ == "__main__":
    main()