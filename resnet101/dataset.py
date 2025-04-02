import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
from charset_normalizer import from_path  # 自动检测编码

class ThermalDataset(Dataset):
    """主轴热变形红外成像数据集类"""
    def __init__(self, dataframe, transform=None):
        """
        参数:
            dataframe: 包含图像路径和标签的数据框
            transform: 图像预处理变换
        """
        self.dataframe = dataframe
        self.transform = transform or transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet均值
                                 std=[0.229, 0.224, 0.225])  # ImageNet标准差
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        # 构建完整图像路径
        img_path = os.path.join(row['round_path'], 'thermal_images', row['Image'])
        img_path = os.path.normpath(img_path).replace('\\', '/')
        image = Image.open(img_path).convert('RGB')  # 确保转换为RGB三通道

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(row['target'], dtype=torch.float32)
        return image, label

def prepare_data(root_dir=r'D:\毕业设计\wzz\code\project\data', test_size=0.2):
    """
    准备训练和验证数据集
    参数:
        root_dir: 数据根目录（包含各轮次仿真结果的文件夹）
        test_size: 验证集比例
    返回:
        包含完整信息的数据框
    """
    # 收集所有有效轮次数据
    rounds = []
    for round_name in os.listdir(root_dir):
        round_path = os.path.join(root_dir, round_name)
        csv_path = os.path.join(round_path, 'thermal_errors.csv')
        if os.path.exists(csv_path):
            rounds.append(round_path)

    # 按轮次划分训练验证集（保持同一轮次数据不分散）
    train_rounds, val_rounds = train_test_split(rounds, test_size=test_size, random_state=42)

    # 构建完整数据框
    df_list = []
    for round_path in rounds:
        csv_path = os.path.join(round_path, 'thermal_errors.csv')
        # 自动检测CSV文件编码
        detection_result = from_path(csv_path).best()
        detected_encoding = detection_result.encoding if detection_result else 'utf-8'
        df = pd.read_csv(csv_path, encoding=detected_encoding)

        # 验证数据列是否存在
        if 'Thermal_Error(nm)' not in df.columns:
            raise ValueError(f"CSV文件 {csv_path} 中缺少 목표 열 Thermal_Error(nm)")

        df['round_path'] = round_path  # 记录原始路径
        df['target'] = df['Thermal_Error(nm)']  # 统一目标列名
        df_list.append(df[['Image', 'target', 'round_path']])

    full_df = pd.concat(df_list)

    # 标记数据用途
    full_df['dataset'] = 'train'
    full_df.loc[full_df['round_path'].isin(val_rounds), 'dataset'] = 'val'

    return full_df

def get_data_loaders(df, batch_size=32, num_workers=4):
    """
    创建训练和验证数据加载器
    参数:
        df: 包含数据集划分信息的数据框
        batch_size: 批次大小
        num_workers: 数据加载线程数
    返回:
        train_loader, val_loader: 训练和验证数据加载器
    """
    # 训练集数据增强
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 验证集预处理（无增强）
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 划分数据集
    train_df = df[df['dataset'] == 'train'].reset_index(drop=True)
    val_df = df[df['dataset'] == 'val'].reset_index(drop=True)

    # 创建数据集实例
    train_dataset = ThermalDataset(train_df, transform=train_transform)
    val_dataset = ThermalDataset(val_df, transform=val_transform)

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

if __name__ == "__main__":
    print("正在测试数据集加载...")
    try:
        # 测试数据准备
        test_df = prepare_data()
        print(f"\n数据集统计:")
        print(f"总样本数: {len(test_df)}")
        print(f"训练集: {len(test_df[test_df['dataset'] == 'train'])} 个样本")
        print(f"验证集: {len(test_df[test_df['dataset'] == 'val'])} 个样本")

        # 测试数据加载
        train_loader, val_loader = get_data_loaders(test_df, batch_size=4)
        images, labels = next(iter(train_loader))
        print(f"\n批次数据测试:")
        print(f"图像尺寸: {images.shape} (batch×channel×height×width)")
        print(f"标签尺寸: {labels.shape}")
        print(f"示例标签值: {labels[:5].tolist()}")

        print("\n测试通过！")
    except Exception as e:
        print(f"\n测试失败: {str(e)}")
        raise