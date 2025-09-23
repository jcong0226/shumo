# 文件名: utils_deep_learning.py (最终修正版)

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm
import numpy as np

class MultiChannelImageDataset(Dataset):
    """
    【修正版】自定义PyTorch数据集，用于加载三通道的特征图像，并记录分组信息。
    """
    def __init__(self, image_dirs: dict, transform=None):
        
        self.ts_dir = image_dirs['ts']
        self.sp_dir = image_dirs['sp']
        self.env_sp_dir = image_dirs['env_sp']
        self.stft_dir = image_dirs['stft']
        self.cwt_dir = image_dirs['cwt']
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        # --- **修正1**: 将 self.groups 的初始化移到 _prepare_dataset 调用之前 ---
        self.groups = []
        
        self._prepare_dataset()

    def _prepare_dataset(self):
        classes = sorted(os.listdir(self.ts_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        
        for cls_name in classes:
            ts_class_path = os.path.join(self.ts_dir, cls_name)
            for fname in sorted(os.listdir(ts_class_path)):
                # --- **修改2**: 构建所有5种图像的路径 ---
                paths = {
                    'ts': os.path.join(self.ts_dir, cls_name, fname),
                    'sp': os.path.join(self.sp_dir, cls_name, fname),
                    'env_sp': os.path.join(self.env_sp_dir, cls_name, fname),
                    'stft': os.path.join(self.stft_dir, cls_name, fname),
                    'cwt': os.path.join(self.cwt_dir, cls_name, fname)
                }
                
                # 确保所有图像文件都存在
                if all(os.path.exists(p) for p in paths.values()):
                    self.samples.append((list(paths.values()), self.class_to_idx[cls_name]))
                    group_id = '_'.join(fname.split('_')[:-2])
                    self.groups.append(group_id)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # --- **修改3**: 加载5张图像并堆叠 ---
        img_paths, label = self.samples[idx]
        
        # 加载所有图像为灰度图
        images = [Image.open(p).convert("L") for p in img_paths]
        
        # PIL.Image.merge 不支持5通道，所以我们先转换为tensor再堆叠
        # 应用基础变换（例如缩放）
        if self.transform:
            images_tensors = [self.transform(img) for img in images]
        else: # 如果没有变换，至少要转成tensor
            to_tensor_transform = transforms.ToTensor()
            images_tensors = [to_tensor_transform(img) for img in images]
            
        # 将5个单通道张量 (1, H, W) 堆叠成一个5通道张量 (5, H, W)
        multi_channel_tensor = torch.cat(images_tensors, dim=0)
        
        return multi_channel_tensor, label

def train_evaluate_2d_resnet(image_dirs: dict, num_classes: int, save_dir: str, num_epochs: int = 10, report_save_dir: str=None, text_report_dir: str=None):
    """【5通道最终修正版】训练、评估并保存接受5通道输入的2D-ResNet模型。"""
    print("\n--- 开始处理 5通道输入 2D-ResNet 模型 ---")
    
    model_save_dir = os.path.join(save_dir, '2D_ResNet_5Channel') # 新建一个目录
    os.makedirs(model_save_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- **修改4**: 调整数据变换 ---
    # Normalize不再在这里做，因为我们需要对5通道张量进行操作
    # 并且每个通道都是灰度图，所以可以先用一个通用的标准化
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), # 将每个灰度图转换为 [0, 1] 的Tensor
    ])
    
    # 定义5通道的标准化
    # 对于灰度图，通常均值和标准差可以近似为0.5。我们可以用ImageNet的均值的平均值
    normalize_5_channel = transforms.Normalize(
        mean=[0.485, 0.456, 0.406, 0.449, 0.449], # 前三是ImageNet的，后两个是前三的均值
        std=[0.229, 0.224, 0.225, 0.226, 0.226]  # 理由同上
    )

    print("正在创建和准备数据集...")
    dataset = MultiChannelImageDataset(image_dirs=image_dirs, transform=base_transform)
    full_class_names = list(dataset.class_to_idx.keys())
    
    # ... (数据划分逻辑 GroupShuffleSplit 保持不变)
    print("正在按原始文件名进行分组划分数据...")
    labels = [sample[1] for sample in dataset.samples]; groups = dataset.groups
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    indices = np.arange(len(dataset))
    train_idx, val_idx = next(gss.split(indices, labels, groups))
    train_dataset = Subset(dataset, train_idx); val_dataset = Subset(dataset, val_idx)
    print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # ... (保存类别索引逻辑保持不变)
    class_to_idx_path = os.path.join(model_save_dir, 'class_to_idx.json')
    with open(class_to_idx_path, 'w') as f: json.dump(dataset.class_to_idx, f)

    # --- **修改5**: 加载模型并修改第一层卷积 ---
    print("正在加载预训练ResNet18并修改第一层以接受5通道输入...")
    model = models.resnet18(weights='IMAGENET1K_V1')
    
    # 获取原始conv1的权重
    original_weights = model.conv1.weight.clone() # (64, 3, 7, 7)
    
    # 创建一个新的5通道输入卷积层
    new_conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # 智能地初始化新权重
    with torch.no_grad():
        # 将原始3通道的权重复制到新层的前3个通道
        new_conv1.weight[:, :3, :, :] = original_weights
        # 对于新增的两个通道，使用原始权重的均值进行初始化
        mean_weights = original_weights.mean(dim=1, keepdim=True)
        new_conv1.weight[:, 3:4, :, :] = mean_weights # 第4个通道
        new_conv1.weight[:, 4:5, :, :] = mean_weights # 第5个通道
        
    # 替换模型的第一层
    model.conv1 = new_conv1

    # 修改最后一层以匹配类别数 (逻辑不变)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- 训练循环 ---
    print("开始训练模型...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]"):
            # 在这里应用5通道的标准化
            inputs = normalize_5_channel(inputs)
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}")

    # ... (保存模型和评估逻辑基本保持不变)
    model_path = os.path.join(model_save_dir, 'resnet18_5channel.pth')
    torch.save(model.state_dict(), model_path)
    print(f"训练好的模型权重已保存至: {model_path}")

    # --- 评估模型 ---
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="[Validation]"):
            # 同样，在这里应用5通道标准化
            inputs = normalize_5_channel(inputs)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # ... (后续的报告和混淆矩阵绘制逻辑完全保持不变)
    # ...
    
    return model, f1_score(all_labels, all_preds, average='macro')