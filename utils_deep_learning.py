# 文件名: utils_deep_learning.py (最终修正版 - 统一数据划分策略)

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
# 【核心修改1】: 导入 train_test_split 用于分层抽样
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

class MultiChannelImageDataset(Dataset):
    # ... (此部分代码完全不变)
    def __init__(self, image_dirs: dict, transform=None):
        self.ts_dir = image_dirs['ts']
        self.sp_dir = image_dirs['sp']
        self.env_sp_dir = image_dirs['env_sp']
        self.stft_dir = image_dirs['stft']
        self.cwt_dir = image_dirs['cwt']
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.groups = []
        self._prepare_dataset()

    def _prepare_dataset(self):
        classes = sorted(os.listdir(self.ts_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        for cls_name in classes:
            ts_class_path = os.path.join(self.ts_dir, cls_name)
            for fname in sorted(os.listdir(ts_class_path)):
                paths = {
                    'ts': os.path.join(self.ts_dir, cls_name, fname),
                    'sp': os.path.join(self.sp_dir, cls_name, fname),
                    'env_sp': os.path.join(self.env_sp_dir, cls_name, fname),
                    'stft': os.path.join(self.stft_dir, cls_name, fname),
                    'cwt': os.path.join(self.cwt_dir, cls_name, fname)
                }
                if all(os.path.exists(p) for p in paths.values()):
                    self.samples.append((list(paths.values()), self.class_to_idx[cls_name]))
                    group_id = '_'.join(fname.split('_')[:-2])
                    self.groups.append(group_id)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_paths, label = self.samples[idx]
        images = [Image.open(p).convert("L") for p in img_paths]
        if self.transform:
            images_tensors = [self.transform(img) for img in images]
        else:
            to_tensor_transform = transforms.ToTensor()
            images_tensors = [to_tensor_transform(img) for img in images]
        multi_channel_tensor = torch.cat(images_tensors, dim=0)
        return multi_channel_tensor, label

def train_evaluate_2d_resnet(image_dirs: dict, num_classes: int, save_dir: str, num_epochs: int = 10, report_save_dir: str=None, text_report_dir: str=None):
    """【5通道最终修正版】训练、评估并保存接受5通道输入的2D-ResNet模型。"""
    print("\n--- 开始处理 5通道输入 2D-ResNet 模型 ---")
    model_save_dir = os.path.join(save_dir, '2D_ResNet_5Channel')
    os.makedirs(model_save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    normalize_5_channel = transforms.Normalize(
        mean=[0.485, 0.456, 0.406, 0.449, 0.449],
        std=[0.229, 0.224, 0.225, 0.226, 0.226]
    )

    print("正在创建和准备数据集...")
    dataset = MultiChannelImageDataset(image_dirs=image_dirs, transform=base_transform)
    full_class_names = list(dataset.class_to_idx.keys())
    
    # 【核心修改2】: 替换 GroupShuffleSplit 为分组分层抽样逻辑
    print("正在执行分组分层抽样...")
    
    # 1. 创建一个包含所有样本索引、标签和分组信息的DataFrame
    df_for_split = pd.DataFrame({
        'index': range(len(dataset.samples)),
        'label': [s[1] for s in dataset.samples],
        'group': dataset.groups
    })

    # 2. 获取每个组的标签
    group_labels = df_for_split.groupby('group')['label'].first()

    # 3. 对“组”进行分层抽样，得到训练组ID和验证组ID
    train_groups, val_groups = train_test_split(
        group_labels.index,
        test_size=0.2, # 验证集比例
        stratify=group_labels.values,
        random_state=42
    )

    # 4. 根据划分好的组ID，筛选出训练和验证样本的原始索引
    train_idx = df_for_split[df_for_split['group'].isin(train_groups)]['index'].tolist()
    val_idx = df_for_split[df_for_split['group'].isin(val_groups)]['index'].tolist()
    # --- 修改结束 ---

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    
    print(f"训练集大小: {len(train_dataset)} (来自 {len(train_groups)} 个独立实验)")
    print(f"验证集大小: {len(val_dataset)} (来自 {len(val_groups)} 个独立实验)")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    class_to_idx_path = os.path.join(model_save_dir, 'class_to_idx.json')
    with open(class_to_idx_path, 'w') as f: json.dump(dataset.class_to_idx, f)

    # ... (模型加载、训练、评估和报告生成的代码完全不变)
    print("正在加载预训练ResNet18并修改第一层以接受5通道输入...")
    model = models.resnet18(weights='IMAGENET1K_V1')
    original_weights = model.conv1.weight.clone()
    new_conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        new_conv1.weight[:, :3, :, :] = original_weights
        mean_weights = original_weights.mean(dim=1, keepdim=True)
        new_conv1.weight[:, 3:4, :, :] = mean_weights
        new_conv1.weight[:, 4:5, :, :] = mean_weights
    model.conv1 = new_conv1
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("开始训练模型...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]"):
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

    model_path = os.path.join(model_save_dir, 'resnet18_5channel.pth')
    torch.save(model.state_dict(), model_path)
    print(f"训练好的模型权重已保存至: {model_path}")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="[Validation]"):
            inputs = normalize_5_channel(inputs)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    report = f"2D-ResNet (5-Channel) 模型的宏 F1 分数: {macro_f1:.4f}\n\n"
    report += classification_report(
        all_labels,
        all_preds,
        target_names=full_class_names,
        labels=np.arange(len(full_class_names))
    )
    print("\n" + report)

    if text_report_dir:
        report_filename = os.path.join(text_report_dir, 'classification_report_2d_resnet.txt')
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"深度学习模型分类报告已保存至: {report_filename}")

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=full_class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('2D-ResNet (5-Channel) Confusion Matrix')
    if report_save_dir:
        cm_path = os.path.join(report_save_dir, 'confusion_matrix_2d_resnet.png')
        plt.savefig(cm_path)
        plt.close()
        print(f"深度学习模型混淆矩阵已保存至: {cm_path}")

    return model, macro_f1