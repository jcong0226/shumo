# 文件名: enhanced_subject_1/utils_stratified_split.py

import pandas as pd
from sklearn.model_selection import train_test_split

def stratified_group_split(features_df: pd.DataFrame, group_col: str, stratify_col: str, test_size: float = 0.3, random_state: int = 42):
    """
    实现分组分层抽样。

    该函数首先按组（例如 experiment_id）对数据进行分组，然后对这些组进行分层抽样，
    确保划分后的训练集和测试集中，每个类别（例如 Fault_Type）的组数比例大致均衡。

    Args:
        features_df (pd.DataFrame): 包含特征、分组列和分层列的完整DataFrame。
        group_col (str): 用于分组的列名 (e.g., 'experiment_id').
        stratify_col (str): 用于分层的列名 (e.g., 'label').
        test_size (float): 测试集所占的比例。
        random_state (int): 随机种子，确保结果可复现。

    Returns:
        tuple: (train_df, test_df)，划分好的训练集和测试集DataFrame。
    """
    print(f"--- 正在执行分组分层抽样 (按'{group_col}'分组, 按'{stratify_col}'分层) ---")
    
    # 1. 获取每个组的标签（我们假设一个组只有一个标签）
    #    创建一个包含每个组及其对应标签的映射
    group_labels = features_df.groupby(group_col)[stratify_col].first()
    
    # 2. 对“组”进行分层抽样
    #    我们在这里划分的是组ID，而不是样本
    train_groups, test_groups = train_test_split(
        group_labels.index,
        test_size=test_size,
        stratify=group_labels.values, # <-- 关键：根据组的标签进行分层
        random_state=random_state
    )
    
    # 3. 根据划分好的组ID，筛选出完整的训练和测试数据
    train_df = features_df[features_df[group_col].isin(train_groups)].copy()
    test_df = features_df[features_df[group_col].isin(test_groups)].copy()
    
    print(f"原始数据集样本数: {len(features_df)}")
    print(f"训练集样本数: {len(train_df)} (来自 {len(train_groups)} 个独立实验)")
    print(f"测试集样本数: {len(test_df)} (来自 {len(test_groups)} 个独立实验)")
    
    return train_df, test_df