# 您可以创建一个新文件 utils_robustness_analysis.py
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from tqdm import tqdm
import pandas as pd


def mmd_rbf(X, Y, gamma=1.0):
    """计算两个样本集X和Y之间的MMD（使用RBF核）"""
    XX = rbf_kernel(X, X, gamma)
    YY = rbf_kernel(Y, Y, gamma)
    XY = rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()

def evaluate_feature_robustness(features_df: pd.DataFrame, save_path: str = None): 
    """
    评估特征在不同载荷下的稳健性。

    Args:
        features_df (pd.DataFrame): 包含手工特征以及'label'和'load'列的DataFrame。
    """
    print(f"\n--- 开始特征稳健性评估 (跨载荷) ---")
    
    feature_cols = features_df.drop(columns=['label', 'experiment_id', 'original_path', 'load', 'rpm', 'fault_size']).columns
    loads = sorted(features_df['load'].unique())
    
    robustness_scores = {}
    
    # 我们以内圈故障为例进行分析
    fault_class_df = features_df[features_df['label'] == 'Inner_Race'].copy()
    if len(fault_class_df) == 0:
        print("未找到'Inner_Race'样本，无法进行稳健性评估。")
        return

    for feature in tqdm(feature_cols, desc="Evaluating Robustness"):
        total_mmd = 0
        pair_count = 0
        
        # 两两比较不同载荷下的特征分布
        for i in range(len(loads)):
            for j in range(i + 1, len(loads)):
                load1_data = fault_class_df[fault_class_df['load'] == loads[i]][feature].values.reshape(-1, 1)
                load2_data = fault_class_df[fault_class_df['load'] == loads[j]][feature].values.reshape(-1, 1)

                if len(load1_data) > 1 and len(load2_data) > 1:
                    total_mmd += mmd_rbf(load1_data, load2_data)
                    pair_count += 1
        
        if pair_count > 0:
            # MMD越小，说明特征在不同载荷下越稳定（分布越相似）
            robustness_scores[feature] = total_mmd / pair_count

    # 按分数从低到高排序（越稳定越靠前）
    sorted_robust_features = sorted(robustness_scores.items(), key=lambda item: item[1])
    
    # --- 【修改】将打印内容改为写入文件 ---
    report_content = "特征稳健性排名 (MMD分数越低越好):\n"
    print("\n" + report_content.strip())
    
    for i, (feature, score) in enumerate(sorted_robust_features[:30]):
        line = f"{i+1}. {feature}: {score:.6f}\n"
        report_content += line
        print(line.strip())
    
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"\n稳健性分析报告已保存至: {save_path}")
        
    return sorted_robust_features