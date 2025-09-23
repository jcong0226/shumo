# 文件名: utils_transfer_preparation.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel
import joblib
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def mmd_rbf(X, Y, gamma=1.0):
    """计算两个样本集X和Y之间的MMD（使用RBF核）"""
    XX = rbf_kernel(X, X, gamma)
    YY = rbf_kernel(Y, Y, gamma)
    XY = rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()

def calculate_domain_discrepancy(source_features_df: pd.DataFrame, 
                                 target_features_df: pd.DataFrame,
                                 scaler: StandardScaler,
                                 pca: PCA,
                                 plot_save_path: str = None) -> float: # 新增路径参数
    """
    计算源域和目标域在PCA特征空间中的MMD差异度。

    Args:
        source_features_df (pd.DataFrame): 源域的手工特征。
        target_features_df (pd.DataFrame): 目标域的手工特征。
        scaler (StandardScaler): 在源域上训练好的缩放器。
        pca (PCA): 在源域上训练好的PCA模型。

    Returns:
        float: MMD值。
    """
    print("\n--- 正在计算源域和目标域的MMD差异度 ---")
    
    # 提取纯特征部分
    X_source = source_features_df.drop(columns=['label', 'experiment_id', 'original_path', 'load', 'rpm', 'fault_size'], errors='ignore')
    X_target = target_features_df.drop(columns=['label', 'experiment_id', 'original_path', 'load', 'rpm', 'fault_size'], errors='ignore')
    
    # **重要**: 使用在源域上fit好的scaler和pca来transform目标域数据
    X_source_scaled = scaler.transform(X_source)
    X_target_scaled = scaler.transform(X_target)
    
    X_source_pca = pca.transform(X_source_scaled)
    X_target_pca = pca.transform(X_target_scaled)
    # 计算MMD值
    mmd_value = mmd_rbf(X_source_pca, X_target_pca)
    # --- 【修改】可视化与保存 ---
    print("正在生成 t-SNE 可视化图...")
    X_combined = np.vstack((X_source_pca, X_target_pca))
    domain_labels = np.array([0] * len(X_source_pca) + [1] * len(X_target_pca))

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_combined)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[domain_labels==0, 0], X_tsne[domain_labels==0, 1], c='blue', label='Source Domain', alpha=0.5)
    plt.scatter(X_tsne[domain_labels==1, 0], X_tsne[domain_labels==1, 1], c='red', label='Target Domain', alpha=0.5)
    plt.title(f't-SNE of Source vs Target (MMD²: {mmd_value:.4f})')
    plt.legend()
    
    if plot_save_path:
        plt.savefig(plot_save_path)
        print(f"t-SNE对比图已保存至: {plot_save_path}")
    plt.close()
    
    print(f"源域与目标域之间的MMD²值为: {mmd_value:.6f}")
    print("（提示：值越大，表示两个领域的特征分布差异越大）")
    
    return mmd_value

    
    
    print(f"源域与目标域之间的MMD²值为: {mmd_value:.6f}")
    print("（提示：值越大，表示两个领域的特征分布差异越大）")
    
    return mmd_value

def aggregate_window_predictions(window_predictions_df: pd.DataFrame, 
                                 method: str = 'average_probability') -> pd.DataFrame:
    """
    将窗口级的预测结果聚合成文件级的诊断结论。

    Args:
        window_predictions_df (pd.DataFrame): 包含窗口预测结果的DataFrame。
            需要包含列: 'original_path', 'predicted_label', 以及各分类的概率列 (如 'prob_Ball', 'prob_Inner_Race'...)
        method (str): 'majority_vote' 或 'average_probability'。

    Returns:
        pd.DataFrame: 文件级的最终诊断结果。
    """
    print(f"\n--- 正在使用 '{method}' 方法进行文件级结果聚合 ---")
    
    # 从路径中提取文件名
    window_predictions_df['filename'] = window_predictions_df['original_path'].apply(lambda x: os.path.basename(x).split('_seg_')[0])
    
    if method == 'majority_vote':
        # 按文件名分组，找到出现次数最多的预测标签
        file_level_preds = window_predictions_df.groupby('filename')['predicted_label'].agg(lambda x: x.mode().iloc[0]).reset_index()
        file_level_preds = file_level_preds.rename(columns={'predicted_label': 'final_diagnosis'})
        
    elif method == 'average_probability':
        # 按文件名分组，计算每个类别的平均概率
        prob_cols = [col for col in window_predictions_df.columns if col.startswith('prob_')]
        file_level_probs = window_predictions_df.groupby('filename')[prob_cols].mean().reset_index()
        
        # 找到每个文件平均概率最高的类别作为最终诊断
        prob_matrix = file_level_probs[prob_cols]
        file_level_probs['final_diagnosis'] = prob_matrix.idxmax(axis=1).str.replace('prob_', '')
        file_level_preds = file_level_probs
        
    else:
        raise ValueError("聚合方法必须是 'majority_vote' 或 'average_probability'")
        
    return file_level_preds