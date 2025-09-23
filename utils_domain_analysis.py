# 文件名: utils_domain_analysis.py (修正版)

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.impute import SimpleImputer

def mmd_rbf(X, Y, gamma=1.0):
    """计算两个样本集X和Y之间的MMD（使用RBF核）"""
    XX = rbf_kernel(X, X, gamma)
    YY = rbf_kernel(Y, Y, gamma)
    XY = rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()

def visualize_domain_distribution(source_df: pd.DataFrame, 
                                  target_df: pd.DataFrame, 
                                  imputer: object, 
                                  scaler: object, 
                                  pca: object,
                                  save_path: str):
    """
    使用t-SNE可视化源域和目标域在PCA降维后的特征空间分布，并计算MMD。
    """
    print("\n--- 开始进行领域分布可视化与MMD计算 ---")
    
    common_cols = [col for col in source_df.columns if col in target_df.columns and col not in ['label', 'experiment_id', 'original_path']]
    X_source = source_df[common_cols]
    X_target = target_df[common_cols]

    X_source_imputed = imputer.transform(X_source)
    X_target_imputed = imputer.transform(X_target)
    
    X_source_scaled = scaler.transform(X_source_imputed)
    X_target_scaled = scaler.transform(X_target_imputed)
    
    X_source_pca = pca.transform(X_source_scaled)
    X_target_pca = pca.transform(X_target_scaled)

    mmd_value = mmd_rbf(X_source_pca, X_target_pca)
    print(f"源域与目标域在PCA空间中的MMD²值为: {mmd_value:.6f} (值越大, 差异越大)")

    # t-SNE 可视化
    X_combined = np.vstack((X_source_pca, X_target_pca))
    domain_labels = np.array([0] * len(X_source_pca) + [1] * len(X_target_pca))

    # --- 【核心修正】将 'n_iter' 修改为 'max_iter' ---
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    X_tsne = tsne.fit_transform(X_combined)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_tsne[domain_labels==0, 0], y=X_tsne[domain_labels==0, 1], 
                    hue=source_df['label'], style=np.array(['Source'] * len(X_source_pca)),
                    palette='viridis', alpha=0.7)
    sns.scatterplot(x=X_tsne[domain_labels==1, 0], y=X_tsne[domain_labels==1, 1],
                    hue=np.array(['Target'] * len(X_target_pca)), style=np.array(['Target'] * len(X_target_pca)),
                    palette=['red'], s=100, marker='X', alpha=0.9)
                    
    plt.title(f't-SNE Visualization of Source vs Target Domains (MMD²: {mmd_value:.4f})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    
    plt.savefig(save_path)
    plt.close()
    print(f"领域分布对比图已保存至: {save_path}")


def rank_features_by_transferability(source_df: pd.DataFrame, 
                                     target_df: pd.DataFrame, 
                                     imputer: object,
                                     scaler: object,
                                     save_path: str):
    """
    对每个特征单独计算源域和目标域之间的MMD，以此评估特征的可迁移性。
    """
    print("\n--- 开始评估各特征的可迁移性 (Transferability) ---")
    
    common_cols = [col for col in source_df.columns if col in target_df.columns and col not in ['label', 'experiment_id', 'original_path']]
    X_source = source_df[common_cols]
    X_target = target_df[common_cols]

    X_source_imputed = imputer.transform(X_source)
    X_target_imputed = imputer.transform(X_target)

    X_source_scaled = scaler.transform(X_source_imputed)
    X_target_scaled = scaler.transform(X_target_imputed)
    
    mmd_scores = {}
    for i, feature_name in enumerate(tqdm(common_cols, desc="Ranking Features")):
        source_feature = X_source_scaled[:, i].reshape(-1, 1)
        target_feature = X_target_scaled[:, i].reshape(-1, 1)
        
        if np.isnan(source_feature).any() or np.isnan(target_feature).any():
            continue
        mmd_scores[feature_name] = mmd_rbf(source_feature, target_feature)
        
    transferability_ranking = sorted(mmd_scores.items(), key=lambda item: item[1])
    
    report_content = "--- 特征可迁移性排名 (MMD分数越低越好) ---\n"
    for rank, (feature, score) in enumerate(transferability_ranking):
        report_content += f"{rank+1}. {feature}: {score:.6f}\n"

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"特征可迁移性排名报告已保存至: {save_path}")
    
    return transferability_ranking