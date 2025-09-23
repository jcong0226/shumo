# 文件名: utils_visualization.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings

def plot_feature_distributions(features_df: pd.DataFrame, 
                               features_to_plot: list, 
                               save_dir: str,
                               plot_type: str = 'kde'):
    """
    【新增功能】为指定的关键特征生成并保存特征分布图，以对比不同故障类型。

    Args:
        features_df (pd.DataFrame): 包含手工特征和'label'列的DataFrame。
        features_to_plot (list): 需要进行可视化对比的特征名称列表。
        save_dir (str): 保存生成图像的目录。
        plot_type (str): 绘图类型, 'kde' (核密度估计) 或 'hist' (直方图)。
    """
    print(f"\n--- 开始生成关键特征的分布对比图 ({plot_type}) ---")
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 忽略绘图时可能出现的警告
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    for feature in features_to_plot:
        if feature not in features_df.columns:
            print(f"警告: 特征 '{feature}' 不在DataFrame中，已跳过。")
            continue
            
        plt.figure(figsize=(10, 6))
        
        if plot_type == 'kde':
            # 使用核密度估计图 (KDE) 来观察分布平滑曲线
            sns.kdeplot(data=features_df, x=feature, hue='label', fill=True, common_norm=False)
            plt.title(f'Distribution of "{feature}" by Fault Type (KDE)')
        
        elif plot_type == 'hist':
            # 使用直方图
            sns.histplot(data=features_df, x=feature, hue='label', multiple='stack', bins=50)
            plt.title(f'Distribution of "{feature}" by Fault Type (Histogram)')

        else:
            print(f"错误: 不支持的绘图类型 '{plot_type}'。请选择 'kde' 或 'hist'。")
            return

        plt.xlabel(f"Feature Value: {feature}")
        plt.ylabel("Density" if plot_type == 'kde' else "Count")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # --- 保存图像 ---
        safe_filename = f"distribution_{feature}.png"
        save_path = os.path.join(save_dir, safe_filename)
        plt.savefig(save_path)
        plt.close()
        
    print(f"所有特征分布图已保存至: {save_dir}")
    warnings.filterwarnings("default", category=FutureWarning)
    
def plot_feature_means_by_fault_type(features_df: pd.DataFrame, 
                                     features_to_compare: list, 
                                     save_dir: str):
    """
    【新增功能】定量对比关键特征在不同故障类型间的均值，并生成对比条形图。

    Args:
        features_df (pd.DataFrame): 包含手工特征和'label'列的DataFrame。
        features_to_compare (list): 需要进行可视化对比的特征名称列表。
        save_dir (str): 保存生成图像的目录。
    """
    print("\n--- 开始生成故障类型间的特征均值定量对比图 ---")
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 计算每个故障类别下, 指定特征的均值
    mean_features = features_df.groupby('label')[features_to_compare].mean()
    
    # 2. 准备绘图
    # 动态确定子图的布局，我们每行最多显示2个图
    num_features = len(features_to_compare)
    n_cols = 2
    n_rows = (num_features + n_cols - 1) // n_cols # 向上取整
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 7, n_rows * 5))
    # 将axes展平，方便索引
    axes = axes.flatten()
    
    fig.suptitle('Quantitative Comparison of Key Feature Means Across Fault Types', fontsize=16)
    
    # 3. 为每个特征绘制一个条形图
    for i, feature in enumerate(features_to_compare):
        ax = axes[i]
        
        # 提取当前特征的数据并按均值排序
        feature_means = mean_features[feature].sort_values(ascending=False)
        
        # 使用seaborn绘制条形图
        sns.barplot(x=feature_means.index, y=feature_means.values, ax=ax, palette='viridis')
        
        ax.set_title(f'Mean of "{feature}"', fontsize=12)
        ax.set_xlabel('Fault Type', fontsize=10)
        ax.set_ylabel('Mean Value', fontsize=10)
        ax.tick_params(axis='x', rotation=15) # X轴标签倾斜，避免重叠
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    # 隐藏多余的子图
    for i in range(num_features, len(axes)):
        fig.delaxes(axes[i])
        
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # 调整布局，为总标题留出空间
    
    # 4. 保存图像
    save_path = os.path.join(save_dir, "feature_mean_comparison_by_fault_type.png")
    plt.savefig(save_path)
    plt.close()
    
    print(f"特征均值对比图已保存至: {save_path}")
    

def plot_feature_correlation_heatmap(features_df: pd.DataFrame, 
                                     top_n_features: int,
                                     save_dir: str):
    """
    【新增功能】计算特征间的相关性，并为最稳定的特征生成和保存相关性热图。

    Args:
        features_df (pd.DataFrame): 包含所有手工特征的DataFrame。
        top_n_features (int): 选择多少个最稳定的特征来绘制热图。
        save_dir (str): 保存生成图像的目录。
    """
    print(f"\n--- 开始为Top {top_n_features}个特征生成相关性热图 ---")
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 仅选择数值类型的特征列进行相关性计算
    features_only_df = features_df.drop(columns=['label', 'experiment_id', 'original_path'], errors='ignore')
    
    # 为了避免热图过于拥挤，我们只选择部分最重要的特征进行可视化
    # 这里我们简单地选择前 top_n_features 个特征
    if features_only_df.shape[1] > top_n_features:
        features_to_plot = features_only_df.iloc[:, :top_n_features]
        print(f"注意: 特征过多({features_only_df.shape[1]}个)，仅选择前 {top_n_features} 个特征进行可视化。")
    else:
        features_to_plot = features_only_df
        
    # 2. 计算相关性矩阵
    correlation_matrix = features_to_plot.corr()
    
    # 3. 准备绘图
    plt.figure(figsize=(16, 14))
    
    # 使用seaborn绘制热图
    sns.heatmap(correlation_matrix, 
                annot=False, # 对于大量特征，显示数值会很乱，故关闭
                cmap='coolwarm', 
                linewidths=.5)
    
    plt.title(f'Feature Correlation Heatmap (Top {len(features_to_plot.columns)} Features)', fontsize=16)
    plt.xticks(rotation=80, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # 4. 保存图像
    save_path = os.path.join(save_dir, "feature_correlation_heatmap.png")
    plt.savefig(save_path)
    plt.close()
    
    print(f"特征相关性热图已保存至: {save_path}")
    
