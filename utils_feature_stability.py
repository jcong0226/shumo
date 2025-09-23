import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

def analyze_feature_stability(features_df: pd.DataFrame, n_iterations: int = 50, top_k: int = 20,
                              text_save_path: str = None, plot_save_path: str = None): 
    """
    【新增功能】通过自举法(Bootstrap)评估特征选择的稳定性，并可视化为热图。
    
    Args:
        features_df (pd.DataFrame): 包含所有特征和'label'列的DataFrame。
        n_iterations (int): 自举采样的迭代次数。
        top_k (int): 每次迭代中记录的重要性排名前k的特征。
    """
    print(f"\n--- 开始进行特征稳定性分析 (迭代 {n_iterations} 次) ---")
    
    X = features_df.drop(columns=['label', 'experiment_id', 'original_path'], errors='ignore')
    y_raw = features_df['label']
    
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    # 用于记录每次迭代中被选中的特征
    selection_counts = pd.Series(np.zeros(X.shape[1]), index=X.columns)
    
    for i in range(n_iterations):
        # 1. 自举采样 (从原始数据中有放回地抽样)
        X_sample, y_sample = resample(X, y, random_state=i)
        
        # 2. 训练一个模型以获取特征重要性
        # 这里使用XGBoost作为示例，它的特征重要性计算很快
        model = XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', random_state=i)
        model.fit(X_sample, y_sample)
        
        # 3. 获取并记录重要性排名前k的特征
        importances = pd.Series(model.feature_importances_, index=X.columns)
        top_features = importances.nlargest(top_k).index
        
        # 4. 更新特征被选中的次数
        selection_counts[top_features] += 1
        
        print(f"  迭代 {i+1}/{n_iterations} 完成", end='\r')

    # 计算选择频率
    selection_frequency = selection_counts / n_iterations
    
    # 筛选出至少被选中过一次的特征并排序
    stable_features = selection_frequency[selection_frequency > 0].sort_values(ascending=False)
    
    # --- 【修改】保存文本报告 ---
    report_content = "--- 特征选择频率 (稳定性排名) ---\n" + stable_features.to_string()
    print("\n\n" + report_content)
    if text_save_path:
        with open(text_save_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"\n特征稳定性报告已保存至: {text_save_path}")

    # --- 【核心修正】替换占位符为完整的绘图代码 ---
    if not stable_features.empty:
        plt.figure(figsize=(10, max(8, len(stable_features) * 0.3)))
        
        # 将1D Series转换为2D DataFrame (列名为 'Selection Frequency')
        heatmap_data = stable_features.to_frame(name='Selection Frequency')
        
        sns.heatmap(heatmap_data, 
                    annot=True,          # 显示数值
                    cmap="YlGnBu",       # 选择一个合适的色板
                    fmt=".2f",           # 格式化数值为两位小数
                    linewidths=.5)
                    
        plt.title(f'Feature Selection Stability (Top {len(stable_features)} features over {n_iterations} iterations)')
        plt.xlabel('Frequency of being selected in Top 20')
        plt.ylabel('Feature Name')
        plt.tight_layout()
        
        if plot_save_path:
            plt.savefig(plot_save_path)
            print(f"特征稳定性热力图已保存至: {plot_save_path}")
        plt.close()
    else:
        print("没有稳定的特征可以绘制热图。")
    
    return stable_features
    
    # print("\n\n--- 特征选择频率 (稳定性排名) ---")
    # print(stable_features)
    
    # # --- 可视化为热图 ---
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(stable_features.to_frame(name='Selection Frequency'), annot=True, cmap="YlGnBu", fmt=".2f")
    # plt.title(f'Feature Selection Stability (Top {len(stable_features)} features over {n_iterations} iterations)')
    # plt.xlabel('Frequency of being selected in Top 20')
    # plt.ylabel('Feature Name')
    # plt.tight_layout()
    # plt.show()
    
    # return stable_features