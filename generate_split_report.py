# 文件名: enhanced_subject_1/generate_split_report.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from utils_stratified_split import stratified_group_split

def generate_report():
    """
    加载特征数据，执行分组分层抽样，并生成数据分布报告和对比图。
    """
    # --- 路径设置 ---
    BASE_OUTPUT_DIR = r'C:\Users\JC\Desktop\shumo\datasets\output_results'
    HANDCRAFTED_FEATURES_CSV_PATH = os.path.join(BASE_OUTPUT_DIR, 'features', 'handcrafted_features.csv')
    REPORTS_DIR = os.path.join(BASE_OUTPUT_DIR, 'reports')
    MODEL_PLOTS_DIR = os.path.join(REPORTS_DIR, 'model_performance_plots')
    os.makedirs(MODEL_PLOTS_DIR, exist_ok=True)

    # --- 加载数据 ---
    if not os.path.exists(HANDCRAFTED_FEATURES_CSV_PATH):
        print(f"错误: 特征文件不存在于 '{HANDCRAFTED_FEATURES_CSV_PATH}'")
        return
    features_df = pd.read_csv(HANDCRAFTED_FEATURES_CSV_PATH)

    # --- 执行分组分层抽样 ---
    train_df, test_df = stratified_group_split(
        features_df=features_df,
        group_col='experiment_id',
        stratify_col='label',
        test_size=0.3,
        random_state=42
    )

    # --- 生成并打印数据分布报告 ---
    print("\n" + "="*50)
    print("           数据分布对比报告")
    print("="*50)

    # 1. 计算原始分布
    original_dist = features_df['label'].value_counts().reset_index()
    original_dist.columns = ['Fault_Type', 'Count']
    original_dist['Percentage'] = (original_dist['Count'] / original_dist['Count'].sum() * 100).round(2)
    original_dist['Dataset'] = 'Original'

    # 2. 计算训练集分布
    train_dist = train_df['label'].value_counts().reset_index()
    train_dist.columns = ['Fault_Type', 'Count']
    train_dist['Percentage'] = (train_dist['Count'] / train_dist['Count'].sum() * 100).round(2)
    train_dist['Dataset'] = 'Train Set'

    # 3. 计算测试集分布
    test_dist = test_df['label'].value_counts().reset_index()
    test_dist.columns = ['Fault_Type', 'Count']
    test_dist['Percentage'] = (test_dist['Count'] / test_dist['Count'].sum() * 100).round(2)
    test_dist['Dataset'] = 'Test Set'

    # 合并报告
    full_report = pd.concat([original_dist, train_dist, test_dist])
    
    print("\n按样本数统计：")
    print(full_report.pivot(index='Fault_Type', columns='Dataset', values='Count').fillna(0).astype(int))
    
    print("\n按百分比统计(%)：")
    print(full_report.pivot(index='Fault_Type', columns='Dataset', values='Percentage').fillna(0))
    print("="*50)

    # --- 生成并保存对比图表 ---
    plt.figure(figsize=(12, 7))
    sns.barplot(x='Fault_Type', y='Percentage', hue='Dataset', data=full_report, palette='viridis')
    plt.title('Comparison of Data Distribution Before and After Stratified Group Split', fontsize=16)
    plt.ylabel('Percentage of Samples (%)', fontsize=12)
    plt.xlabel('Fault Type', fontsize=12)
    plt.legend(title='Dataset Split')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    save_path = os.path.join(MODEL_PLOTS_DIR, 'stratified_split_distribution_comparison.png')
    plt.savefig(save_path)
    plt.close()
    
    print(f"\n数据分布对比图已成功保存至: {save_path}")

if __name__ == '__main__':
    generate_report()