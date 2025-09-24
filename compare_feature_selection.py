# 文件名: compare_feature_selection.py (最终严谨版)
# 描述: 集成异常值处理并使用分组交叉验证，严谨对比mRMR和PCA效果的脚本。

import os
import pandas as pd
import warnings
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupShuffleSplit # <--- 【核心修改1】导入正确的划分工具
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

# --- 导入您项目中已有的关键函数 ---
from utils_handle_outliers import handle_outliers_iqr
from utils_feature_selection import perform_feature_selection

# ==============================================================================
#                      1. 统一路径和参数设置 (与main.py一致)
# ==============================================================================
BASE_OUTPUT_DIR = r'C:\Users\JC\Desktop\shumo\datasets\output_results'
HANDCRAFTED_FEATURES_CSV_PATH = os.path.join(BASE_OUTPUT_DIR, 'features', 'handcrafted_features.csv')
SAVE_MODEL_DIR = os.path.join(BASE_OUTPUT_DIR, 'saved_models')
REPORTS_DIR = os.path.join(BASE_OUTPUT_DIR, 'reports')
MODEL_PLOTS_DIR = os.path.join(REPORTS_DIR, 'model_performance_plots')


def train_and_get_f1(X_train, y_train, X_test, y_test, model_type='xgboost'):
    """一个简化的训练评估函数，只返回F1分数。"""
    if model_type == 'xgboost':
        model = XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', random_state=42)
    else:
        raise NotImplementedError("此函数仅为XGBoost实现简化对比")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return f1_score(y_test, y_pred, average='macro')


def compare_mrmr_vs_pca_with_groups():
    """主执行函数，用于生成对比图表。"""
    print("--- 开始对比mRMR与PCA的特征选择效果 (使用分组划分) ---")
    warnings.filterwarnings("ignore", category=FutureWarning)
    os.makedirs(MODEL_PLOTS_DIR, exist_ok=True)

    # --- 步骤 1: 加载数据 ---
    if not os.path.exists(HANDCRAFTED_FEATURES_CSV_PATH):
        print(f"错误: 手工特征文件 '{HANDCRAFTED_FEATURES_CSV_PATH}' 不存在。")
        return
    
    print("加载手工特征...")
    handcrafted_features_df = pd.read_csv(HANDCRAFTED_FEATURES_CSV_PATH)

    # --- 步骤 2: 异常值处理 ---
    print("正在对源域数据进行异常值处理...")
    features_df_capped, _ = handle_outliers_iqr(handcrafted_features_df)
    print("异常值处理完成。")

    # --- 步骤 3: 特征选择 ---
    print("在处理过异常值的数据上执行特征选择流程...")
    selection_results = perform_feature_selection(
        features_df=features_df_capped, 
        n_features_mrmr=15, 
        n_components_pca=10, 
        save_dir=SAVE_MODEL_DIR
    )

    # --- 步骤 4: 准备数据 ---
    y = selection_results['labels']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_mrmr = selection_results['mrmr_selected_features']
    X_pca = selection_results['pca_transformed_features']
    
    # 【核心修改2】从处理过的数据中获取分组信息
    groups = features_df_capped['experiment_id']

    # --- 步骤 5: 使用GroupShuffleSplit进行训练和评估 ---
    
    # a. 评估 mRMR
    print("正在评估基于 mRMR 特征的模型...")
    gss_mrmr = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx_mrmr, test_idx_mrmr = next(gss_mrmr.split(X_mrmr, y_encoded, groups))
    
    X_train_mrmr_raw = X_mrmr.iloc[train_idx_mrmr]
    X_test_mrmr_raw = X_mrmr.iloc[test_idx_mrmr]
    y_train_mrmr = y_encoded[train_idx_mrmr]
    y_test_mrmr = y_encoded[test_idx_mrmr]
    
    scaler_for_mrmr = StandardScaler()
    X_train_mrmr_scaled = scaler_for_mrmr.fit_transform(X_train_mrmr_raw)
    X_test_mrmr_scaled = scaler_for_mrmr.transform(X_test_mrmr_raw)
    
    f1_mrmr = train_and_get_f1(X_train_mrmr_scaled, y_train_mrmr, X_test_mrmr_scaled, y_test_mrmr)
    print(f"mRMR 特征集的 F1-Score: {f1_mrmr:.4f}")

    # b. 评估 PCA
    print("正在评估基于 PCA 特征的模型...")
    gss_pca = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx_pca, test_idx_pca = next(gss_pca.split(X_pca, y_encoded, groups))

    X_train_pca = X_pca.iloc[train_idx_pca]
    X_test_pca = X_pca.iloc[test_idx_pca]
    y_train_pca = y_encoded[train_idx_pca]
    y_test_pca = y_encoded[test_idx_pca]

    f1_pca = train_and_get_f1(X_train_pca, y_train_pca, X_test_pca, y_test_pca)
    print(f"PCA 特征集的 F1-Score: {f1_pca:.4f}")

    # --- 步骤 6: 生成并保存对比图表 ---
    print("正在生成对比图表...")
    
    results_df = pd.DataFrame({
        'Feature Selection Method': ['mRMR', 'PCA'],
        'Macro F1-Score': [f1_mrmr, f1_pca]
    })
    
    plt.figure(figsize=(8, 6))
    barplot = sns.barplot(x='Feature Selection Method', y='Macro F1-Score', data=results_df, palette='viridis')
    
    for p in barplot.patches:
        barplot.annotate(format(p.get_height(), '.4f'), 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'center', 
                       xytext = (0, 9), 
                       textcoords = 'offset points',
                       fontsize=12)

    plt.title('Comparison of mRMR vs. PCA with Group Splitting (XGBoost)', fontsize=16)
    plt.xlabel('Feature Selection Method', fontsize=12)
    plt.ylabel('Macro F1-Score', fontsize=12)
    plt.ylim(0, max(1.0, f1_mrmr * 1.2, f1_pca * 1.2))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    save_path = os.path.join(MODEL_PLOTS_DIR, 'mrmr_vs_pca_comparison_groupsplit.png')
    plt.savefig(save_path)
    plt.close()
    
    print(f"\n对比图表已成功保存至: {save_path}")
    print("--- 对比完成 ---")


if __name__ == '__main__':
    compare_mrmr_vs_pca_with_groups()