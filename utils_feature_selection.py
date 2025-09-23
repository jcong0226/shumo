# 文件名: utils_feature_selection.py (最终修正版)
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import mrmr
import joblib
import os
from sklearn.impute import SimpleImputer

def perform_feature_selection(features_df: pd.DataFrame, n_features_mrmr: int = 20, 
                              n_components_pca: int = 10, save_dir: str = None):
    """
    对提取的手工特征执行一个完整的特征选择和降维流程。
    """
    print(f"\n--- 开始特征选择与降维 ---")
    
    # 1. 分离特征 (X) 和标签 (y)
    X = features_df.drop(columns=['label', 'experiment_id', 'original_path'], errors='ignore')
    y = features_df['label']

    # --- 【核心修正】无条件创建并应用Imputer ---
    print("应用均值插补策略处理任何可能的缺失值...")
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)

    print(f"原始特征数量: {len(X.columns)}")

    # 2. 方差阈值
    selector_var = VarianceThreshold(threshold=0.01)
    X_var_selected = selector_var.fit_transform(X)
    selected_cols_var = X.columns[selector_var.get_support()]
    print(f"经过方差阈值筛选后，剩余特征数量: {X_var_selected.shape[1]}")

    # 3. mRMR
    X_var_df = pd.DataFrame(X_var_selected, columns=selected_cols_var)
    selected_features_mrmr = mrmr.mrmr_classif(X=X_var_df, y=y, K=n_features_mrmr)
    X_mrmr = X_var_df[selected_features_mrmr]
    print(f"\n使用mRMR选择出的前 {n_features_mrmr} 个特征:")
    print(selected_features_mrmr)

    # 4. PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X) # 在插补后的完整X上进行缩放
    
    pca = PCA(n_components=n_components_pca)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"\nPCA降维完成，得到 {n_components_pca} 个主成分。")
    print(f"前 {n_components_pca} 个主成分解释的总方差比例: {sum(pca.explained_variance_ratio_):.2%}")

    # 5. 保存所有预处理模型
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        # 【核心修正】无条件保存imputer
        joblib.dump(imputer, os.path.join(save_dir, 'source_domain_imputer.joblib'))
        joblib.dump(scaler, os.path.join(save_dir, 'source_domain_scaler.joblib'))
        joblib.dump(pca, os.path.join(save_dir, 'source_domain_pca.joblib'))
        print(f"\n源域Imputer, Scaler和PCA模型已保存至: {save_dir}")

    return {
        "mrmr_selected_features": X_mrmr,
        "pca_transformed_features": pd.DataFrame(X_pca, columns=[f'PC_{i+1}' for i in range(n_components_pca)]),
        "labels": y
    }