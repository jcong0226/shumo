import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import mrmr
import joblib
import os
from sklearn.impute import SimpleImputer
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

def perform_feature_selection(features_df: pd.DataFrame, n_features_mrmr: int = 20, 
                              n_components_pca: int = 10, save_dir: str = None):
    """
    对提取的手工特征执行一个完整的特征选择和降维流程。
    针对轴承故障诊断优化，增强可解释性和迁移学习效果。
    """
    print(f"\n--- 开始轴承故障特征选择与降维 ---")
    
    # 1. 分离特征 (X) 和标签 (y)
    X = features_df.drop(columns=['label', 'experiment_id', 'original_path'], errors='ignore')
    y = features_df['label']

    # --- 【核心修正】无条件创建并应用Imputer ---
    print("应用均值插补策略处理任何可能的缺失值...")
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)

    print(f"原始特征数量: {len(X.columns)}")
    
    # 【新增】特征类型分组分析（针对轴承故障特征）
    time_domain_features = [col for col in X.columns if any(td in col.lower() 
                          for td in ['rms', 'std', 'mean', 'peak', 'kurtosis', 'skewness', 'crest'])]
    freq_domain_features = [col for col in X.columns if any(fd in col.lower() 
                          for fd in ['freq', 'psd', 'spectral', 'band'])]
    
    print(f"时域特征数量: {len(time_domain_features)}")
    print(f"频域特征数量: {len(freq_domain_features)}")

    # 2. 方差阈值（针对轴承信号调整阈值）
    # 轴承故障信号中，微小的方差也可能包含重要信息
    selector_var = VarianceThreshold(threshold=0.001)  # 降低阈值
    X_var_selected = selector_var.fit_transform(X)
    selected_cols_var = X.columns[selector_var.get_support()]
    print(f"经过方差阈值筛选后，剩余特征数量: {X_var_selected.shape[1]}")

    # 3. mRMR - 轴承故障诊断的核心特征选择
    X_var_df = pd.DataFrame(X_var_selected, columns=selected_cols_var)
    
    # 【优化】增加mRMR参数调整，适应轴承故障特征
    try:
        selected_features_mrmr = mrmr.mrmr_classif(
            X=X_var_df, y=y, K=n_features_mrmr,
            relevance='f',  # 使用F统计量衡量相关性
            redundancy='c'   # 使用相关系数衡量冗余性
        )
        X_mrmr = X_var_df[selected_features_mrmr]
        
        print(f"\n使用mRMR选择出的前 {n_features_mrmr} 个轴承故障关键特征:")
        for i, feature in enumerate(selected_features_mrmr[:10]):  # 显示前10个
            print(f"{i+1:2d}. {feature}")
        
        # 【新增】分析选中特征的类型分布
        selected_time_features = [f for f in selected_features_mrmr if f in time_domain_features]
        selected_freq_features = [f for f in selected_features_mrmr if f in freq_domain_features]
        
        print(f"\nmRMR选中的特征类型分布:")
        print(f"  时域特征: {len(selected_time_features)} 个")
        print(f"  频域特征: {len(selected_freq_features)} 个")
        print(f"  其他特征: {len(selected_features_mrmr) - len(selected_time_features) - len(selected_freq_features)} 个")
        
    except Exception as e:
        print(f"mRMR执行出错: {e}")
        print("使用备用方案：基于相关性的特征选择")
        # 备用方案：使用相关性分析
        correlations = []
        for col in X_var_df.columns:
            corr, _ = pearsonr(X_var_df[col], y.astype(int))
            correlations.append((col, abs(corr)))
        correlations.sort(key=lambda x: x[1], reverse=True)
        selected_features_mrmr = [item[0] for item in correlations[:n_features_mrmr]]
        X_mrmr = X_var_df[selected_features_mrmr]

    # 4. PCA（主要用于对比和可视化）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # 在插补后的完整X上进行缩放
    
    pca = PCA(n_components=n_components_pca)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"\nPCA降维完成，得到 {n_components_pca} 个主成分。")
    print(f"前 {n_components_pca} 个主成分解释的总方差比例: {sum(pca.explained_variance_ratio_):.2%}")
    print("各主成分方差贡献比:")
    for i, ratio in enumerate(pca.explained_variance_ratio_[:5]):  # 显示前5个
        print(f"  PC{i+1}: {ratio:.3f} ({ratio*100:.1f}%)")

    # 【新增】特征重要性可视化保存
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存mRMR特征重要性
        mrmr_importance = pd.DataFrame({
            'Feature': selected_features_mrmr,
            'Rank': range(1, len(selected_features_mrmr) + 1)
        })
        mrmr_importance.to_csv(os.path.join(save_dir, 'mrmr_feature_ranking.csv'), index=False)
        
        # 保存PCA解释方差比
        pca_variance = pd.DataFrame({
            'Component': [f'PC{i+1}' for i in range(n_components_pca)],
            'Variance_Ratio': pca.explained_variance_ratio_,
            'Cumulative_Ratio': np.cumsum(pca.explained_variance_ratio_)
        })
        pca_variance.to_csv(os.path.join(save_dir, 'pca_variance_explained.csv'), index=False)

    # 5. 保存所有预处理模型
    if save_dir:
        # 【核心修正】无条件保存imputer
        joblib.dump(imputer, os.path.join(save_dir, 'source_domain_imputer.joblib'))
        joblib.dump(scaler, os.path.join(save_dir, 'source_domain_scaler.joblib'))
        joblib.dump(pca, os.path.join(save_dir, 'source_domain_pca.joblib'))
        
        # 保存方差选择器
        joblib.dump(selector_var, os.path.join(save_dir, 'source_domain_variance_selector.joblib'))
        
        # 保存mRMR选择的特征名称
        with open(os.path.join(save_dir, 'mrmr_selected_features.txt'), 'w', encoding='utf-8') as f:
            for feature in selected_features_mrmr:
                f.write(f"{feature}\n")
        
        print(f"\n源域预处理模型和特征选择结果已保存至: {save_dir}")

    # 【新增】返回更详细的结果，便于后续分析
    results = {
        "mrmr_selected_features": X_mrmr,
        "pca_transformed_features": pd.DataFrame(X_pca, columns=[f'PC_{i+1}' for i in range(n_components_pca)]),
        "labels": y,
        # 额外信息，便于分析和论文写作
        "feature_analysis": {
            "mrmr_feature_names": selected_features_mrmr,
            "time_domain_selected": [f for f in selected_features_mrmr if f in time_domain_features],
            "freq_domain_selected": [f for f in selected_features_mrmr if f in freq_domain_features],
            "pca_variance_ratio": pca.explained_variance_ratio_,
            "total_features_before": len(X.columns),
            "features_after_variance": X_var_selected.shape[1],
            "features_after_mrmr": len(selected_features_mrmr)
        }
    }
    
    return results