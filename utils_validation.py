# 建议新增一个 'utils_validation.py'
from sklearn.model_selection import GroupKFold
import numpy as np

def cross_validate_by_condition(model, features_df, selected_feature_names, condition_col='RPM'):
    """按工况进行交叉验证"""
    X = features_df[selected_feature_names]
    y = features_df['label'] # 假设已编码
    groups = features_df[condition_col]
    unique_conditions = np.unique(groups)

    print(f"\n--- 开始按 '{condition_col}' 进行留一法交叉验证 ---")
    scores = []
    for condition in unique_conditions:
        train_idx = np.where(groups != condition)[0]
        test_idx = np.where(groups == condition)[0]

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # ... (此处省略标准化和模型训练/评估流程) ...
        # model.fit(X_train_scaled, y_train)
        # score = model.score(X_test_scaled, y_test)
        # print(f"测试工况: {condition}, F1-Score: {score:.4f}")
        # scores.append(score)
    # print(f"平均跨工况F1-Score: {np.mean(scores):.4f}")