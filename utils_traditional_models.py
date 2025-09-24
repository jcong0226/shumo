# 文件名: utils_traditional_models.py (最终修正版 - 已修复KeyError)
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.svm import SVC
import shap

# 导入我们之前创建的分组分层抽样函数
from utils_stratified_split import stratified_group_split

def train_and_evaluate_traditional_models(features_df: pd.DataFrame, selected_feature_names: list,
                                          model_type: str, save_dir: str,
                                          report_save_dir: str, text_report_dir: str):
    """
    【最终修正版】加载手工特征，使用按组划分并遵循最佳实践的流程来训练、评估和保存模型。
    """
    print(f"\n--- 正在处理 {model_type.upper()} 模型 (使用分组分层抽样) ---")

    # 准备数据
    X = features_df[selected_feature_names]
    y_raw = features_df['label']
    groups = features_df['experiment_id']

    # 创建独立的保存子目录
    model_save_dir = os.path.join(save_dir, model_type.upper())
    os.makedirs(model_save_dir, exist_ok=True)

    # 1. 标签编码
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    class_names = le.classes_
    joblib.dump(le, os.path.join(model_save_dir, 'label_encoder.joblib'))
    print(f"标签编码器已保存至: {model_save_dir}")

    # 【核心修改点 1】: 将编码后的标签添加到DataFrame副本中，以便进行分层抽样
    # 这样做可以避免直接修改传入的DataFrame，是更安全的操作
    df_for_split = features_df.copy()
    df_for_split['label_encoded'] = y_encoded

    # 2. **核心修改点 2**: 使用新的 stratified_group_split 函数进行数据划分
    train_df, test_df = stratified_group_split(
        features_df=df_for_split,
        group_col='experiment_id',
        stratify_col='label', # 依然使用原始标签列进行分层
        test_size=0.3,
        random_state=42
    )

    # 从划分好的DataFrame中提取 X 和 y
    X_train_raw = train_df[selected_feature_names]
    y_train = train_df['label_encoded'].values
    X_test_raw = test_df[selected_feature_names]
    y_test = test_df['label_encoded'].values
    
    # 验证分组划分是否正确 (可选，但建议保留)
    train_groups = train_df['experiment_id'].unique()
    test_groups = test_df['experiment_id'].unique()
    intersection = set(train_groups).intersection(set(test_groups))
    if len(intersection) == 0:
        print("验证成功：训练集和测试集之间没有重叠的实验ID，分组划分正确！")
    else:
        print(f"验证失败：训练集和测试集之间存在数据泄漏！共享ID: {intersection}")

    # 3. 特征缩放 (只在训练集上 fit_transform)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    joblib.dump(scaler, os.path.join(model_save_dir, 'scaler.joblib'))
    print(f"数据缩放器 (已在训练集上拟合) 已保存至: {model_save_dir}")

    # 4. 初始化和训练模型
    if model_type == 'xgboost':
        model = XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', random_state=42)
    elif model_type == 'svm':
        model = SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')
    else:
        raise ValueError("模型类型必须是 'xgboost' 或 'svm'")

    # 处理类别不平衡问题
    if model_type == 'xgboost':
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
        model.fit(X_train, y_train, sample_weight=sample_weights)
    else: # SVC通过class_weight='balanced'参数自动处理
        model.fit(X_train, y_train)

    # 5. 保存训练好的模型
    model_path = os.path.join(model_save_dir, 'model.joblib')
    joblib.dump(model, model_path)
    print(f"训练好的模型已保存至: {model_path}")

    # 6. 模型评估和报告生成 (此部分逻辑不变)
    y_pred = model.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    report = f"模型 {model_type.upper()} 的宏 F1 分数: {macro_f1:.4f}\n\n"
    report += classification_report(y_test, y_pred, target_names=class_names)
    print("\n" + report)

    report_filename = os.path.join(text_report_dir, f'classification_report_{model_type}.txt')
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"分类报告已保存至: {report_filename}")

    # 保存混淆矩阵图像
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'{model_type.upper()} Confusion Matrix')
    cm_path = os.path.join(report_save_dir, f'confusion_matrix_{model_type}.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"混淆矩阵已保存至: {cm_path}")

    # 保存SHAP图
    if model_type == 'xgboost':
        print("\n正在计算 SHAP 值以解释模型...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test, feature_names=selected_feature_names,
                        class_names=class_names, show=False)
        plt.title(f'SHAP Summary Plot for {model_type.upper()}')
        plt.tight_layout()
        shap_path = os.path.join(report_save_dir, 'shap_summary_plot.png')
        plt.savefig(shap_path)
        plt.close()
        print(f"SHAP图已保存至: {shap_path}")