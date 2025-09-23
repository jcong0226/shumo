# 文件名: utils_traditional_models.py (最终修正版)
import pandas as pd
import joblib
import os
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.svm import SVC
import shap

def train_and_evaluate_traditional_models(features_df: pd.DataFrame, selected_feature_names: list, 
                                          model_type: str, save_dir: str,
                                          report_save_dir: str, text_report_dir: str): # 新增路径参数
    """
    【最终修正版】加载手工特征，使用按组划分并遵循最佳实践的流程来训练、评估和保存模型。
    """
    print(f"\n--- 正在处理 {model_type.upper()} 模型 (使用按组划分) ---")
    
    # 准备数据
    X = features_df[selected_feature_names]
    y_raw = features_df['label']
    groups = features_df['experiment_id']

    # 创建独立的保存子目录o
    model_save_dir = os.path.join(save_dir, model_type.upper())
    os.makedirs(model_save_dir, exist_ok=True)

    # 1. 标签编码并保存编码器
    le = LabelEncoder()
    # **修正1**: 使用正确的变量名 y_raw
    y_encoded = le.fit_transform(y_raw)
    class_names = le.classes_
    joblib.dump(le, os.path.join(model_save_dir, 'label_encoder.joblib'))
    print(f"标签编码器已保存至: {model_save_dir}")

    # 2. **核心改动**: 使用 GroupShuffleSplit 先划分数据索引
    print("正在按 'experiment_id' 进行分组划分数据...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    # 注意：此时的X是未缩放的
    train_idx, test_idx = next(gss.split(X, y_encoded, groups))

    X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
    # ----------------临时加入这段代码进行验证---------------
    train_groups = groups.iloc[train_idx].unique()
    test_groups = groups.iloc[test_idx].unique()

    intersection = set(train_groups).intersection(set(test_groups))

    print(f"训练集中的独立实验ID数量: {len(train_groups)}")
    print(f"测试集中的独立实验ID数量: {len(test_groups)}")
    print(f"训练集和测试集共享的实验ID: {intersection}")

    if len(intersection) == 0:
        print("验证成功：训练集和测试集之间没有重叠的实验ID，分组划分正确！")
    else:
        print("验证失败：训练集和测试集之间存在数据泄漏！")
    # -----------------------------------------------------
    # 3. **核心改动**: 特征缩放 (遵循最佳实践)
    scaler = StandardScaler()
    # 只在训练集上 fit_transform
    X_train = scaler.fit_transform(X_train_raw)
    # 在测试集上只进行 transform
    X_test = scaler.transform(X_test_raw)
    
    # 保存已在训练集上拟合好的缩放器
    joblib.dump(scaler, os.path.join(model_save_dir, 'scaler.joblib'))
    print(f"数据缩放器 (已在训练集上拟合) 已保存至: {model_save_dir}")
    
    # 4. 初始化和训练模型
    if model_type == 'xgboost':
        # XGBoost < 2.0.0 an use_label_encoder=False, later versions remove it.
        # This code is compatible with both.
        model = XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', random_state=42)
    elif model_type == 'svm':
        model = SVC(kernel='rbf', probability=True, random_state=42)
    else:
        raise ValueError("模型类型必须是 'xgboost' 或 'svm'")
    
    print("正在训练模型...")
    model.fit(X_train, y_train)
    
    # 5. 保存训练好的模型
    model_path = os.path.join(model_save_dir, 'model.joblib')
    joblib.dump(model, model_path)
    print(f"训练好的模型已保存至: {model_path}")

    # 6. 模型评估
    y_pred = model.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
        # --- 【修改】保存分类报告到文本文件 ---
    report = f"模型 {model_type.upper()} 的宏 F1 分数: {macro_f1:.4f}\n\n"
    report += classification_report(y_test, y_pred, target_names=class_names)
    print("\n" + report)
    
    report_filename = os.path.join(text_report_dir, f'classification_report_{model_type}.txt')
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"分类报告已保存至: {report_filename}")

    # --- 【修改】保存混淆矩阵图像 ---
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'{model_type.upper()} Confusion Matrix')
    
    cm_path = os.path.join(report_save_dir, f'confusion_matrix_{model_type}.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"混淆矩阵已保存至: {cm_path}")

    # --- 【修改】保存SHAP图 ---
    if model_type == 'xgboost':
        # ... (计算 SHAP 值的逻辑不变) ...
        print("\n正在计算 SHAP 值以解释模型...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        # 绘制 SHAP summary plot
        shap.summary_plot(shap_values, X_test, feature_names=selected_feature_names,
                        class_names=class_names, show=False)
        plt.title(f'SHAP Summary Plot for {model_type.upper()}')
        plt.tight_layout()
        shap_path = os.path.join(report_save_dir, 'shap_summary_plot.png')
        plt.savefig(shap_path)
        plt.close()
        print(f"SHAP图已保存至: {shap_path}")
    
    # print(f"\n模型 {model_type.upper()} 的宏 F1 分数: {macro_f1:.4f}")
    # print(classification_report(y_test, y_pred, target_names=class_names))

    # cm = confusion_matrix(y_test, y_pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    # disp.plot(cmap=plt.cm.Blues)
    # plt.title(f'{model_type.upper()} Confusion Matrix')
    # plt.show()
    # if model_type == 'xgboost':
    #     print("\n正在计算 SHAP 值以解释模型...")
    #     explainer = shap.TreeExplainer(model)
    #     shap_values = explainer.shap_values(X_test)
    #     # 绘制 SHAP summary plot
    #     shap.summary_plot(shap_values, X_test, feature_names=selected_feature_names,
    #                     class_names=class_names, show=False)
    #     plt.title(f'SHAP Summary Plot for {model_type.upper()}')
    #     plt.tight_layout()
    #     plt.show()