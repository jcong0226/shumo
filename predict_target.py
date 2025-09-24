import pandas as pd
import joblib
import os
from utils_transfer_preparation import aggregate_window_predictions
import numpy as np # 导入 numpy

def predict_and_label_target_data():
    """
    加载迁移模型，对目标域数据进行预测、聚合，并输出最终的文件级标签及置信度。
    """
    # --- 1. 定义路径 (保持不变) ---
    BASE_OUTPUT_DIR = r'C:\Users\JC\Desktop\shumo\datasets\output_results'
    SAVE_MODEL_DIR = os.path.join(BASE_OUTPUT_DIR, 'saved_models')
    TARGET_FEATURES_PATH = os.path.join(BASE_OUTPUT_DIR, 'features', 'target_handcrafted_features.csv')
    FINAL_RESULTS_PATH = os.path.join(BASE_OUTPUT_DIR, 'reports', 'target_diagnosis_results_with_confidence.csv') # 新的保存文件名

    # --- 2. 加载模型 (保持不变) ---
    model_dir = os.path.join(SAVE_MODEL_DIR, 'transfer_model_xgboost', 'XGBOOST')
    
    print(f"从 '{model_dir}' 加载模型...")
    model = joblib.load(os.path.join(model_dir, 'model.joblib'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
    label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.joblib'))
    print("模型和预处理器加载成功。")

    # --- 3. 加载目标域特征数据 (保持不变) ---
    target_features_df = pd.read_csv(TARGET_FEATURES_PATH)

    # --- 4. 准备预测数据 ---
    #    【核心】必须使用模型训练时所用的那一组特征
    #    XGBoost模型保存了特征名称，我们可以直接获取
    feature_names = [
        'time_rms', 'env_BPFO_3x_ratio', 'env_BPFI_2x_ratio', 
        'time_moment_6', 'env_2BSF_1x_ratio', 'time_skew', 'time_moment_5', 'time_peak', 
        'time_crest_factor', 'time_clearance_factor', 'time_impulse_factor', 'time_kurtosis','time_p2p',
        'freq_skew', 'time_shape_factor', 'time_mean', 'env_2BSF_3x_ratio', 'time_mean_abs',
        'env_2BSF_2x_amp', 'env_2BSF_2x_ratio', 'env_BPFO_1x_ratio', 'freq_kurtosis', 'env_BPFI_1x_ratio',
        'env_2BSF_3x_amp', 'env_BPFO_3x_amp', 'env_BPFO_2x_ratio', 'env_BPFI_3x_ratio', 'wpt_energy_node_0',
        'segment_index', 'env_BPFI_2x_amp', 'freq_rmsf', 'env_2BSF_1x_amp', 'freq_std', 'env_BPFO_1x_amp'
        ]
    # feature_names = [
    #     'time_rms','env_BPFO_3x_ratio', 'env_BPFI_2x_ratio', 
    #     'time_moment_6', 'env_2BSF_1x_ratio', 'time_skew', 'time_moment_5', 'time_peak', 
    #     'time_crest_factor', 'time_clearance_factor', 'time_impulse_factor', 'time_kurtosis','time_p2p',
    #     'freq_skew', 
    #     ]
    print(f"\n模型使用的特征 ({len(feature_names)}个): \n{feature_names}")
    X_target_raw = target_features_df[feature_names]
    X_target_scaled = scaler.transform(X_target_raw)

    # --- 5. 【核心修改点 1】执行窗口级预测，获取概率 ---
    #    使用 model.predict_proba() 而不是 model.predict()
    #    它返回一个数组，每行是对应样本的概率分布，列顺序与 label_encoder.classes_ 一致
    window_probabilities = model.predict_proba(X_target_scaled)

    #    将概率数组转换为带有清晰列名的DataFrame
    prob_cols = [f"prob_{cls}" for cls in label_encoder.classes_]
    window_probs_df = pd.DataFrame(window_probabilities, columns=prob_cols)

    # --- 6. 【核心修改点 2】将窗口级预测聚合成文件级诊断结果 ---
    #    创建一个包含原始路径和所有概率列的DataFrame
    window_results_df = pd.concat([
        target_features_df[['original_path']],
        window_probs_df
    ], axis=1)

    #    调用您代码中已有的聚合函数
    final_diagnosis_df = aggregate_window_predictions(
        window_results_df, 
        method='average_probability'
    )

    # --- 7. 【核心修改点 3】显示并保存包含置信度的最终结果 ---
    print("\n" + "="*25 + " 目标域文件最终诊断标签与置信度 " + "="*25)
    
    # 从完整路径中提取A-P的文件名
    final_diagnosis_df['filename'] = final_diagnosis_df['filename'].str.replace(r'\.csv$', '', regex=True)
    
    # 找到最终诊断结果对应的置信度（即该类别的平均概率）
    # final_diagnosis_df.lookup() 在新版pandas中已弃用，我们使用更现代的方法
    confidence_scores = []
    for index, row in final_diagnosis_df.iterrows():
        diagnosis = row['final_diagnosis']
        prob_column_name = f'prob_{diagnosis}'
        confidence = row[prob_column_name]
        confidence_scores.append(confidence)

    final_diagnosis_df['confidence'] = confidence_scores

    # 调整列顺序，使其更清晰，并格式化输出
    # 我们只保留最重要的列：文件名、最终诊断、置信度
    output_df = final_diagnosis_df[['filename', 'final_diagnosis', 'confidence']]
    # 也可以显示所有类别的概率
    # output_df = final_diagnosis_df[['filename', 'final_diagnosis', 'confidence'] + prob_cols]

    # 格式化置信度为百分比
    output_df['confidence'] = output_df['confidence'].map('{:.2%}'.format)

    print(output_df.to_string(index=False))
    
    # 保存结果到CSV文件
    final_diagnosis_df.to_csv(FINAL_RESULTS_PATH, index=False)
    print(f"\n包含置信度的诊断结果已保存至: {FINAL_RESULTS_PATH}")

if __name__ == '__main__':
    predict_and_label_target_data()