# 文件名: train_no_window_advanced.py

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib
import warnings

# --- 信号处理与特征工程 ---
from scipy.stats import skew, kurtosis
from scipy.signal import hilbert, butter, filtfilt
from scipy.fft import fft
import pywt
from antropy import perm_entropy, higuchi_fd

# --- 机器学习与评估 ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier
from sklearn.svm import SVC

# --- 可视化 ---
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
#                      辅助函数定义区
# ==============================================================================

def calculate_fault_frequencies(rpm: float, bearing_type: str):
    """根据轴承类型(DE/FE)计算理论故障频率。"""
    BEARING_PARAMS = {
        'DE': {'n': 9, 'd': 0.3126, 'D': 1.537},
        'FE': {'n': 9, 'd': 0.2656, 'D': 1.122}
    }
    params = BEARING_PARAMS.get(bearing_type.upper())
    if not params: return None
    n, d, D = params['n'], params['d'], params['D']
    fr = rpm / 60.0
    bpfo = (n / 2.0) * fr * (1 - (d / D)); bpfi = (n / 2.0) * fr * (1 + (d / D))
    bsf = (D / (2.0 * d)) * fr * (1 - (d / D)**2)
    ftf = (1 / 2.0) * fr * (1 - (d / D))
    return {'Fr': fr, 'BPFO': bpfo, 'BPFI': bpfi, 'BSF': bsf, 'FTF': ftf}

def _extract_advanced_features(signal: np.ndarray, fs: int) -> dict:
    """提取小波包能量、排列熵、分形维数等高级特征。"""
    adv_features = {}
    
    # 1. 小波包能量 (WPT Energy) - 无改动
    wp = pywt.WaveletPacket(data=signal, wavelet='db4', mode='symmetric', maxlevel=3)
    nodes = wp.get_level(3, order='natural')
    for node in nodes:
        adv_features[f'wpt_energy_{node.path}'] = np.sum(np.square(node.data))
        
    # 2. 排列熵 (Permutation Entropy) - 无改动
    adv_features['complexity_perm_entropy'] = perm_entropy(signal, order=3, delay=1, normalize=True)
    
    # 3. 分形维数 (Fractal Dimension)
    # **核心修正**: 将参数名 k 修改为 kmax
    adv_features['complexity_higuchi_fd'] = higuchi_fd(signal, kmax=10)
    
    return adv_features

def extract_features_no_window(metadata_df: pd.DataFrame, signal_type: str = 'DE') -> pd.DataFrame:
    """【升级修正版】对每个完整的信号文件提取一套丰富的手工特征（基础+高级）。"""
    print(f"开始为每个完整的 {signal_type} 信号文件提取特征 (基础+高级)...")
    all_file_features = []
    path_column = f'{signal_type}_path'
    df_filtered = metadata_df.dropna(subset=[path_column]).copy()

    for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Extracting features per file"):
        signal_path = row[path_column]
        if pd.isna(signal_path) or not os.path.exists(signal_path): continue
        signal = pd.read_csv(signal_path, header=None).iloc[:, 0].values
        fs, rpm = row['Sampling_Rate'], row['RPM']
        
        features = {}
        
        # --- **核心修正**: 采用逐个赋值的方式避免KeyError ---
        # 1. 基础时域特征
        rms = np.sqrt(np.mean(signal**2))
        peak = np.max(np.abs(signal))
        
        features['time_rms'] = rms
        features['time_peak'] = peak
        features['time_variance'] = np.var(signal)
        features['time_skew'] = skew(signal)
        features['time_kurtosis'] = kurtosis(signal)
        # 现在可以安全地计算裕度因子
        features['time_crest_factor'] = peak / rms if rms > 1e-8 else 0
        
        # 2. 基础频域特征
        N = len(signal); yf = fft(signal); spectrum = 2.0/N * np.abs(yf[0:N//2])
        features['freq_mean'] = np.mean(spectrum)
        
        # 3. 基础包络谱特征
        nyquist = 0.5 * fs
        b, a = butter(3, [2000/nyquist, 5000/nyquist], btype='band'); filtered = filtfilt(b, a, signal)
        envelope = np.abs(hilbert(filtered)); N_env = len(envelope); yf_env = fft(envelope)
        env_spectrum = 2.0/N_env * np.abs(yf_env[0:N_env//2]); env_freq_axis = np.linspace(0.0, fs/2.0, N_env//2)
        fault_freqs = calculate_fault_frequencies(rpm, signal_type)
        if fault_freqs:
            for name, freq in fault_freqs.items():
                if name in ['BPFO', 'BPFI', 'BSF']:
                    indices = np.where(np.abs(env_freq_axis - freq) <= 5)[0]
                    features[f'env_{name}_peak'] = np.max(env_spectrum[indices]) if len(indices) > 0 else 0
        
        # 4. 提取高级特征
        advanced_features = _extract_advanced_features(signal, fs)
        features.update(advanced_features)
        
        # 5. 添加元数据
        features.update({'label': row['Fault_Type'], 'experiment_id': row['Experiment_ID']})
        all_file_features.append(features)
        
    return pd.DataFrame(all_file_features)

# ==============================================================================
#                                主执行流程
# ==============================================================================
if __name__ == '__main__':
    # --- 1. 设置路径 ---
    METADATA_CSV_PATH = r'C:\Users\JC\Desktop\shumo\datasets\output_results\metadata\metadata_summary.csv'
    FEATURES_NO_WINDOW_CSV_PATH = r'C:\Users\JC\Desktop\shumo\datasets\features_no_window_advanced.csv' # 新文件名
    
    # --- 2. 提取特征 (如果不使用窗口) ---
    if os.path.exists(FEATURES_NO_WINDOW_CSV_PATH):
        print(f"检测到已存在的特征文件 '{FEATURES_NO_WINDOW_CSV_PATH}'，直接加载。")
        features_df = pd.read_csv(FEATURES_NO_WINDOW_CSV_PATH)
    else:
        if not os.path.exists(METADATA_CSV_PATH):
            raise FileNotFoundError("错误: 元数据文件 metadata_summary.csv 不存在，请先运行主脚本生成。")
        metadata_df = pd.read_csv(METADATA_CSV_PATH)
        features_df = extract_features_no_window(metadata_df, signal_type='DE')
        features_df.to_csv(FEATURES_NO_WINDOW_CSV_PATH, index=False)
        print(f"特征已提取并保存至: {FEATURES_NO_WINDOW_CSV_PATH}")

    # --- 3. 准备数据用于训练 ---
    X = features_df.drop(columns=['label', 'experiment_id'])
    y_raw = features_df['label']
    
    le = LabelEncoder(); y = le.fit_transform(y_raw)
    
    # 标准随机划分 (因为是在文件级别操作)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    scaler = StandardScaler(); X_train = scaler.fit_transform(X_train_raw); X_test = scaler.transform(X_test_raw)
    
    # --- 4. 训练和评估模型 ---
    models_to_train = ['xgboost', 'svm']
    evaluation_results = []

    for model_type in models_to_train:
        print(f"\n{'='*25} 正在训练和评估 {model_type.upper()} (不使用窗口) {'='*25}")
        
        if model_type == 'xgboost':
            model = XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', random_state=42)
        else:
            model = SVC(kernel='rbf', probability=True, random_state=42, decision_function_shape='ovr')
            
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # --- **新增**: 生成详尽的评估报告 ---
        
        # 1. 定量评价
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        
        evaluation_results.append({
            "Model": model_type.upper(),
            "Accuracy": accuracy,
            "Precision (Macro)": precision,
            "Recall (Macro)": recall,
            "F1-score (Macro)": macro_f1
        })
        
        print("\n--- 分类报告 ---")
        print(classification_report(y_test, y_pred, target_names=le.classes_))
        
        # 2. 混淆矩阵热力图
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title(f'Confusion Matrix for {model_type.upper()} (No Windowing)')
        plt.xlabel('Predicted Label'); plt.ylabel('True Label'); plt.show()

        # 3. 类别指标柱状图
        report_dict = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        # 筛选出每个具体类别的数据
        class_metrics_df = report_df.loc[le.classes_, ['precision', 'recall', 'f1-score']]
        
        class_metrics_df.plot(kind='bar', figsize=(12, 6), rot=0)
        plt.title(f'Per-Class Metrics for {model_type.upper()} (No Windowing)')
        plt.ylabel("Score"); plt.grid(axis='y', linestyle='--'); plt.legend(loc='lower right')
        plt.tight_layout(); plt.show()
        
    # --- 5. 最终结果汇总 ---
    print(f"\n{'='*25} 最终定量评价汇总 {'='*25}")
    results_df = pd.DataFrame(evaluation_results)
    print(results_df.to_string())