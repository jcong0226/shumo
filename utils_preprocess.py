# 文件名: utils_preprocess.py
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.signal import resample, resample_poly, detrend


def preprocess_from_metadata(metadata_df: pd.DataFrame, signal_type: str, target_fs: int, window_size: int, step_size: int) -> list:
    """根据元数据DataFrame进行完整的预处理流程。"""
    processed_samples = []
    path_column = f'{signal_type.upper()}_path'
    if path_column not in metadata_df.columns: raise ValueError(f"列 '{path_column}' 不在DataFrame中！")
    print(f"开始基于元数据预处理 '{signal_type}' 信号...")
    for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc=f"Preprocessing {signal_type}"):
        signal_path = row[path_column]
        if pd.isna(signal_path): continue
        try:
            original_signal = pd.read_csv(signal_path, header=None).iloc[:, 0].values
            original_fs = row['Sampling_Rate']
            if original_fs != target_fs:
                up = target_fs
                down = original_fs
                resampled_signal = resample_poly(original_signal, up, down)
                # num_new_samples = int(len(original_signal) * target_fs / original_fs)
                # resampled_signal = resample(original_signal, num_new_samples)
            else: 
                resampled_signal = original_signal
            num_segments = (len(resampled_signal) - window_size) // step_size + 1
            for i in range(num_segments):
                segment = resampled_signal[i * step_size : i * step_size + window_size]
                segment = detrend(segment, type='linear')
                mean, std = np.mean(segment), np.std(segment)
                if std > 1e-8: normalized_segment = (segment - mean) / std
                else: normalized_segment = segment - mean
                processed_samples.append({'signal_segment': normalized_segment, 'label': row['Fault_Type'],
                                          'experiment_id': row['Experiment_ID'], 'fault_size': row['Fault_Size_Inch'],
                                          'load': row['Load_HP'], 'rpm': row['RPM'], 'original_path': signal_path,
                                          'segment_index': i})
        except FileNotFoundError: print(f"警告: 文件未找到 {signal_path}，已跳过。")
        except Exception as e: print(f"处理文件 {signal_path} 时发生错误: {e}")
    print(f"预处理完成！共生成 {len(processed_samples)} 个 '{signal_type}' 信号样本。")
    return processed_samples