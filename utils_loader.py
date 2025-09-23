# 文件名: utils_loader.py
import os
import re
import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy.signal import welch

def calculate_snr(signal, fs, signal_freq_range, noise_freq_range=(10, 100)):
    """
    使用Welch方法估计信号和噪声功率，计算SNR。
    这是一个简化的例子，实际中可能需要更复杂的噪声估计。
    """
    freqs, psd = welch(signal, fs, nperseg=1024)
    signal_power = np.mean(psd[(freqs >= signal_freq_range[0]) & (freqs <= signal_freq_range[1])])
    noise_power = np.mean(psd[(freqs >= noise_freq_range[0]) & (freqs <= noise_freq_range[1])])

    if noise_power < 1e-10: return 99.0 # 避免除零
    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db

def create_structured_dataset(root_path: str, save_path: str = None) -> pd.DataFrame:
    """【最终版】扫描CWRU数据集的目录结构，解析元数据，并创建一个包含所有信息的结构化DataFrame。"""
    label_mapping = {'IR': 'Inner_Race', 'B': 'Ball', 'OR': 'Outer_Race', 'Normal': 'Normal', 'N': 'Normal'}
    or_position_mapping = {'@3': 'Orthogonal_3oclock', '@6': 'Centered_6oclock', '@12': 'Opposite_12oclock'}
    experiments = {}
    print("阶段一：扫描文件并按实验分组...")
    for root, _, files in tqdm(os.walk(root_path), desc="Scanning files"):
        for filename in files:
            match = re.search(r'_X(\d+)', filename)
            if not match:
                if 'Normal' in root or '_N_' in filename:
                    base_id = os.path.splitext(filename)[0].replace('_DE_time','').replace('_FE_time','').replace('_BA_time','').replace('RPM','')
                    exp_id = f"Normal_{base_id}"
                else: continue
            else: exp_id = f"X{match.group(1)}"
            if exp_id not in experiments: experiments[exp_id] = {'root': root}
            if filename.endswith('_DE_time.csv'): experiments[exp_id]['de_file'] = filename
            elif filename.endswith('_FE_time.csv'): experiments[exp_id]['fe_file'] = filename
            elif filename.endswith('_BA_time.csv'): experiments[exp_id]['ba_file'] = filename
            elif 'RPM.csv' in filename: experiments[exp_id]['rpm_file'] = filename
    all_experiments_data = []
    print("\n阶段二：解析每个实验的元数据...")
    for exp_id, data in tqdm(experiments.items(), desc="Parsing metadata"):
        main_file = data.get('de_file') or data.get('fe_file') or data.get('ba_file')
        if not main_file: continue
        if '_X' in main_file: fault_def_part = main_file.split('_X')[0]
        else: fault_def_part = main_file.split('(')[0].rstrip('_')
        match = re.match(r'([A-Za-z]+)(\d{3})?(@\d+)?(?:_(\d))?', fault_def_part)
        if not match: continue
        fault_type_key, size_str, pos_key, load_str = match.groups()
        rpm = None
        rpm_match = re.search(r'\((\d+)rpm\)', main_file)
        if rpm_match: rpm = int(rpm_match.group(1))
        elif 'rpm_file' in data:
            try:
                rpm_path = os.path.join(data['root'], data['rpm_file'])
                rpm = pd.read_csv(rpm_path, header=None).iloc[0, 0]
            except (KeyError, FileNotFoundError): pass
        sampling_rate = 12000 if '12k' in data['root'] else (48000 if '48k' in data['root'] else 0)
        row = {'Experiment_ID': exp_id, 'Fault_Type': label_mapping.get(fault_type_key, 'Unknown'),
               'Fault_Size_Inch': float(f"0.0{size_str}") if size_str else 0.0,
               'Load_HP': int(load_str) if load_str else 0, 'OR_Position': or_position_mapping.get(pos_key, None),
               'RPM': rpm, 'Sampling_Rate': sampling_rate,
               'DE_path': os.path.join(data['root'], data.get('de_file')) if 'de_file' in data else None,
               'FE_path': os.path.join(data['root'], data.get('fe_file')) if 'fe_file' in data else None,
               'BA_path': os.path.join(data['root'], data.get('ba_file')) if 'ba_file' in data else None,
               'RPM_path': os.path.join(data['root'], data.get('rpm_file')) if 'rpm_file' in data else None}
        all_experiments_data.append(row)
    df = pd.DataFrame(all_experiments_data)
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"\n结构化数据集已成功保存到: {save_path}")
    return df

# 文件名: utils_loader.py (在文件末尾追加)

def create_target_structured_dataset(root_path: str, save_path: str = None) -> pd.DataFrame:
    """
    为无标签的目标域数据集（A-P）创建结构化的元数据DataFrame。

    Args:
        root_path (str): 目标域数据的根目录。
        save_path (str, optional): 保存CSV文件的路径. Defaults to None.

    Returns:
        pd.DataFrame: 包含目标域元数据和文件路径的DataFrame。
    """
    print("正在为目标域数据创建元数据索引...")
    target_data_list = []
    
    for filename in tqdm(sorted(os.listdir(root_path)), desc="Processing target files"):
        if filename.lower().endswith(('.csv', '.mat')): # 支持多种格式
            exp_id = os.path.splitext(filename)[0]
            file_path = os.path.join(root_path, filename)

            # 根据描述填充元数据
            row = {
                'Experiment_ID': exp_id,
                'Fault_Type': 'Unknown', # 标签未知
                'Fault_Size_Inch': None,
                'Load_HP': None,
                'OR_Position': None,
                'RPM': 600, # 转速约600 rpm
                'Sampling_Rate': 32000, # 采样率32kHz
                # **重要假设**: 我们将目标域信号统一视为'DE'信号，以便与处理流程兼容
                'DE_path': file_path,
                'FE_path': None,
                'BA_path': None,
                'RPM_path': None,
            }
            target_data_list.append(row)

    df = pd.DataFrame(target_data_list)
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"\n目标域结构化数据集已成功保存到: {save_path}")
        
    return df