# 文件名: utils_comprehensive_features.py

import numpy as np
import pywt
from scipy.stats import skew, kurtosis, moment
from scipy.fft import fft
from scipy.signal import hilbert, butter, filtfilt
from antropy import perm_entropy, higuchi_fd
from utils_analysis import calculate_fault_frequencies

"""
这是一个全面特征提取模块，整合了多个维度的特征。
"""

# ==============================================================================
#                      1. 时域特征 (Time Domain)
# ==============================================================================
def _extract_time_domain_features(segment: np.ndarray) -> dict:
    """提取全面的时域统计和形态特征。"""
    features = {}
    
    # --- 基础统计量 ---
    mean_abs = np.mean(np.abs(segment))
    rms = np.sqrt(np.mean(segment**2))
    std = np.std(segment)
    
    features['time_mean'] = np.mean(segment)
    features['time_mean_abs'] = mean_abs
    features['time_rms'] = rms
    features['time_std'] = std
    features['time_variance'] = np.var(segment)
    features['time_peak'] = np.max(np.abs(segment))
    features['time_p2p'] = np.ptp(segment) # Peak-to-Peak
    
    # --- 高阶统计量 ---
    features['time_skew'] = skew(segment)
    features['time_kurtosis'] = kurtosis(segment)
    features['time_moment_5'] = moment(segment, moment=5) # 5阶中心矩
    features['time_moment_6'] = moment(segment, moment=6) # 6阶中心矩
    
    # --- 形态指标 (Morphological Indicators) ---
    # 避免除以零
    if rms < 1e-8:
        features['time_crest_factor'] = 0
        features['time_shape_factor'] = 0
    else:
        features['time_crest_factor'] = features['time_peak'] / rms
        features['time_shape_factor'] = rms / mean_abs if mean_abs > 1e-8 else 0

    if mean_abs < 1e-8:
        features['time_impulse_factor'] = 0
    else:
        features['time_impulse_factor'] = features['time_peak'] / mean_abs

    if np.mean(np.sqrt(np.abs(segment)))**2 < 1e-8:
        features['time_clearance_factor'] = 0
    else:
        features['time_clearance_factor'] = features['time_peak'] / (np.mean(np.sqrt(np.abs(segment)))**2)
        
    return features

# ==============================================================================
#                      2. 频域特征 (Frequency Domain)
# ==============================================================================
def _extract_freq_domain_features(segment: np.ndarray, fs: int) -> dict:
    """从FFT频谱中提取频域统计特征。"""
    N = len(segment)
    yf = fft(segment)
    # 计算单边谱
    spectrum = 2.0/N * np.abs(yf[0:N//2])
    freq_axis = np.linspace(0.0, fs/2.0, N//2)
    
    features = {}
    
    # --- 基础统计量 ---
    features['freq_mean'] = np.mean(spectrum)
    features['freq_std'] = np.std(spectrum)
    features['freq_rms'] = np.sqrt(np.mean(spectrum**2))

    # --- 频谱形态特征 ---
    sum_spectrum = np.sum(spectrum) if np.sum(spectrum) > 0 else 1e-8
    
    # 频率重心 (Centroid)
    features['freq_centroid'] = np.sum(spectrum * freq_axis) / sum_spectrum
    # 均方根频率 (RMSF)
    features['freq_rmsf'] = np.sqrt(np.sum(spectrum**2 * freq_axis) / sum_spectrum)
    
    # --- 频谱高阶统计量 ---
    # 需要以重心为中心计算
    features['freq_skew'] = np.sum(((freq_axis - features['freq_centroid'])**3) * spectrum) / (sum_spectrum * features['freq_rmsf']**3)
    features['freq_kurtosis'] = np.sum(((freq_axis - features['freq_centroid'])**4) * spectrum) / (sum_spectrum * features['freq_rmsf']**4)

    return features

# ==============================================================================
#                  3. 包络谱特征 (Envelope Spectrum)
# ==============================================================================
def _extract_envelope_features(segment: np.ndarray, fs: int, rpm: float, bearing_type: str) -> dict:
    """提取包络谱上的故障特征频率相关指标。"""
    features = {}
    nyquist_freq = 0.5 * fs

    # 使用固定的、鲁棒性好的带通滤波范围
    low = 2000 / nyquist_freq
    high = 5000 / nyquist_freq
    b, a = butter(3, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, segment)
    
    envelope = np.abs(hilbert(filtered_signal))
    
    N_env = len(envelope)
    yf_env = fft(envelope)
    env_spectrum = 2.0/N_env * np.abs(yf_env[0:N_env//2])
    env_freq_axis = np.linspace(0.0, fs/2.0, N_env//2)
    
    fault_freqs = calculate_fault_frequencies(rpm, bearing_type=bearing_type)
    if not fault_freqs:
        return {}

    total_energy = np.sum(env_spectrum**2)
    
    # 关注外圈、内圈和滚动体（2倍BSF）的基频和前3次谐波
    freq_map = {'BPFO': fault_freqs['BPFO'], 'BPFI': fault_freqs['BPFI'], '2BSF': fault_freqs['2BSF']}

    for name, base_freq in freq_map.items():
        for i in range(1, 4): # 基频 + 2次谐波
            harmonic_freq = base_freq * i
            indices = np.where(np.abs(env_freq_axis - harmonic_freq) <= 5)[0] # ±5Hz 搜索范围
            
            peak_amp = np.max(env_spectrum[indices]) if len(indices) > 0 else 0
            
            features[f'env_{name}_{i}x_amp'] = peak_amp
            features[f'env_{name}_{i}x_ratio'] = peak_amp**2 / total_energy if total_energy > 1e-8 else 0
            
    return features

# ==============================================================================
#                      4. 小波与复杂度特征 (Wavelet & Complexity)
# ==============================================================================
def _extract_wavelet_complexity_features(segment: np.ndarray) -> dict:
    """提取小波包能量、熵和信号复杂度特征。"""
    features = {}
    
    # --- 小波包变换 (WPT) 特征 ---
    wp = pywt.WaveletPacket(data=segment, wavelet='db4', mode='symmetric', maxlevel=3)
    nodes = wp.get_level(3, order='natural')
    
    wpt_energies = [np.sum(np.square(node.data)) for node in nodes]
    for i, energy in enumerate(wpt_energies):
        features[f'wpt_energy_node_{i}'] = energy
        
    # 计算WPT能量分布的香农熵
    total_wpt_energy = np.sum(wpt_energies)
    if total_wpt_energy > 1e-8:
        p = np.array(wpt_energies) / total_wpt_energy
        p = p[p > 0] # 避免log(0)
        features['wpt_shannon_entropy'] = -np.sum(p * np.log2(p))
    else:
        features['wpt_shannon_entropy'] = 0
        
    # --- 复杂度特征 ---
    features['complexity_perm_entropy'] = perm_entropy(segment, order=3, delay=1, normalize=True)
    features['complexity_higuchi_fd'] = higuchi_fd(segment, kmax=10)
    
    return features

# ==============================================================================
#                        主函数 (Main Function)
# ==============================================================================

def extract_comprehensive_features(segment: np.ndarray, fs: int, rpm: float, bearing_type: str) -> dict:
    """
    【主调用函数】执行所有特征提取步骤，并返回一个包含所有特征的扁平字典。

    Args:
        segment (np.ndarray): 输入的1D信号片段。
        fs (int): 采样率。
        rpm (float): 转速。
        bearing_type (str): 轴承类型 ('DE' 或 'FE')。

    Returns:
        dict: 包含超过40种手工特征的字典。
    """
    all_features = {}
    
    # 确保信号片段已标准化
    if np.std(segment) > 1e-8:
        segment = (segment - np.mean(segment)) / np.std(segment)
        
    # 1. 提取时域特征 (14个)
    time_features = _extract_time_domain_features(segment)
    all_features.update(time_features)
    
    # 2. 提取频域特征 (6个)
    freq_features = _extract_freq_domain_features(segment, fs)
    all_features.update(freq_features)
    
    # 3. 提取包络谱特征 (3 * 2 * 3 = 18个)
    envelope_features = _extract_envelope_features(segment, fs, rpm, bearing_type)
    all_features.update(envelope_features)
    
    # 4. 提取小波与复杂度特征 (8 + 1 + 2 = 11个)
    wavelet_complexity_features = _extract_wavelet_complexity_features(segment)
    all_features.update(wavelet_complexity_features)
    
    return all_features