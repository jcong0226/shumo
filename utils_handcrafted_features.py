import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
from scipy.fft import fft
from scipy.signal import butter, filtfilt, hilbert, stft
from utils_analysis import calculate_fault_frequencies
from utils_feature_extraction import find_optimal_band_with_sk
# 确保您能从其他文件中导入这些函数
# from utils_analysis import calculate_fault_frequencies
# from utils_feature_extraction import create_envelope_spectrum_image # 我们需要复用其中的逻辑

def extract_handcrafted_features(segment: np.ndarray, fs: int, rpm: float, bearing_type: str) -> dict:
    """
    从单个信号片段中提取一套完整的手工特征。

    Args:
        segment (np.ndarray): 输入的1D信号片段 (已标准化)。
        fs (int): 采样率。
        rpm (float): 转速。
        bearing_type (str): 轴承类型 ('DE' 或 'FE')。

    Returns:
        dict: 包含所有手工特征的字典。
    """
    features = {}

    # --- 1. 时域统计特征 (Time-domain Statistics) ---
    rms = np.sqrt(np.mean(segment**2))
    peak = np.max(np.abs(segment))
    features['time_rms'] = rms
    features['time_peak'] = peak
    features['time_variance'] = np.var(segment)
    features['time_skew'] = skew(segment)
    features['time_kurtosis'] = kurtosis(segment)
    features['time_impulse_factor'] = peak / np.mean(np.abs(segment)) if np.mean(np.abs(segment)) > 1e-8 else 0
    features['time_clearance_factor'] = np.max(np.abs(segment)) / (np.mean(np.sqrt(np.abs(segment)))**2) if np.mean(np.sqrt(np.abs(segment))) > 1e-8 else 0
    features['time_shape_factor'] = rms / np.mean(np.abs(segment)) if np.mean(np.abs(segment)) > 1e-8 else 0
    features['time_p2p'] = np.ptp(segment) # Peak-to-peak
    if rms > 1e-8:
        features['time_crest_factor'] = peak / rms  # 裕度因子
    else:
        features['time_crest_factor'] = 0

    # --- 2. 频域特征 (Frequency-domain Statistics) ---
    N = len(segment)
    yf = fft(segment)
    spectrum = 2.0/N * np.abs(yf[0:N//2])
    freq_axis = np.linspace(0.0, fs/2.0, N//2)
    features['freq_mean'] = np.mean(spectrum)
    features['freq_std'] = np.std(spectrum)
    # 频率重心
    features['freq_centroid'] = np.sum(spectrum * freq_axis) / np.sum(spectrum) if np.sum(spectrum) > 0 else 0

    # --- 3. 包络谱特征 (Envelope Spectrum Features) ---
    # a. 计算包络谱 (复用之前的逻辑)
    optimal_band = find_optimal_band_with_sk(segment, fs)
    filter_band = optimal_band if optimal_band is not None else (2000, 5000) # 自适应频带
    nyquist_freq = 0.5 * fs
    low = filter_band[0] / nyquist_freq
    high = filter_band[1] / nyquist_freq
    b, a = butter(3, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, segment)
    envelope = np.abs(hilbert(filtered_signal))
    N_env = len(envelope)
    yf_env = fft(envelope)
    env_spectrum = 2.0/N_env * np.abs(yf_env[0:N_env//2])
    env_freq_axis = np.linspace(0.0, fs/2.0, N_env//2)
    
    # b. 计算理论故障频率
    fault_freqs = calculate_fault_frequencies(rpm, bearing_type=bearing_type)
    
    if fault_freqs:
        # c. 提取包络谱上的量化指标
        total_energy = np.sum(env_spectrum**2)
        
        freq_map = {
            'BPFO': fault_freqs['BPFO'],
            'BPFI': fault_freqs['BPFI'],
            'BSF2x': fault_freqs['BSF'] * 2 # 通常关注2倍BSF
        }

        for name, freq_val in freq_map.items():
            # 寻找在理论频率附近±5Hz内的峰值
            indices = np.where(np.abs(env_freq_axis - freq_val) <= 5)[0]
            if len(indices) > 0:
                peak_amp = np.max(env_spectrum[indices])
                features[f'env_{name}_peak_amp'] = peak_amp
                # 峰值能量比 (域不变特征)
                features[f'env_{name}_peak_ratio'] = peak_amp**2 / total_energy if total_energy > 0 else 0
            else:
                features[f'env_{name}_peak_amp'] = 0
                features[f'env_{name}_peak_ratio'] = 0

        # d. 旁带调制指数 (Sideband Modulation Index)
        fr = fault_freqs['Fr']
        bpfi_freq = fault_freqs['BPFI']
        # 寻找BPFI主峰和左右第一边带的峰值
        idx_center = np.argmin(np.abs(env_freq_axis - bpfi_freq))
        idx_left = np.argmin(np.abs(env_freq_axis - (bpfi_freq - fr)))
        idx_right = np.argmin(np.abs(env_freq_axis - (bpfi_freq + fr)))
        
        amp_center = env_spectrum[idx_center]
        amp_left = env_spectrum[idx_left]
        amp_right = env_spectrum[idx_right]
        
        if amp_center > 1e-8:
            features['env_bpfi_smi'] = (amp_left + amp_right) / amp_center
        else:
            features['env_bpfi_smi'] = 0
            
    return features