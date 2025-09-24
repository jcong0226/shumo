# 文件名: utils_analysis.py (精简版 - 移除STFT绘图)
import numpy as np
import matplotlib.pyplot as plt
import re
import os
from scipy.fft import fft
from scipy.signal import butter, filtfilt, hilbert # 移除了 stft
from scipy.stats import kurtosis
import pywt # CWT 绘图仍然保留在原来的函数中

def plot_kurtogram(signal: np.ndarray, fs: int, title: str, save_dir: str, nperseg_range=None):
    """
    计算并可视化信号的谱峭度图 (Kurtogram)，用于辅助选择最佳解调频带。
    """
    # ... (此函数内容保持不变)
    print(f"正在为 '{title}' 生成谱峭度图 (Kurtogram)...")

    if nperseg_range is None:
        # 为了计算谱峭度，仍然需要STFT的计算逻辑，但不会生成最终的STFT图
        from scipy.signal import stft
        nperseg_range = [2**i for i in range(5, 11)]

    kurt_values = []
    freq_bins = []

    for nperseg in nperseg_range:
        if len(signal) < nperseg:
            continue
        f, _, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
        kurt = kurtosis(np.abs(Zxx), axis=1, fisher=True)
        kurt_values.append(kurt)
        freq_bins.append(f)

    max_kurt = -np.inf
    best_nperseg = None
    best_freq_idx = None
    for i, nperseg in enumerate(nperseg_range):
        if i < len(kurt_values):
            kurt = kurt_values[i]
            if np.max(kurt) > max_kurt:
                max_kurt = np.max(kurt)
                best_nperseg = nperseg
                best_freq_idx = np.argmax(kurt)

    if best_nperseg is None:
        print(f"警告: 无法为 '{title}' 找到最佳频带。")
        return

    best_center_freq = freq_bins[nperseg_range.index(best_nperseg)][best_freq_idx]
    bandwidth = fs / best_nperseg

    fig, ax = plt.subplots(figsize=(12, 8))

    max_freq = fs / 2
    display_freq_axis = np.linspace(0, max_freq, 512)
    kurt_map = np.zeros((len(nperseg_range), len(display_freq_axis)))

    for i, nperseg in enumerate(nperseg_range):
        if i < len(kurt_values):
            kurt_map[i, :] = np.interp(display_freq_axis, freq_bins[i], kurt_values[i])

    img = ax.imshow(kurt_map, aspect='auto', origin='lower',
                    extent=[0, max_freq, 0, len(nperseg_range)],
                    cmap='hot')
    fig.colorbar(img, ax=ax, label='Kurtosis Value')

    ax.plot(best_center_freq, nperseg_range.index(best_nperseg) + 0.5, 'c+', markersize=15, markeredgewidth=2)

    ax.set_yticks(np.arange(len(nperseg_range)) + 0.5)
    ax.set_yticklabels([f'Level {i} (n={n})' for i, n in enumerate(nperseg_range)])
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Decomposition Level (Window Size)')
    ax.set_title(f"Kurtogram for '{title}'\nOptimal Band Found at {best_center_freq:.1f} Hz, Bandwidth={bandwidth:.1f} Hz")

    plt.tight_layout()
    safe_filename = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_') + "_Kurtogram.png"
    save_path = os.path.join(save_dir, safe_filename)
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Kurtogram已保存至: {save_path}")


def calculate_fault_frequencies(rpm: float, bearing_type: str, alpha_deg=0):
    """
    【修正版】根据轴承类型(DE/FE)计算理论故障频率。
    """
    # ... (此函数内容保持不变)
    BEARING_PARAMS = {
        'DE': {'name': 'SKF6205', 'n': 9, 'd': 0.3126, 'D': 1.537},
        'FE': {'name': 'SKF6203', 'n': 9, 'd': 0.2656, 'D': 1.122}
    }
    params = BEARING_PARAMS.get(bearing_type.upper())
    if not params:
        return None
    n, d, D = params['n'], params['d'], params['D']
    alpha_rad = np.deg2rad(alpha_deg)
    fr = rpm / 60.0
    bpfo = (n / 2.0) * fr * (1 - (d / D) * np.cos(alpha_rad))
    bpfi = (n / 2.0) * fr * (1 + (d / D) * np.cos(alpha_rad))
    bsf = (D / (2.0 * d)) * fr * (1 - ((d / D) * np.cos(alpha_rad))**2)
    ftf = (1 / 2.0) * fr * (1 - (d / D) * np.cos(alpha_rad))
    return {'Fr': fr, 'BPFO': bpfo, 'BPFI': bpfi, 'BSF': bsf, '2BSF': 2*bsf, 'FTF': ftf}


def tkeo(signal: np.ndarray) -> np.ndarray:
    """
    计算信号的Teager-Kaiser能量算子 (TKEO)。
    """
    # ... (此函数内容保持不变)
    signal_padded = np.pad(signal, pad_width=1, mode='constant', constant_values=0)
    energy = signal_padded[1:-1]**2 - signal_padded[:-2] * signal_padded[2:]
    return energy


# 【已删除】: analyze_and_compare_stft_cwt 函数已被完全移除


def analyze_and_plot_sample(signal: np.ndarray, fs: int, rpm: float, title: str, signal_type: str, save_dir: str):
    """
    【精简版】对单个信号样本进行多维度分析和可视化（已移除STFT图）。
    """
    # 【核心修改1】: 将子图布局从 5x1 改为 4x1，并调整画布大小
    fig, axes = plt.subplots(4, 1, figsize=(15, 16))
    fig.suptitle(title, fontsize=16)

    # --- 子图1: 时域波形 (axes[0]) ---
    time_axis = np.arange(len(signal)) / fs
    axes[0].plot(time_axis, signal, color='b', linewidth=0.75)
    axes[0].set_title("1. Raw Time-domain Waveform")
    axes[0].set_xlabel("Time (s)"); axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].set_xlim(0, 0.2)

    # --- 子图2: TKEO能量算子 (axes[1]) ---
    nyquist_freq = 0.5 * fs
    b, a = butter(3, 100 / nyquist_freq, btype='highpass')
    signal_filtered = filtfilt(b, a, signal)
    energy = tkeo(signal_filtered)
    axes[1].plot(time_axis, energy, color='green')
    axes[1].set_title("2. Instantaneous Energy (TKEO)")
    axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("Energy")
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].set_xlim(0, 0.2)

    # --- 子图3: FFT频谱图 (axes[2]) ---
    N = len(signal_filtered); yf = fft(signal_filtered)
    spectrum = 2.0/N * np.abs(yf[0:N//2])
    freq_axis = np.linspace(0.0, fs/2.0, N//2)
    axes[2].plot(freq_axis, spectrum, label='FFT Spectrum')
    axes[2].set_title("3. FFT Spectrum of Filtered Signal")
    axes[2].set_xlabel("Frequency (Hz)"); axes[2].set_ylabel("Amplitude")
    axes[2].grid(True, linestyle='--', alpha=0.6)

    # --- 【已删除】: 原来的 axes[3] STFT 谱图已被移除 ---

    # --- 子图4: 包络谱图 (现在是 axes[3]) ---
    # 【核心修改2】: 将所有对 axes[4] 的引用改为 axes[3]
    b_bp, a_bp = butter(3, [2000 / nyquist_freq, 5000 / nyquist_freq], btype='band')
    filtered_signal_bp = filtfilt(b_bp, a_bp, signal)
    envelope = np.abs(hilbert(filtered_signal_bp))
    N_env = len(envelope); yf_env = fft(envelope)
    env_spectrum = 2.0/N_env * np.abs(yf_env[0:N_env//2])
    env_freq_axis = np.linspace(0.0, fs/2.0, N_env//2)
    axes[3].plot(env_freq_axis, env_spectrum, color='purple', label='Envelope Spectrum')
    axes[3].set_title("4. Envelope Spectrum (with Harmonics & Sidebands)")
    axes[3].set_xlabel("Frequency (Hz)"); axes[3].set_ylabel("Amplitude")
    axes[3].grid(True, linestyle='--', alpha=0.6)

    # --- 谐波与边频带标记逻辑 (标记到新的子图索引) ---
    fault_freqs = calculate_fault_frequencies(rpm, bearing_type=signal_type)
    if fault_freqs:
        ftf = fault_freqs['FTF']
        xlim_env = max(5.5 * fault_freqs['BPFO'], 10 * fault_freqs['Fr'])
        axes[3].set_xlim(0, xlim_env) # 修改 axes[4] -> axes[3]
        axes[2].set_xlim(0, xlim_env * 2)

        plot_config = {
            'BPFO': ('r', 'Outer Race'), 'BPFI': ('g', 'Inner Race'), '2BSF': ('m', 'Ball')
        }

        for name, (color, label_prefix) in plot_config.items():
            base_freq = fault_freqs[name]
            # 在FFT谱上标记谐波
            for i in range(1, 4):
                axes[2].axvline(x=base_freq * i, color=color, linestyle='--', alpha=0.8,
                                label=f'{i}x {label_prefix}' if i==1 else f'_{i}x {label_prefix}')
            # 在包络谱上标记
            for i in range(1, 6):
                freq_harmonic = base_freq * i
                axes[3].axvline(x=freq_harmonic, color=color, linestyle='--', alpha=0.8, # 修改 axes[4] -> axes[3]
                                label=f'{i}x {label_prefix}' if i==1 else f'_{i}x {label_prefix}')
                if name == '2BSF':
                    axes[3].axvline(x=freq_harmonic - ftf, color=color, linestyle=':', alpha=0.6) # 修改 axes[4] -> axes[3]
                    axes[3].axvline(x=freq_harmonic + ftf, color=color, linestyle=':', alpha=0.6) # 修改 axes[4] -> axes[3]

    axes[2].legend(loc='upper right')
    axes[3].legend(loc='upper right') # 修改 axes[4] -> axes[3]

    # --- 保存图像 ---
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    safe_filename = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_') + "_Enhanced.png"
    save_path = os.path.join(save_dir, safe_filename)
    plt.savefig(save_path)
    plt.close(fig)
    print(f"精简版分析图已保存至: {save_path}")