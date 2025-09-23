# 文件名: utils_feature_extraction.py
import os
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import io
from PIL import Image
from scipy.fft import fft
from scipy.signal import butter, filtfilt, hilbert, stft
from scipy.stats import kurtosis
from utils_create import create_timeseries_image, create_spectrum_image, create_envelope_spectrum_image
import pywt
from utils_create import (
    create_timeseries_image,
    create_spectrum_image,
    create_envelope_spectrum_image,
    create_stft_image,  # 新增
    create_cwt_image    # 新增
)

# (此处粘贴 create_timeseries_image, create_spectrum_image, 和 create_envelope_spectrum_image 三个函数的完整代码)
# ...
# ...
def find_optimal_band_with_sk(signal: np.ndarray, fs: int, nperseg: int = 256):
    """【SK自适应版】通过包络分析生成包络谱图像。"""
    f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
    spectral_kurt = kurtosis(np.abs(Zxx), axis=1, fisher=True)
    max_kurt_idx = np.argmax(spectral_kurt)
    if spectral_kurt[max_kurt_idx] < 1.0:
        return None
    center_freq = f[max_kurt_idx]
    bandwidth = fs / nperseg
    min_freq = center_freq - bandwidth / 2
    max_freq = center_freq + bandwidth / 2
    if min_freq < 0: min_freq = 0
    if max_freq > fs / 2: max_freq = fs / 2
    return (min_freq, max_freq)

def create_envelope_spectrum_image_sk(segment: np.ndarray, fs: int, image_size: tuple,
                                      filter_order: int = 3) -> np.ndarray:
    """【SK自适应版】通过包络分析生成包络谱图像。"""
    optimal_band = find_optimal_band_with_sk(segment, fs)
    filter_band = optimal_band if optimal_band is not None else (2000, 5000)

    nyquist_freq = 0.5 * fs
    low = filter_band[0] / nyquist_freq
    high = filter_band[1] / nyquist_freq

    if low <= 0: low = 1e-5
    if high >= 1: high = 0.99999
    if low >= high:
        low, high = 2000 / nyquist_freq, 5000 / nyquist_freq

    b, a = butter(filter_order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, segment)
    envelope = np.abs(hilbert(filtered_signal))
    N = len(envelope)
    yf = fft(envelope)
    envelope_spectrum = 2.0/N * np.abs(yf[0:N//2])

    fig = plt.figure(figsize=(image_size[1]/100, image_size[0]/100), dpi=100)
    ax = plt.Axes(fig, [0., 0., 1., 1.]); ax.set_axis_off(); fig.add_axes(ax)
    ax.plot(np.log1p(envelope_spectrum), color='black')
    buf = io.BytesIO(); plt.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
    img = Image.open(buf).convert('L').resize((image_size[1], image_size[0]))

    return np.array(img)

def generate_and_save_images(processed_samples: list, dataset_type: str, signal_type: str,
                             image_dirs: dict, # 使用一个字典来传递所有路径
                             image_size: tuple, target_fs: int):
    """
    为每个样本生成五种特征图并保存到各自的目录中。
    (时序图, 频谱图, 包络谱图, STFT谱图, CWT时标图)
    """
    print(f"\n开始为 [{dataset_type}] 数据集的 [{signal_type}] 信号生成五种图像...")
    
    # 从字典中解包路径
    ts_dir_base = image_dirs['ts']
    sp_dir_base = image_dirs['sp']
    env_sp_dir_base = image_dirs['env_sp']
    stft_dir_base = image_dirs['stft'] # 新增
    cwt_dir_base = image_dirs['cwt']   # 新增

    for i, sample in enumerate(tqdm(processed_samples, desc=f"Generating {signal_type} images")):
        segment, label = sample['signal_segment'], sample['label']
        original_filename = os.path.splitext(os.path.basename(sample['original_path']))[0]
        save_filename = f"{original_filename}_seg_{i}.png"

        # 统一的保存逻辑
        def save_image(img_array, base_dir, label, filename):
            save_dir = os.path.join(base_dir, dataset_type, signal_type, label)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, filename)
            Image.fromarray(img_array).save(save_path)

        # 1. 时序图
        timeseries_img = create_timeseries_image(segment, image_size)
        save_image(timeseries_img, ts_dir_base, label, save_filename)

        # 2. 频谱图
        spectrum_img = create_spectrum_image(segment, fs=target_fs, image_size=image_size)
        save_image(spectrum_img, sp_dir_base, label, save_filename)

        # 3. 包络谱图 (使用SK自适应版本)
        envelope_spectrum_img = create_envelope_spectrum_image_sk(segment, fs=target_fs, image_size=image_size)
        save_image(envelope_spectrum_img, env_sp_dir_base, label, save_filename)
        
        # 4. 【新增】STFT谱图
        stft_img = create_stft_image(segment, fs=target_fs, image_size=image_size)
        save_image(stft_img, stft_dir_base, label, save_filename)

        # 5. 【新增】CWT时标图
        cwt_img = create_cwt_image(segment, image_size=image_size)
        save_image(cwt_img, cwt_dir_base, label, save_filename)

    print(f"[{signal_type}] 信号的五种图像生成完成！")
    
    # 文件名: utils_feature_extraction.py (新增函数)

