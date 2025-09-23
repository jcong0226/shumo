import numpy as np
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import io # 用于在内存中操作图像
from PIL import Image
from scipy.fft import fft
from scipy.signal import butter, filtfilt
from scipy.signal import hilbert, stft
import pywt

def create_timeseries_image(segment: np.ndarray, image_size: tuple) -> np.ndarray:
    """将一维信号片段转换为二维时序图图像。"""
    fig = plt.figure(figsize=(image_size[1]/100, image_size[0]/100), dpi=100)
    ax = plt.Axes(fig, [0., 0., 1., 1.]); ax.set_axis_off(); fig.add_axes(ax)
    ax.plot(segment, color='black')
    buf = io.BytesIO(); plt.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
    img = Image.open(buf).convert('L').resize((image_size[1], image_size[0]))
    return np.array(img)

def create_spectrum_image(segment: np.ndarray, fs: int, image_size: tuple) -> np.ndarray:
    """将一维信号片段转换为二维频谱图图像。"""
    N = len(segment); yf = fft(segment)
    spectrum = np.abs(yf[0:N//2])
    fig = plt.figure(figsize=(image_size[1]/100, image_size[0]/100), dpi=100)
    ax = plt.Axes(fig, [0., 0., 1., 1.]); ax.set_axis_off(); fig.add_axes(ax)
    ax.plot(np.log1p(spectrum), color='black')
    buf = io.BytesIO(); plt.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
    img = Image.open(buf).convert('L').resize((image_size[1], image_size[0]))
    return np.array(img)

def create_envelope_spectrum_image(segment: np.ndarray, fs: int, image_size: tuple, 
                                   filter_band: tuple = (2000, 5000), filter_order: int = 3) -> np.ndarray:
    """
    通过包络分析，将一维信号片段转换为二维包络谱图像。

    Args:
        segment (np.ndarray): 输入的一维信号片段 (已标准化)。
        fs (int): 信号的采样率。
        image_size (tuple): 输出图像的目标尺寸 (height, width)。
        filter_band (tuple): 带通滤波的频带 (min_freq, max_freq)。
        filter_order (int): 滤波器的阶数。

    Returns:
        np.ndarray: 代表包络谱的二维数组 (uint8, 0-255)。
    """
    # --- 步骤 1: 带通滤波 ---
    nyquist_freq = 0.5 * fs
    low = filter_band[0] / nyquist_freq
    high = filter_band[1] / nyquist_freq
    # 使用巴特沃斯滤波器
    b, a = butter(filter_order, [low, high], btype='band')
    # 使用filtfilt进行零相位滤波
    filtered_signal = filtfilt(b, a, segment)

    # --- 步骤 2: 包络解调 (希尔伯特变换) ---
    # 计算解析信号，然后取其绝对值得到包络
    analytic_signal = hilbert(filtered_signal)
    envelope = np.abs(analytic_signal)

    # --- 步骤 3: 计算包络的频谱 ---
    N = len(envelope)
    # 对包络信号进行FFT
    yf = fft(envelope)
    # 取前半部分并归一化
    envelope_spectrum = 2.0/N * np.abs(yf[0:N//2])

    # --- 步骤 4: 将包络谱转换为图像 (与create_spectrum_image逻辑相同) ---
    fig = plt.figure(figsize=(image_size[1]/100, image_size[0]/100), dpi=100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    # 绘制对数包络谱
    ax.plot(np.log1p(envelope_spectrum), color='black')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    
    img = Image.open(buf).convert('L').resize((image_size[1], image_size[0]))
    
    return np.array(img)

def create_stft_image(segment: np.ndarray, fs: int, image_size: tuple,
                      nperseg_ratio=0.025) -> np.ndarray:
    """
    将一维信号片段转换为STFT时频谱图。
    Args:
        segment (np.ndarray): 输入信号片段。
        fs (int): 采样率。
        image_size (tuple): 输出图像尺寸。
        nperseg_ratio (float): 窗口长度占信号总长的比例，用于自适应窗口。
                               根据建议 Fs*0.02s，对于2048点信号和12k Fs，约为 Fs * 0.17s / 12000 = 0.014。
                               我们取一个折中值。
    """
    nperseg = int(len(segment) * nperseg_ratio)
    f, t, Zxx = stft(segment, fs=fs, nperseg=nperseg, noverlap=int(nperseg * 0.75)) # 75%重叠
    
    # 使用对数刻度并处理零值
    Sxx = np.abs(Zxx)
    Sxx[Sxx == 0] = 1e-12
    log_Sxx = np.log1p(Sxx)

    # 绘制图像，去除所有边框和坐标轴
    fig = plt.figure(figsize=(image_size[1]/100, image_size[0]/100), dpi=100)
    ax = plt.Axes(fig, [0., 0., 1., 1.]); ax.set_axis_off(); fig.add_axes(ax)
    # 使用 pcolormesh 并选择灰度图
    ax.pcolormesh(t, f, log_Sxx, shading='gouraud', cmap='gray')
    ax.set_ylim(0, fs / 8) # 根据经验，可以只关注较低的频带

    buf = io.BytesIO(); plt.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
    img = Image.open(buf).convert('L').resize((image_size[1], image_size[0]))
    return np.array(img)

def create_cwt_image(segment: np.ndarray, image_size: tuple,
                     scales=np.arange(1, 129), wavelet='morl') -> np.ndarray:
    """
    将一维信号片段转换为CWT时标图 (scalogram)。
    Args:
        segment (np.ndarray): 输入信号片段。
        image_size (tuple): 输出图像尺寸。
        scales (np.ndarray): 小波变换的尺度范围。
        wavelet (str): 母小波，'morl' (Morlet) 是常用选择。
    """
    # 执行CWT
    coefficients, frequencies = pywt.cwt(segment, scales, wavelet)
    
    # 计算能量并取对数
    power = np.abs(coefficients)**2
    log_power = np.log1p(power)

    # 绘制图像
    fig = plt.figure(figsize=(image_size[1]/100, image_size[0]/100), dpi=100)
    ax = plt.Axes(fig, [0., 0., 1., 1.]); ax.set_axis_off(); fig.add_axes(ax)
    # CWT的结果可以直接用imshow绘制
    ax.imshow(log_power, cmap='gray', aspect='auto', interpolation='bilinear')

    buf = io.BytesIO(); plt.savefig(buf, format='png'); plt.close(fig); buf.seek(0)
    img = Image.open(buf).convert('L').resize((image_size[1], image_size[0]))
    return np.array(img)