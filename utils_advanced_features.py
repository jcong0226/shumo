# 文件名: utils_advanced_features.py

import numpy as np
import pywt
from antropy import perm_entropy, higuchi_fd

def extract_wpt_features(segment: np.ndarray, wavelet: str = 'db4', maxlevel: int = 3) -> dict:
    """
    使用小波包变换(WPT)提取信号在不同频带的能量特征。

    Returns:
        dict: 包含 2^maxlevel 个能量特征的字典。
    """
    # 1. 执行WPT分解
    wp = pywt.WaveletPacket(data=segment, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
    
    # 2. 获取最后一层的所有节点（频带）
    nodes = wp.get_level(maxlevel, order='natural')
    
    # 3. 计算每个节点的能量 (RMS的平方)
    #    我们使用 reconstruction=True 来获取信号分量
    labels = [node.path for node in nodes]
    energy = [np.sum(np.square(node.data)) for node in nodes]
    
    # 4. 创建特征字典
    features = {f'wpt_energy_{label}': E for label, E in zip(labels, energy)}
    
    return features

def extract_entropy_features(segment: np.ndarray, order: int = 3, delay: int = 1) -> dict:
    """
    计算信号的排列熵。

    Args:
        order (int): 排列的阶数（嵌入维度）。
        delay (int): 时间延迟。

    Returns:
        dict: 包含排列熵的字典。
    """
    # 使用antropy库计算排列熵，normalize=True使其值在0-1之间
    p_entropy = perm_entropy(segment, order=order, delay=delay, normalize=True)
    return {'complexity_perm_entropy': p_entropy}

def extract_fractal_features(segment: np.ndarray, k_max: int = 10) -> dict:
    """
    计算信号的Higuchi分形维数。

    Args:
        k_max (int): 计算中使用的最大时间间隔。

    Returns:
        dict: 包含Higuchi分形维数的字典。
    """
    hfd = higuchi_fd(segment, k=k_max)
    return {'complexity_higuchi_fd': hfd}


def extract_all_advanced_features(segment: np.ndarray, fs: int) -> dict:
    """
    一个主函数，调用以上所有函数，提取全部高级特征。

    Returns:
        dict: 包含所有高级特征的扁平字典。
    """
    # 为了数值稳定性，对输入片段再做一次标准化
    if np.std(segment) > 1e-8:
        segment = (segment - np.mean(segment)) / np.std(segment)
    
    # 提取各类高级特征
    wpt_features = extract_wpt_features(segment)
    entropy_features = extract_entropy_features(segment)
    fractal_features = extract_fractal_features(segment)
    
    # 合并所有特征到一个字典中
    all_features = {}
    all_features.update(wpt_features)
    all_features.update(entropy_features)
    all_features.update(fractal_features)
    
    return all_features