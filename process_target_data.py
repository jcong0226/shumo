# 文件名: process_target_data.py (修正并优化版)

import os
import pandas as pd
from tqdm import tqdm
import warnings

# --- 从我们已经创建的模块化文件中导入所有需要的函数 ---
from utils_loader import create_target_structured_dataset
from utils_preprocess import preprocess_from_metadata
from utils_feature_extraction import generate_and_save_images
# 【核心修正1】导入新的、统一的特征提取函数
from utils_comprehensive_features import extract_comprehensive_features

# ==============================================================================
#                      1. 统一路径和参数设置 (与main.py完全一致)
# ==============================================================================

# --- 顶级输入路径 ---
TARGET_DATA_PATH = r'C:\Users\JC\Desktop\shumo\datasets\converted\target'

# --- 【核心修正2】与main.py保持一致的顶级输出目录 ---
BASE_OUTPUT_DIR = r'C:\Users\JC\Desktop\shumo\datasets\output_results'

# --- 所有输出路径现在都基于 BASE_OUTPUT_DIR ---
# 1. CSV 数据路径
TARGET_METADATA_CSV_PATH = os.path.join(BASE_OUTPUT_DIR, 'metadata', 'target_metadata_summary.csv')
# 【核心修正3】修正手工特征的保存路径
TARGET_HANDCRAFTED_FEATURES_CSV_PATH = os.path.join(BASE_OUTPUT_DIR, 'features', 'target_handcrafted_features.csv')

# 2. 图像特征路径
IMAGE_FEATURES_DIR = os.path.join(BASE_OUTPUT_DIR, 'image_features')
TARGET_TIMESERIES_OUTPUT_DIR = os.path.join(IMAGE_FEATURES_DIR, 'timeseries')
TARGET_SPECTRUM_OUTPUT_DIR = os.path.join(IMAGE_FEATURES_DIR, 'spectrum')
TARGET_ENVELOPE_SPECTRUM_OUTPUT_DIR = os.path.join(IMAGE_FEATURES_DIR, 'envelope_spectrum')
TARGET_STFT_OUTPUT_DIR = os.path.join(IMAGE_FEATURES_DIR, 'stft')
TARGET_CWT_OUTPUT_DIR = os.path.join(IMAGE_FEATURES_DIR, 'cwt')


# 3. 预处理与图像参数 (必须与源域处理时完全一致)
TARGET_FS = 32000
WINDOW_SIZE = 4096
OVERLAP = 0.5
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP))
IMAGE_SIZE = (224, 224)


def process_target_domain_data():
    """处理目标域数据的完整流程"""
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    # 确保所有输出目录都存在
    os.makedirs(os.path.dirname(TARGET_METADATA_CSV_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(TARGET_HANDCRAFTED_FEATURES_CSV_PATH), exist_ok=True)
    for path in [TARGET_TIMESERIES_OUTPUT_DIR, TARGET_SPECTRUM_OUTPUT_DIR, TARGET_ENVELOPE_SPECTRUM_OUTPUT_DIR, TARGET_STFT_OUTPUT_DIR, TARGET_CWT_OUTPUT_DIR]:
        os.makedirs(path, exist_ok=True)

    # ==================== 步骤 1: 准备目标域元数据 ====================
    print(f"\n{'='*25} 步骤 1: 准备目标域元数据 {'='*25}")
    if not os.path.exists(TARGET_METADATA_CSV_PATH):
        target_metadata_df = create_target_structured_dataset(
            root_path=TARGET_DATA_PATH, save_path=TARGET_METADATA_CSV_PATH
        )
    else:
        print("检测到已存在的目标域元数据文件，直接加载。")
        target_metadata_df = pd.read_csv(TARGET_METADATA_CSV_PATH)
    print("目标域元数据准备完毕。")

    # ==================== 步骤 2: 统一预处理 ====================
    # 【核心修正4】只执行一次预处理，并将结果复用于后续步骤
    print(f"\n{'='*25} 步骤 2: 统一预处理 {'='*25}")
    
    # 检查手工特征文件，如果它存在，我们可以认为所有处理都已完成
    if os.path.exists(TARGET_HANDCRAFTED_FEATURES_CSV_PATH):
        print("检测到最终的目标域手工特征文件已存在，跳过所有预处理和特征提取步骤。")
        processed_target_samples = [] # 置为空列表，后续步骤将直接跳过
    else:
        # 我们假设目标域信号为DE类型
        processed_target_samples = preprocess_from_metadata(
            metadata_df=target_metadata_df, signal_type='DE', target_fs=TARGET_FS,
            window_size=WINDOW_SIZE, step_size=STEP_SIZE
        )

    # ==================== 步骤 3: 生成图像特征 ====================
    print(f"\n{'='*25} 步骤 3: 生成目标域图像特征 {'='*25}")
    # 检查图像特征是否已存在
    check_dir = os.path.join(TARGET_TIMESERIES_OUTPUT_DIR, 'target', 'DE', 'Unknown')
    if os.path.exists(check_dir) and len(os.listdir(check_dir)) > 0:
        print("检测到目标域的特征图像已存在，跳过生成。")
    elif processed_target_samples:
        print("开始为目标域信号生成所有五种特征图像...")
        # 【核心修正5】将所有路径打包成一个字典传入
        image_dirs_to_generate = {
            'ts': TARGET_TIMESERIES_OUTPUT_DIR,
            'sp': TARGET_SPECTRUM_OUTPUT_DIR,
            'env_sp': TARGET_ENVELOPE_SPECTRUM_OUTPUT_DIR,
            'stft': TARGET_STFT_OUTPUT_DIR,
            'cwt': TARGET_CWT_OUTPUT_DIR
        }
        generate_and_save_images(
            processed_samples=processed_target_samples, dataset_type='target', signal_type='DE',
            image_dirs=image_dirs_to_generate,
            image_size=IMAGE_SIZE, target_fs=TARGET_FS
        )
    else:
        print("没有需要处理的样本来生成图像特征。")

    # ==================== 步骤 4: 生成手工特征 ====================
    print(f"\n{'='*25} 步骤 4: 生成目标域手工特征 {'='*25}")
    if os.path.exists(TARGET_HANDCRAFTED_FEATURES_CSV_PATH):
        print("检测到已存在的目标域手工特征文件，跳过生成。")
    elif processed_target_samples:
        print("开始为目标域信号提取全面的手工特征...")
        all_features_list = []
        for sample in tqdm(processed_target_samples, desc="Extracting target comprehensive features"):
            # 【核心修正6】调用新的、统一的特征提取函数
            features = extract_comprehensive_features(
                segment=sample['signal_segment'], 
                fs=TARGET_FS,
                rpm=sample['rpm'], 
                bearing_type='DE' # 统一按DE轴承参数计算
            )
            features.update({k: v for k, v in sample.items() if k != 'signal_segment'})
            all_features_list.append(features)

        target_handcrafted_df = pd.DataFrame(all_features_list)
        target_handcrafted_df.to_csv(TARGET_HANDCRAFTED_FEATURES_CSV_PATH, index=False)
        print(f"目标域全面手工特征提取完成！已保存至: {TARGET_HANDCRAFTED_FEATURES_CSV_PATH}")
    else:
        print("没有需要处理的样本来提取手工特征。")

    print(f"\n{'='*25} 目标域数据处理完毕 {'='*25}")

if __name__ == '__main__':
    process_target_domain_data()