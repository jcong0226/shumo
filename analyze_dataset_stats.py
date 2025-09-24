import pandas as pd
import os
import numpy as np

# 从您现有的代码中导入核心的数据加载和预处理逻辑
from utils_loader import create_structured_dataset
from utils_preprocess import preprocess_from_metadata

def analyze_and_print_stats():
    """
    加载源域数据集元数据，并专门为驱动端(DE)数据生成详细的统计信息，
    以供论文撰写使用。
    """
    # --- 1. 定义路径和参数 (与您的 main.py 保持完全一致) ---
    SOURCE_DATA_PATH = r'C:\Users\JC\Desktop\shumo\datasets\converted\source'
    BASE_OUTPUT_DIR = r'C:\Users\JC\Desktop\shumo\datasets\output_results'
    METADATA_CSV_PATH = os.path.join(BASE_OUTPUT_DIR, 'metadata', 'metadata_summary.csv')
    
    # 窗口化参数，用于计算样本总数
    WINDOW_SIZE = 4096
    OVERLAP = 0.5
    STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP))

    # --- 2. 加载或创建元数据 ---
    print("--- 正在加载数据集元数据 ---")
    if not os.path.exists(METADATA_CSV_PATH):
        print(f"元数据文件不存在，正在调用 utils_loader.py 生成...")
        # 调用您项目中的函数来创建元数据文件
        metadata_df = create_structured_dataset(root_path=SOURCE_DATA_PATH, save_path=METADATA_CSV_PATH)
    else:
        print("直接加载已存在的元数据文件。")
        metadata_df = pd.read_csv(METADATA_CSV_PATH)
    
    print("元数据加载成功。")

    # --- 3. 筛选出仅包含驱动端(DE)数据的记录 ---
    de_metadata_df = metadata_df.dropna(subset=['DE_path']).copy()
    print(f"\n总共找到 {len(de_metadata_df)} 个包含驱动端(DE)信号的原始数据文件。")

    # --- 4. 统计核心信息：文件数 ---
    #    这是论文中最基础也最重要的表格之一
    print("\n" + "="*50)
    print("      源域数据集驱动端(DE)文件数量统计")
    print("="*50)
    
    # 按故障类型和采样率进行分组计数
    file_counts = de_metadata_df.groupby(['Fault_Type', 'Sampling_Rate']).size().reset_index(name='File_Count')
    
    # 为了更美观地展示，我们使用数据透视表
    file_counts_pivot = file_counts.pivot_table(index='Fault_Type', columns='Sampling_Rate', values='File_Count', fill_value=0)
    
    print(file_counts_pivot)
    print("-"*50)


    # --- 5. 统计核心信息：样本数（切窗后） ---
    #    这个指标更能反应模型实际训练时使用的数据量
    print("\n" + "="*60)
    print(f"      源域驱动端(DE)样本数量统计 (Window Size: {WINDOW_SIZE})")
    print("="*60)

    # 为了避免重复计算，我们只计算一次样本数
    # 这里我们不能直接用 preprocess_from_metadata, 因为它会加载所有信号消耗大量内存
    # 我们只模拟计算过程
    
    de_metadata_df['sample_count'] = 0 # 初始化样本计数列

    for index, row in de_metadata_df.iterrows():
        # 估算信号长度。CWRU数据集中12k采样率的文件长度约为120k点，48k的约为480k点。
        # 这是一个合理的估算，避免了读取所有文件带来的巨大开销。
        signal_length = 120000 if row['Sampling_Rate'] == 12000 else 480000
        
        # 计算经过重采样到32kHz后的长度
        resampled_length = int(signal_length * 32000 / row['Sampling_Rate'])
        
        # 计算可以切出多少个窗口（样本）
        num_samples = (resampled_length - WINDOW_SIZE) // STEP_SIZE + 1
        
        de_metadata_df.loc[index, 'sample_count'] = num_samples

    # 对计算出的样本数进行分组求和
    sample_counts = de_metadata_df.groupby(['Fault_Type', 'Sampling_Rate'])['sample_count'].sum().reset_index()
    
    # 同样使用数据透视表进行展示
    sample_counts_pivot = sample_counts.pivot_table(index='Fault_Type', columns='Sampling_Rate', values='sample_count', fill_value=0)

    print(sample_counts_pivot)
    print("注意: 样本数是基于信号长度估算得出的窗口切片总数。")
    print("-"*60)


if __name__ == '__main__':
    analyze_and_print_stats()