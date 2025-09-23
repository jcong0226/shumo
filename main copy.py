# 文件名: main.py (优化版)
import os
import pandas as pd
import warnings
from tqdm import tqdm
import joblib
from utils_domain_analysis import visualize_domain_distribution, rank_features_by_transferability
# --- 从模块化文件中导入所有需要的函数 ---
# (假设您的文件结构和函数导入是正确的)
from utils_loader import create_structured_dataset
from utils_visualization import plot_feature_distributions, plot_feature_means_by_fault_type, plot_feature_correlation_heatmap
from utils_preprocess import preprocess_from_metadata
from utils_analysis import analyze_and_plot_sample, analyze_and_compare_stft_cwt, plot_kurtogram
from utils_feature_stability import analyze_feature_stability
from utils_handcrafted_features import extract_handcrafted_features
# 建议将以下两个函数放入新的 utils_feature_processing.py 文件中
from utils_feature_selection import perform_feature_selection
from utils_robustness_analysis import evaluate_feature_robustness
from utils_traditional_models import train_and_evaluate_traditional_models
from utils_deep_learning import train_evaluate_2d_resnet
from utils_comprehensive_features import extract_comprehensive_features
from utils_transfer_preparation import calculate_domain_discrepancy, aggregate_window_predictions
from utils_feature_extraction import (
    create_timeseries_image, 
    create_spectrum_image, 
    create_envelope_spectrum_image_sk,
    generate_and_save_images
)
from utils_advanced_features import extract_all_advanced_features

# # --- 定义唯一的顶级输出目录 ---
# BASE_OUTPUT_DIR = r'C:\Users\JC\Desktop\shumo\datasets\output_results'
# # --- 1. 设置所有路径和参数 ---
# # (此部分与您的代码完全一致，保持不变)
# SAVE_MODEL_DIR = r'C:\Users\JC\Desktop\shumo\datasets\saved_models'
# SOURCE_DATA_PATH = r'C:\Users\JC\Desktop\shumo\datasets\converted\source'
# METADATA_CSV_PATH = r'C:\Users\JC\Desktop\shumo\datasets\metadata_summary.csv'
# ANALYSIS_PLOTS_DIR = r'C:\Users\JC\Desktop\shumo\datasets\analysis_plots'
# TIMESERIES_OUTPUT_DIR = r'C:\Users\JC\Desktop\shumo\datasets\timeseries_images'
# SPECTRUM_OUTPUT_DIR = r'C:\Users\JC\Desktop\shumo\datasets\spectrum_images'
# ENVELOPE_SPECTRUM_OUTPUT_DIR = r'C:\Users\JC\Desktop\shumo\datasets\envelope_spectrum_images'
# HANDCRAFTED_FEATURES_CSV_PATH = r'C:\Users\JC\Desktop\shumo\datasets\handcrafted_features.csv'
# STFT_OUTPUT_DIR = r'C:\Users\JC\Desktop\shumo\datasets\stft_images'
# CWT_OUTPUT_DIR = r'C:\Users\JC\Desktop\shumo\datasets\cwt_images'
# # --- 准备目标域数据 ---
# TARGET_DATA_PATH = r'C:\Users\JC\Desktop\shumo\datasets\converted\target'
# TARGET_HANDCRAFTED_FEATURES_CSV_PATH = r'C:\Users\JC\Desktop\shumo\datasets\target_handcrafted_features.csv'
# TARGET_FS = 32000
# WINDOW_SIZE = 2048
# OVERLAP = 0.5
# STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP))
# IMAGE_SIZE = (224, 224)

# ==============================================================================
#                      1. 统一路径和参数设置
# ==============================================================================

# --- 顶级输入路径 ---
SOURCE_DATA_PATH = r'C:\Users\JC\Desktop\shumo\datasets\converted\source'
TARGET_DATA_PATH = r'C:\Users\JC\Desktop\shumo\datasets\converted\target'

# --- 【核心修改】定义唯一的顶级输出目录 ---
BASE_OUTPUT_DIR = r'C:\Users\JC\Desktop\shumo\datasets\output_results'

# --- 所有输出路径现在都基于 BASE_OUTPUT_DIR ---
# 1. CSV 数据路径
METADATA_CSV_PATH = os.path.join(BASE_OUTPUT_DIR, 'metadata', 'metadata_summary.csv')
TARGET_METADATA_CSV_PATH = os.path.join(BASE_OUTPUT_DIR, 'metadata', 'target_metadata_summary.csv')
HANDCRAFTED_FEATURES_CSV_PATH = os.path.join(BASE_OUTPUT_DIR, 'features', 'handcrafted_features.csv')
TARGET_HANDCRAFTED_FEATURES_CSV_PATH = os.path.join(BASE_OUTPUT_DIR, 'features', 'target_handcrafted_features.csv')

# 2. 图像特征路径
IMAGE_FEATURES_DIR = os.path.join(BASE_OUTPUT_DIR, 'image_features')
TIMESERIES_OUTPUT_DIR = os.path.join(IMAGE_FEATURES_DIR, 'timeseries')
SPECTRUM_OUTPUT_DIR = os.path.join(IMAGE_FEATURES_DIR, 'spectrum')
ENVELOPE_SPECTRUM_OUTPUT_DIR = os.path.join(IMAGE_FEATURES_DIR, 'envelope_spectrum')
STFT_OUTPUT_DIR = os.path.join(IMAGE_FEATURES_DIR, 'stft')
CWT_OUTPUT_DIR = os.path.join(IMAGE_FEATURES_DIR, 'cwt')

# 3. 分析与报告路径
REPORTS_DIR = os.path.join(BASE_OUTPUT_DIR, 'reports')
ANALYSIS_PLOTS_DIR = os.path.join(REPORTS_DIR, 'exploratory_plots')
TEXT_REPORTS_DIR = os.path.join(REPORTS_DIR, 'text_reports')
MODEL_PLOTS_DIR = os.path.join(REPORTS_DIR, 'model_performance_plots')

# 4. 模型保存路径
SAVE_MODEL_DIR = os.path.join(BASE_OUTPUT_DIR, 'saved_models')

# --- 全局参数 ---
TARGET_FS = 32000
WINDOW_SIZE = 4096
OVERLAP = 0.5
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP))
IMAGE_SIZE = (224, 224)

# --- 主执行流程 ---
def main():
    """主执行函数"""
    # 确保所有顶级输出目录都存在
    for path in [os.path.dirname(METADATA_CSV_PATH), os.path.dirname(HANDCRAFTED_FEATURES_CSV_PATH),
                 IMAGE_FEATURES_DIR, REPORTS_DIR, ANALYSIS_PLOTS_DIR, TEXT_REPORTS_DIR,
                 MODEL_PLOTS_DIR, SAVE_MODEL_DIR]:
        os.makedirs(path, exist_ok=True)

    # ==================== 步骤 1: 准备元数据 ====================
    print(f"\n{'='*25} 步骤 1: 准备元数据 {'='*25}")
    if not os.path.exists(METADATA_CSV_PATH):
        print("元数据文件不存在，正在创建...")
        metadata_df = create_structured_dataset(root_path=SOURCE_DATA_PATH, save_path=METADATA_CSV_PATH)
    else:
        print("检测到已存在的元数据文件，直接加载。")
        metadata_df = pd.read_csv(METADATA_CSV_PATH)
    print("元数据准备完毕。")

    # ==================== 步骤 2: 探索性特征分析 ====================
    print(f"\n{'='*25} 步骤 2: 探索性信号分析 {'='*25}")
    plot_dir = os.path.join(ANALYSIS_PLOTS_DIR, "signal_analysis_plots")
    os.makedirs(plot_dir, exist_ok=True)
    if len(os.listdir(plot_dir)) > 0:
        print(f"检测到信号分析图已存在，跳过此步骤。")
    else:
        print("开始生成分析图...")
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        for sig_type_to_analyze in ['DE', 'FE']:
            path_col = f'{sig_type_to_analyze}_path'
            analysis_df = metadata_df.dropna(subset=[path_col])
            if analysis_df.empty: 
                continue
            samples_to_analyze = {
            'Normal': analysis_df[analysis_df['Fault_Type'] == 'Normal'].iloc[0],
            'Inner_Race': analysis_df[analysis_df['Fault_Type'] == 'Inner_Race'].iloc[0],
            'Outer_Race': analysis_df[analysis_df['Fault_Type'] == 'Outer_Race'].iloc[0],
            'Ball': analysis_df[analysis_df['Fault_Type'] == 'Ball'].iloc[0]
            }

            for name, sample_row in samples_to_analyze.items():
                signal_path = sample_row[path_col]
                if pd.notna(signal_path) and os.path.exists(signal_path):
                    signal_data = pd.read_csv(signal_path, header=None).iloc[:, 0].values
                    
                    # 调用原始的详细分析函数
                    analyze_and_plot_sample(
                        signal=signal_data, fs=sample_row['Sampling_Rate'], rpm=sample_row['RPM'],
                        title=f"Detailed Analysis of {name} Fault ({sig_type_to_analyze})",
                        signal_type=sig_type_to_analyze, save_dir=ANALYSIS_PLOTS_DIR
                    )

                    # 【新增调用】调用对比函数
                    analyze_and_compare_stft_cwt(
                        signal=signal_data, fs=sample_row['Sampling_Rate'],
                        title=f"STFT vs CWT for {name} Fault ({sig_type_to_analyze})",
                        save_dir=ANALYSIS_PLOTS_DIR
                    )
                    plot_kurtogram(
                        signal=signal_data, fs=sample_row['Sampling_Rate'],
                        title=f"Kurtogram for {name} Fault ({sig_type_to_analyze})",
                        save_dir=ANALYSIS_PLOTS_DIR
                    )

    # --- **优化点**: 创建一个字典来缓存预处理结果，避免重复计算 ---
    all_processed_samples = {}
    
    # ==================== 步骤 3: 系统化图像特征提取 ====================
    print(f"\n{'='*25} 步骤 3: 系统化图像特征提取 {'='*25}")
    # SIGNAL_TYPES_TO_PROCESS = ['DE', 'FE', 'BA']
    SIGNAL_TYPES_TO_PROCESS = ['DE']
    for sig_type in SIGNAL_TYPES_TO_PROCESS:
        print(f"\n--- 检查并处理 {sig_type} 信号的图像特征 ---")
        path_column = f'{sig_type}_path'
        df_filtered = metadata_df.dropna(subset=[path_column]).copy()
        
        if df_filtered.empty:
            print(f"元数据中没有找到有效的 {sig_type} 信号数据，跳过。"); continue
            
        ts_dir = os.path.join(TIMESERIES_OUTPUT_DIR, 'source', sig_type)
        sp_dir = os.path.join(SPECTRUM_OUTPUT_DIR, 'source', sig_type)
        env_sp_dir = os.path.join(ENVELOPE_SPECTRUM_OUTPUT_DIR, 'source', sig_type)
        stft_dir = os.path.join(STFT_OUTPUT_DIR, 'source', sig_type) # 新增
        cwt_dir = os.path.join(CWT_OUTPUT_DIR, 'source', sig_type)   # 新增

        if (os.path.exists(ts_dir) and len(os.listdir(ts_dir)) > 0 and
            os.path.exists(sp_dir) and len(os.listdir(sp_dir)) > 0 and
            os.path.exists(env_sp_dir) and len(os.listdir(env_sp_dir)) > 0 and
            os.path.exists(stft_dir) and len(os.listdir(stft_dir)) > 0 and # 新增检查
            os.path.exists(cwt_dir) and len(os.listdir(cwt_dir)) > 0):    # 新增检查
            print(f"检测到 {sig_type} 信号的所有五种特征图像均已存在，跳过生成。")
            continue
        
        print(f"开始为 {len(df_filtered)} 个 {sig_type} 信号实验生成特征图像...")
        processed_samples = preprocess_from_metadata(
            metadata_df=df_filtered, signal_type=sig_type, target_fs=TARGET_FS,
            window_size=WINDOW_SIZE, step_size=STEP_SIZE
        )
        # **优化点**: 缓存预处理结果
        all_processed_samples[sig_type] = processed_samples
        
        if processed_samples:
            # 【修改】将所有路径打包成一个字典传入
            image_dirs_to_generate = {
                'ts': TIMESERIES_OUTPUT_DIR,
                'sp': SPECTRUM_OUTPUT_DIR,
                'env_sp': ENVELOPE_SPECTRUM_OUTPUT_DIR,
                'stft': STFT_OUTPUT_DIR,
                'cwt': CWT_OUTPUT_DIR
            }
            generate_and_save_images(
                processed_samples=processed_samples, dataset_type='source', signal_type=sig_type,
                image_dirs=image_dirs_to_generate, # 传入字典
                image_size=IMAGE_SIZE, target_fs=TARGET_FS
            )
            
    # ==================== 步骤 4: 系统化手工特征提取 ====================
    print(f"\n{'='*25} 步骤 4: 系统化手工特征提取 (全面版) {'='*25}")
    if os.path.exists(HANDCRAFTED_FEATURES_CSV_PATH):
        print(f"检测到已存在的手工特征文件 '{HANDCRAFTED_FEATURES_CSV_PATH}'，跳过此步骤。")
    else:
        print("开始提取全面手工特征 (主要基于DE信号)...")
        sig_type_for_handcrafted = 'DE'
        
        if sig_type_for_handcrafted in all_processed_samples:
            print("复用步骤3中已完成的预处理数据。")
            processed_de_samples = all_processed_samples[sig_type_for_handcrafted]
        else:
            print("步骤3被跳过，正在按需为DE信号执行预处理...")
            df_de_filtered = metadata_df.dropna(subset=[f'{sig_type_for_handcrafted}_path']).copy()
            processed_de_samples = preprocess_from_metadata(
                metadata_df=df_de_filtered, signal_type=sig_type_for_handcrafted, target_fs=TARGET_FS,
                window_size=WINDOW_SIZE, step_size=STEP_SIZE
            ) if not df_de_filtered.empty else []

        if not processed_de_samples:
            print("没有可用于提取手工特征的DE信号样本。")
        else:
            all_features_list = []
            for sample in tqdm(processed_de_samples, desc="Extracting Comprehensive Features"):
                # 【核心修改】只调用这一个函数
                comprehensive_features = extract_comprehensive_features(
                    segment=sample['signal_segment'], 
                    fs=TARGET_FS,
                    rpm=sample['rpm'], 
                    bearing_type=sig_type_for_handcrafted
                )
                
                # 合并特征和元数据
                comprehensive_features.update({k: v for k, v in sample.items() if k != 'signal_segment'})
                all_features_list.append(comprehensive_features)

            # 将结果转换为DataFrame并保存
            features_df = pd.DataFrame(all_features_list)
            features_df.to_csv(HANDCRAFTED_FEATURES_CSV_PATH, index=False)
            print(f"所有特征提取完成！共提取 {len(features_df.columns) - 7} 个特征。") # -7 for metadata columns
            print(f"已保存至: {HANDCRAFTED_FEATURES_CSV_PATH}")
    # ==================== 步骤 5: 特征选择与稳健性评估 (思路D) ====================
    # **优化点**: 修正了步骤编号
    print(f"\n{'='*25} 步骤 5: 特征选择与稳健性评估 {'='*25}")
    if not os.path.exists(HANDCRAFTED_FEATURES_CSV_PATH):
        print(f"错误: 手工特征文件 '{HANDCRAFTED_FEATURES_CSV_PATH}' 不存在，无法执行此步骤。")
    else:
        # (此部分逻辑正确，保持不变)
        print(f"正在加载手工特征文件: {HANDCRAFTED_FEATURES_CSV_PATH}")
        handcrafted_features_df = pd.read_csv(HANDCRAFTED_FEATURES_CSV_PATH)
        selection_results = perform_feature_selection(
            features_df=handcrafted_features_df, 
            n_features_mrmr=15, 
            n_components_pca=10,
            save_dir=SAVE_MODEL_DIR  # <--- 传入模型保存目录
        )
        # 筛选出的特征矩阵可用于后续的传统模型训练
        X_train_mrmr = selection_results['mrmr_selected_features']
        X_train_pca = selection_results['pca_transformed_features']
        y_train = selection_results['labels']
        if 'load' in handcrafted_features_df.columns:
            robustness_ranking = evaluate_feature_robustness(
                handcrafted_features_df,
                save_path=os.path.join(TEXT_REPORTS_DIR, 'feature_robustness_ranking.txt')
            )

        # 【修改】调用特征稳定性分析并保存文本和图像结果
        analyze_feature_stability(
            handcrafted_features_df,
            text_save_path=os.path.join(TEXT_REPORTS_DIR, 'feature_stability_ranking.txt'),
            plot_save_path=os.path.join(MODEL_PLOTS_DIR, 'feature_stability_heatmap.png')
        )
        # 【新增调用】可视化关键特征的分布
        print("\n--- 开始可视化关键特征分布 ---")
        # 选择一些您认为重要的或者在稳健性/稳定性分析中排名靠前的特征
        key_features_for_plotting = [
            'time_kurtosis',
            'time_crest_factor',
            'freq_centroid',
            'env_BPFI_peak_amp',
            'env_BPFO_peak_amp',
            'complexity_perm_entropy',
            'wpt_energy_ddd' # 示例：小波包能量的一个频带
        ]    
        # 指定一个新的目录来保存这些分布图
        feature_dist_plots_dir = os.path.join(ANALYSIS_PLOTS_DIR, 'feature_distributions')
        plot_feature_distributions(
            features_df=handcrafted_features_df,
            features_to_plot=key_features_for_plotting,
            save_dir=feature_dist_plots_dir,
            plot_type='kde'  # 您可以改为 'hist' 来生成直方图
        )
            # 【新增调用】定量对比特征均值
        print("\n--- 开始定量对比关键特征均值 ---")
        # 我们可以复用之前的关键特征列表，或者选择新的
        key_features_for_comparison = [
            'time_kurtosis',
            'time_crest_factor',
            'env_BPFI_1x_amp',
            'complexity_higuchi_fd'
        ]

        # 同样保存到分析图文件夹
        plot_feature_means_by_fault_type(
            features_df=handcrafted_features_df,
            features_to_compare=key_features_for_comparison,
            save_dir=ANALYSIS_PLOTS_DIR # 直接保存在主分析图目录下
        )
        print("\n--- 开始生成特征相关性热图 ---")
        plot_feature_correlation_heatmap(
            features_df=handcrafted_features_df,
            top_n_features=30, # 选择最前面的30个特征进行可视化，避免图像过于拥挤
            save_dir=ANALYSIS_PLOTS_DIR
        )
    print(f"\n{'='*25} 任务一所有步骤执行完毕 {'='*25}")
    
    # ==================== 步骤 6: 【新增】源域基线训练 (思路E) ====================
    print(f"\n{'='*25} 步骤 6: 源域基线训练 {'='*25}")
    os.makedirs(SAVE_MODEL_DIR, exist_ok=True)
    # a. 训练传统模型
    if not os.path.exists(HANDCRAFTED_FEATURES_CSV_PATH):
        print("警告: 手工特征文件不存在，跳过传统模型训练。")
    else:
        # 假设 selection_results 变量在步骤5中已生成
        if selection_results:
            # **修正3**: 正确准备调用 train_and_evaluate_traditional_models 所需的参数
            # 准备mRMR特征和标签
            mrmr_features_names = selection_results['mrmr_selected_features'].columns.tolist()
            # 准备PCA特征和标签
            pca_features_names = selection_results['pca_transformed_features'].columns.tolist()
            # --- 第1组实验: 使用 mRMR 特征集 ---
            print("\n" + "*"*20 + " 正在使用 mRMR 特征集进行训练 " + "*"*20)
            train_and_evaluate_traditional_models(
                features_df=handcrafted_features_df, selected_feature_names=mrmr_features_names, 
                model_type='xgboost', save_dir=SAVE_MODEL_DIR,
                report_save_dir=MODEL_PLOTS_DIR, # 新增
                text_report_dir=TEXT_REPORTS_DIR  # 新增
            )
            train_and_evaluate_traditional_models(
                features_df=handcrafted_features_df, selected_feature_names=mrmr_features_names, 
                model_type='svm', save_dir=SAVE_MODEL_DIR,
                report_save_dir=MODEL_PLOTS_DIR, # 新增
                text_report_dir=TEXT_REPORTS_DIR  # 新增
            )
            # --- 第2组实验: 使用 PCA 特征集 ---
            print("\n" + "*"*20 + " 正在使用 PCA 特征集进行训练 " + "*"*20)
            # 注意：PCA特征已经经过了标准化，但为保持函数接口一致，仍传入原始df
            # 函数内部会重新标准化，这不影响结果，但保持了代码的一致性。
            # 更优的做法是修改训练函数，使其能接受已处理的数据。但当前方式更简单直接。
            # 创建一个包含了PCA结果和元数据的新DataFrame
            pca_df_for_training = pd.concat([
                handcrafted_features_df[['label', 'experiment_id']],
                selection_results['pca_transformed_features']
            ], axis=1)
            train_and_evaluate_traditional_models(
                features_df=pca_df_for_training, selected_feature_names=pca_features_names, 
                model_type='xgboost', save_dir=SAVE_MODEL_DIR,                
                report_save_dir=MODEL_PLOTS_DIR, # 新增
                text_report_dir=TEXT_REPORTS_DIR  # 新增
            )
            train_and_evaluate_traditional_models(
                features_df=pca_df_for_training, selected_feature_names=pca_features_names, 
                model_type='svm', save_dir=SAVE_MODEL_DIR,
                report_save_dir=MODEL_PLOTS_DIR, # 新增
                text_report_dir=TEXT_REPORTS_DIR  # 新增
            )
            # --- 第3组实验: 使用 稳健性排名最高的特征集 ---
            if robustness_ranking:
                print("\n" + "*"*20 + " 正在使用 稳健性排名最高的特征集 进行训练 " + "*"*20)
                top_robust_feature_names = [feature for feature, score in robustness_ranking[:15]]
                train_and_evaluate_traditional_models(
                    features_df=handcrafted_features_df, selected_feature_names=top_robust_feature_names, 
                    model_type='xgboost', save_dir=SAVE_MODEL_DIR,       
                    report_save_dir=MODEL_PLOTS_DIR, # 新增
                    text_report_dir=TEXT_REPORTS_DIR  # 新增
                )
                train_and_evaluate_traditional_models(
                    features_df=handcrafted_features_df, selected_feature_names=top_robust_feature_names, 
                    model_type='svm', save_dir=SAVE_MODEL_DIR,
                    report_save_dir=MODEL_PLOTS_DIR, # 新增
                    text_report_dir=TEXT_REPORTS_DIR  # 新增
                )
        else:
            print("警告: 特征选择结果未找到，无法训练传统模型。请确保步骤5已运行。")

    # b. 训练深度学习模型
    try:
        # 检查主通道的图像文件夹是否存在
        de_ts_dir = os.path.join(TIMESERIES_OUTPUT_DIR, 'source', 'DE')
        if not os.path.exists(de_ts_dir) or len(os.listdir(de_ts_dir)) == 0:
            print("警告: 图像特征文件夹不存在或为空，跳过深度学习模型训练。")
        else:
            # 【修改】构建包含5种图像路径的字典
            image_dirs_de_5_channel = {
                'ts': os.path.join(TIMESERIES_OUTPUT_DIR, 'source', 'DE'),
                'sp': os.path.join(SPECTRUM_OUTPUT_DIR, 'source', 'DE'),
                'env_sp': os.path.join(ENVELOPE_SPECTRUM_OUTPUT_DIR, 'source', 'DE'),
                'stft': os.path.join(STFT_OUTPUT_DIR, 'source', 'DE'),
                'cwt': os.path.join(CWT_OUTPUT_DIR, 'source', 'DE')
            }
            num_classes = len(metadata_df['Fault_Type'].unique())
            
            train_evaluate_2d_resnet(
                image_dirs=image_dirs_de_5_channel, 
                num_classes=num_classes, 
                save_dir=SAVE_MODEL_DIR,
                num_epochs=10,
                report_save_dir=MODEL_PLOTS_DIR, # 新增
                text_report_dir=TEXT_REPORTS_DIR  # 新增
            )
    except ImportError as e:
        print(f"\n警告: 导入PyTorch失败 ({e})，跳过深度学习模型训练。")
    except Exception as e:
        print(f"\n深度学习模型训练过程中发生错误: {e}")

    # ==================== 步骤 7: 【迁移学习预备分析】领域差异性分析 ====================
    print(f"\n{'='*25} 步骤 7: 迁移学习预备分析 {'='*25}")

    # 确保源域和目标域的特征文件都存在
    if not os.path.exists(HANDCRAFTED_FEATURES_CSV_PATH) or not os.path.exists(TARGET_HANDCRAFTED_FEATURES_CSV_PATH):
        print("错误: 缺少源域或目标域的特征文件，无法进行差异性分析。")
        print("请确保已运行 process_target_data.py 来生成目标域特征。")
    else:
        # 加载数据
        source_df = pd.read_csv(HANDCRAFTED_FEATURES_CSV_PATH)
        target_df = pd.read_csv(TARGET_HANDCRAFTED_FEATURES_CSV_PATH)

        # 加载在源域上训练好的Scaler和PCA模型
        scaler_path = os.path.join(SAVE_MODEL_DIR, 'source_domain_scaler.joblib')
        pca_path = os.path.join(SAVE_MODEL_DIR, 'source_domain_pca.joblib')

        if not os.path.exists(scaler_path) or not os.path.exists(pca_path):
            print("错误: 未找到已保存的source_domain_scaler.joblib或source_domain_pca.joblib。")
            print("请确保已完整运行步骤5。")
        else:
            scaler = joblib.load(scaler_path)
            pca = joblib.load(pca_path)

            # 1. 可视化整体领域分布
            tsne_save_path = os.path.join(MODEL_PLOTS_DIR, 'domain_distribution_tsne.png')
            visualize_domain_distribution(source_df, target_df, scaler, pca, tsne_save_path)

            # 2. 对每个特征的可迁移性进行排序
            ranking_save_path = os.path.join(TEXT_REPORTS_DIR, 'feature_transferability_ranking.txt')
            transferability_ranking = rank_features_by_transferability(source_df, target_df, scaler, ranking_save_path)

            # --- 【衔接任务3的关键】 ---
            print("\n--- 迁移学习策略建议 ---")
            print("已生成特征可迁移性排名报告。在任务3中，您应优先考虑使用MMD值较低的特征子集来构建迁移模型。")
            print("例如，可以选择排名前15或前20的特征，重新执行任务2中的模型训练流程，得到一个更适合迁移的源域模型。")

            # 示范：获取排名前15的最具可迁移性的特征
            top_15_transferable_features = [feature for feature, score in transferability_ranking[:15]]
            print("\n排名前15的可迁移特征为:")
            print(top_15_transferable_features)

    print(f"\n{'='*25} 迁移预备分析执行完毕 {'='*25}")
    print(f"\n{'='*25} 思路F执行完毕 {'='*25}")

if __name__ == '__main__':
    main()