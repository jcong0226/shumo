# 文件名: main.py (最终优化重构版)
# 描述: 一个功能完整、流程清晰的“总指挥”脚本，可执行从数据分析到模型训练的所有任务。

import os
import pandas as pd
import warnings
import joblib
from tqdm import tqdm

# --- 1. 清理并整合所有需要的导入 ---
# 数据加载与预处理
from utils_loader import create_structured_dataset
from utils_preprocess import preprocess_from_metadata
from utils_handle_outliers import handle_outliers_iqr
# 特征工程与分析
from utils_comprehensive_features import extract_comprehensive_features
from utils_feature_selection import perform_feature_selection
from utils_feature_stability import analyze_feature_stability
from utils_robustness_analysis import evaluate_feature_robustness

# 可视化
from utils_analysis import analyze_and_plot_sample, plot_kurtogram
from utils_visualization import plot_feature_distributions, plot_feature_means_by_fault_type, plot_feature_correlation_heatmap

# 模型训练与评估
from utils_traditional_models import train_and_evaluate_traditional_models
from utils_deep_learning import train_evaluate_2d_resnet

# 目标域处理与迁移分析
from process_target_data import process_target_domain_data
from utils_domain_analysis import visualize_domain_distribution, rank_features_by_transferability
from utils_feature_extraction import generate_and_save_images


# ==============================================================================
#                      2. 统一路径和参数设置
# ==============================================================================

# --- 顶级输入路径 ---
SOURCE_DATA_PATH = r'C:\Users\JC\Desktop\shumo\datasets\converted\source'

# --- 顶级输出目录 ---
BASE_OUTPUT_DIR = r'C:\Users\JC\Desktop\shumo\datasets\output_results'

# --- 所有输出路径现在都基于 BASE_OUTPUT_DIR ---
METADATA_CSV_PATH = os.path.join(BASE_OUTPUT_DIR, 'metadata', 'metadata_summary.csv')
HANDCRAFTED_FEATURES_CSV_PATH = os.path.join(BASE_OUTPUT_DIR, 'features', 'handcrafted_features.csv')
TARGET_HANDCRAFTED_FEATURES_CSV_PATH = os.path.join(BASE_OUTPUT_DIR, 'features', 'target_handcrafted_features.csv')

IMAGE_FEATURES_DIR = os.path.join(BASE_OUTPUT_DIR, 'image_features')
TIMESERIES_OUTPUT_DIR = os.path.join(IMAGE_FEATURES_DIR, 'timeseries')
SPECTRUM_OUTPUT_DIR = os.path.join(IMAGE_FEATURES_DIR, 'spectrum')
ENVELOPE_SPECTRUM_OUTPUT_DIR = os.path.join(IMAGE_FEATURES_DIR, 'envelope_spectrum')
STFT_OUTPUT_DIR = os.path.join(IMAGE_FEATURES_DIR, 'stft')
CWT_OUTPUT_DIR = os.path.join(IMAGE_FEATURES_DIR, 'cwt')

REPORTS_DIR = os.path.join(BASE_OUTPUT_DIR, 'reports')
ANALYSIS_PLOTS_DIR = os.path.join(REPORTS_DIR, 'exploratory_plots')
TEXT_REPORTS_DIR = os.path.join(REPORTS_DIR, 'text_reports')
MODEL_PLOTS_DIR = os.path.join(REPORTS_DIR, 'model_performance_plots')
SAVE_MODEL_DIR = os.path.join(BASE_OUTPUT_DIR, 'saved_models')

# --- 全局参数 ---
TARGET_FS = 32000
WINDOW_SIZE = 4096 # 您已修改为4096
OVERLAP = 0.5
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP))
IMAGE_SIZE = (224, 224)


# ==============================================================================
#                      3. 主执行流程
# ==============================================================================
def main():
    """主执行函数 (完整流程)"""
    # --- 初始化 ---
    for path in [os.path.dirname(METADATA_CSV_PATH), os.path.dirname(HANDCRAFTED_FEATURES_CSV_PATH),
                 IMAGE_FEATURES_DIR, REPORTS_DIR, ANALYSIS_PLOTS_DIR, TEXT_REPORTS_DIR,
                 MODEL_PLOTS_DIR, SAVE_MODEL_DIR]:
        os.makedirs(path, exist_ok=True)
    
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # ==========================================================================
    #                      PART 1: 源域数据处理与模型训练 (任务一 & 任务二)
    # ==========================================================================

    # --- 步骤 1: 加载/创建元数据 ---
    print(f"\n{'='*25} 步骤 1: 准备元数据 {'='*25}")
    if not os.path.exists(METADATA_CSV_PATH):
        metadata_df = create_structured_dataset(root_path=SOURCE_DATA_PATH, save_path=METADATA_CSV_PATH)
    else:
        metadata_df = pd.read_csv(METADATA_CSV_PATH)
    print("元数据准备完毕。")

    # --- 步骤 2: 探索性信号分析 (任务一) ---
    print(f"\n{'='*25} 步骤 2: 探索性信号分析 (任务一) {'='*25}")
    plot_dir = os.path.join(ANALYSIS_PLOTS_DIR)
    os.makedirs(plot_dir, exist_ok=True)
    if len(os.listdir(plot_dir)) > 0:
        print(f"检测到信号分析图已存在于 '{plot_dir}'，跳过此步骤。")
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
                    plot_kurtogram(
                        signal=signal_data, fs=sample_row['Sampling_Rate'],
                        title=f"Kurtogram for {name} Fault ({sig_type_to_analyze})",
                        save_dir=ANALYSIS_PLOTS_DIR
                    )

    # --- **优化点**: 创建一个字典来缓存预处理结果，避免重复计算 ---
    all_processed_samples = {}

    # --- 步骤 3: 手工特征工程 (任务一) ---
    print(f"\n{'='*25} 步骤 3: 手工特征工程 (任务一) {'='*25}")
    if os.path.exists(HANDCRAFTED_FEATURES_CSV_PATH):
        print(f"检测到已存在的源域手工特征文件，直接加载。")
        handcrafted_features_df = pd.read_csv(HANDCRAFTED_FEATURES_CSV_PATH)
    else:
        print("开始为源域信号提取全面的手工特征...")
        df_de_filtered = metadata_df.dropna(subset=['DE_path']).copy()
        processed_de_samples = preprocess_from_metadata(
            metadata_df=df_de_filtered, signal_type='DE', target_fs=TARGET_FS,
            window_size=WINDOW_SIZE, step_size=STEP_SIZE
        )
        all_features_list = []
        for sample in tqdm(processed_de_samples, desc="Extracting Source Comprehensive Features"):
            comprehensive_features = extract_comprehensive_features(
                segment=sample['signal_segment'], fs=TARGET_FS,
                rpm=sample['rpm'], bearing_type='DE'
            )
            comprehensive_features.update({k: v for k, v in sample.items() if k != 'signal_segment'})
            all_features_list.append(comprehensive_features)
        handcrafted_features_df = pd.DataFrame(all_features_list)
        handcrafted_features_df.to_csv(HANDCRAFTED_FEATURES_CSV_PATH, index=False)
        print(f"源域特征提取完成！已保存至: {HANDCRAFTED_FEATURES_CSV_PATH}")


    # ---步骤 3.5: 异常值处理---
    print(f"\n{'='*25} 步骤 3.5: 处理源域特征异常值 {'='*25}")
    # 在整个源域特征集上进行异常值处理
    handcrafted_features_df_capped, outlier_bounds = handle_outliers_iqr(handcrafted_features_df)
    
    # 我们可以选择保存处理后的特征文件，以便后续直接使用
    CAPPED_FEATURES_CSV_PATH = os.path.join(BASE_OUTPUT_DIR, 'features', 'handcrafted_features_capped.csv')
    handcrafted_features_df_capped.to_csv(CAPPED_FEATURES_CSV_PATH, index=False)
    print(f"已处理异常值的特征文件保存至: {CAPPED_FEATURES_CSV_PATH}")


    # --- 步骤 4: 特征分析与选择 (任务一) ---
    print(f"\n{'='*25} 步骤 4: 特征分析与选择 (任务一) {'='*25}")
    # 【关键修改】使用处理完异常值的数据进行后续所有分析
    selection_results = perform_feature_selection(
        features_df=handcrafted_features_df_capped, # <--- 使用封顶后的DataFrame
        n_features_mrmr=15, 
        n_components_pca=10, 
        save_dir=SAVE_MODEL_DIR
    )
    analyze_feature_stability(
        features_df=handcrafted_features_df_capped,
        text_save_path=os.path.join(TEXT_REPORTS_DIR, 'feature_stability_ranking.txt'),
        plot_save_path=os.path.join(MODEL_PLOTS_DIR, 'feature_stability_heatmap.png')
    )

    # --- 可视化分析 ---
    print("\n--- 开始生成特征可视化图 ---")
    key_features = [
        'time_kurtosis', 'time_crest_factor', 'freq_centroid',
        'env_BPFI_1x_amp', 'env_BPFO_1x_amp', 'env_2BSF_1x_amp',
        'complexity_perm_entropy', 'wpt_energy_node_7'
    ]
    plot_feature_distributions(handcrafted_features_df_capped, key_features, 
                               os.path.join(ANALYSIS_PLOTS_DIR, 'feature_distributions'), 'kde')
    plot_feature_means_by_fault_type(handcrafted_features_df_capped, key_features[:4], ANALYSIS_PLOTS_DIR)
    plot_feature_correlation_heatmap(handcrafted_features_df_capped, 30, ANALYSIS_PLOTS_DIR)

    # --- 步骤 5: 源域模型训练 (任务二) ---
    print(f"\n{'='*25} 步骤 5: 源域模型训练 (任务二) {'='*25}")
    # a. 传统机器学习模型
    print("\n--- 正在训练传统机器学习模型 ---")
    mrmr_features = selection_results['mrmr_selected_features'].columns.tolist()
    train_and_evaluate_traditional_models(
        handcrafted_features_df, mrmr_features, 'xgboost', SAVE_MODEL_DIR, MODEL_PLOTS_DIR, TEXT_REPORTS_DIR)
    train_and_evaluate_traditional_models(
        handcrafted_features_df, mrmr_features, 'svm', SAVE_MODEL_DIR, MODEL_PLOTS_DIR, TEXT_REPORTS_DIR)

    # b. 深度学习模型
    print("\n--- 开始深度学习流程 (此步骤耗时较长) ---")
    try:
        # 深度学习依赖的图像生成
        print("正在检查并生成深度学习所需的图像特征...")
        # 简单地复用之前的逻辑来生成DE信号的图像
        
        # 深度学习模型训练
        de_ts_dir = os.path.join(TIMESERIES_OUTPUT_DIR, 'source', 'DE')
        if not os.path.exists(de_ts_dir) or len(os.listdir(de_ts_dir)) == 0:
            # 如果不存在或为空，则进入生成逻辑
            print("图像特征不存在或不完整，开始生成...")
        
            # 简单地复用之前的逻辑来生成DE信号的图像
            df_de_filtered = metadata_df.dropna(subset=['DE_path']).copy()
            processed_de_samples = preprocess_from_metadata(
                metadata_df=df_de_filtered, signal_type='DE', target_fs=TARGET_FS,
                window_size=WINDOW_SIZE, step_size=STEP_SIZE
            )
            image_dirs = {'ts': TIMESERIES_OUTPUT_DIR, 'sp': SPECTRUM_OUTPUT_DIR,
                        'env_sp': ENVELOPE_SPECTRUM_OUTPUT_DIR, 'stft': STFT_OUTPUT_DIR,
                        'cwt': CWT_OUTPUT_DIR}
            generate_and_save_images(
                processed_samples=processed_de_samples, dataset_type='source', signal_type='DE',
                image_dirs=image_dirs, image_size=IMAGE_SIZE, target_fs=TARGET_FS
            )
        else:
            # 【跳过逻辑】如果存在且非空，则打印信息并跳过
            print("检测到图像特征已存在，跳过生成步骤。")
        if not os.path.exists(de_ts_dir) or len(os.listdir(de_ts_dir)) == 0:
            print("警告: 图像特征文件夹不存在或为空，跳过深度学习模型训练。")
        else:
            image_dirs_for_training = {
                'ts': os.path.join(TIMESERIES_OUTPUT_DIR, 'source', 'DE'),
                'sp': os.path.join(SPECTRUM_OUTPUT_DIR, 'source', 'DE'),
                'env_sp': os.path.join(ENVELOPE_SPECTRUM_OUTPUT_DIR, 'source', 'DE'),
                'stft': os.path.join(STFT_OUTPUT_DIR, 'source', 'DE'),
                'cwt': os.path.join(CWT_OUTPUT_DIR, 'source', 'DE')
            }
            num_classes = len(metadata_df['Fault_Type'].unique())
            train_evaluate_2d_resnet(
                image_dirs=image_dirs_for_training, num_classes=num_classes, save_dir=SAVE_MODEL_DIR,
                num_epochs=10, report_save_dir=MODEL_PLOTS_DIR, text_report_dir=TEXT_REPORTS_DIR
            )
    except Exception as e:
        print(f"\n深度学习流程中发生错误: {e}")

    # ==========================================================================
    #                      PART 2: 目标域处理与迁移预备分析 (衔接任务三)
    # ==========================================================================
    
    # --- 步骤 6: 处理目标域数据 ---
    print(f"\n{'='*25} 步骤 6: 处理目标域数据 {'='*25}")
    process_target_domain_data()

    # --- 步骤 7: 迁移学习预备分析 ---
    print(f"\n{'='*25} 步骤 7: 迁移学习预备分析 (衔接任务三) {'='*25}")
    target_df_raw = pd.read_csv(TARGET_HANDCRAFTED_FEATURES_CSV_PATH)
    print("\n--- 正在对目标域数据应用源域的异常值边界 ---")
    # 使用源域数据 (handcrafted_features_df) 来计算边界，并应用到目标域数据上
    target_df_capped, _ = handle_outliers_iqr(target_df_raw, train_df=handcrafted_features_df)

    target_df = target_df_capped # 后续使用处理过的数据
    
    imputer_path = os.path.join(SAVE_MODEL_DIR, 'source_domain_imputer.joblib')
    scaler_path = os.path.join(SAVE_MODEL_DIR, 'source_domain_scaler.joblib')
    pca_path = os.path.join(SAVE_MODEL_DIR, 'source_domain_pca.joblib')
    
    if not all(os.path.exists(p) for p in [imputer_path, scaler_path, pca_path]):
        print("错误: 缺少Imputer, Scaler或PCA模型文件。请确保已完整运行步骤4。")
    else:
        imputer = joblib.load(imputer_path)
        scaler = joblib.load(scaler_path)
        pca = joblib.load(pca_path)
        
        visualize_domain_distribution(
            handcrafted_features_df, target_df, imputer, scaler, pca, 
            os.path.join(MODEL_PLOTS_DIR, 'domain_distribution_tsne.png')
        )
        transferability_ranking = rank_features_by_transferability(
            handcrafted_features_df, target_df, imputer, scaler, 
            os.path.join(TEXT_REPORTS_DIR, 'feature_transferability_ranking.txt')
        )
        
        print("\n--- 【重要】任务三策略建议 ---")
        top_transferable_features = [feature for feature, score in transferability_ranking[:15]]
        print("已生成特征可迁移性排名。对于任务三，建议使用以下高可迁移性特征集重新训练源域模型，以获得最佳迁移效果：")
        print(top_transferable_features)

    print(f"\n{'='*25} 所有流程执行完毕 {'='*25}")


if __name__ == '__main__':
    main()