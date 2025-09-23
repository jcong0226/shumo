# 文件名: run_traditional_ml.py
# 描述: 一个专门用于快速运行传统机器学习流程的脚本。

import os
import pandas as pd
import warnings
import joblib
from tqdm import tqdm

# --- 导入所有需要的函数 ---
from utils_loader import create_structured_dataset
from utils_preprocess import preprocess_from_metadata
from utils_comprehensive_features import extract_comprehensive_features
from utils_feature_selection import perform_feature_selection
from utils_robustness_analysis import evaluate_feature_robustness
from utils_feature_stability import analyze_feature_stability
from utils_traditional_models import train_and_evaluate_traditional_models
# 导入为迁移学习准备的分析函数
from utils_domain_analysis import visualize_domain_distribution, rank_features_by_transferability
# 导入目标域数据处理脚本的功能
from process_target_data import process_target_domain_data
from sklearn.impute import SimpleImputer


# ==============================================================================
#                      1. 统一路径和参数设置 (与main.py完全一致)
# ==============================================================================

# --- 顶级输入路径 ---
SOURCE_DATA_PATH = r'C:\Users\JC\Desktop\shumo\datasets\converted\source'

# --- 顶级输出目录 ---
BASE_OUTPUT_DIR = r'C:\Users\JC\Desktop\shumo\datasets\output_results'

# --- 所有输出路径现在都基于 BASE_OUTPUT_DIR ---
METADATA_CSV_PATH = os.path.join(BASE_OUTPUT_DIR, 'metadata', 'metadata_summary.csv')
TARGET_METADATA_CSV_PATH = os.path.join(BASE_OUTPUT_DIR, 'metadata', 'target_metadata_summary.csv')
HANDCRAFTED_FEATURES_CSV_PATH = os.path.join(BASE_OUTPUT_DIR, 'features', 'handcrafted_features.csv')
TARGET_HANDCRAFTED_FEATURES_CSV_PATH = os.path.join(BASE_OUTPUT_DIR, 'features', 'target_handcrafted_features.csv')
REPORTS_DIR = os.path.join(BASE_OUTPUT_DIR, 'reports')
TEXT_REPORTS_DIR = os.path.join(REPORTS_DIR, 'text_reports')
MODEL_PLOTS_DIR = os.path.join(REPORTS_DIR, 'model_performance_plots')
SAVE_MODEL_DIR = os.path.join(BASE_OUTPUT_DIR, 'saved_models')

# --- 全局参数 ---
TARGET_FS = 32000
WINDOW_SIZE = 4096
OVERLAP = 0.5
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP))

# --- 主执行流程 ---
def main_traditional():
    """专用于传统机器学习的主执行函数"""
    # 确保所有需要的输出目录都存在
    for path in [os.path.dirname(METADATA_CSV_PATH), os.path.dirname(HANDCRAFTED_FEATURES_CSV_PATH),
                 REPORTS_DIR, TEXT_REPORTS_DIR, MODEL_PLOTS_DIR, SAVE_MODEL_DIR]:
        os.makedirs(path, exist_ok=True)
    
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # ==================== 步骤 1: 准备源域元数据 ====================
    print(f"\n{'='*25} 步骤 1: 准备源域元数据 {'='*25}")
    if not os.path.exists(METADATA_CSV_PATH):
        metadata_df = create_structured_dataset(root_path=SOURCE_DATA_PATH, save_path=METADATA_CSV_PATH)
    else:
        metadata_df = pd.read_csv(METADATA_CSV_PATH)
    print("源域元数据准备完毕。")

    # ==================== 步骤 2: 提取源域手工特征 ====================
    print(f"\n{'='*25} 步骤 2: 提取源域手工特征 {'='*25}")
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

    # ==================== 步骤 3: 特征选择与分析 ====================
    print(f"\n{'='*25} 步骤 3: 特征选择与分析 {'='*25}")
    selection_results = perform_feature_selection(
        features_df=handcrafted_features_df, n_features_mrmr=15, 
        n_components_pca=10, save_dir=SAVE_MODEL_DIR
    )
    evaluate_feature_robustness(
        handcrafted_features_df,
        save_path=os.path.join(TEXT_REPORTS_DIR, 'feature_robustness_ranking.txt')
    )
    analyze_feature_stability(
        handcrafted_features_df,
        text_save_path=os.path.join(TEXT_REPORTS_DIR, 'feature_stability_ranking.txt'),
        plot_save_path=os.path.join(MODEL_PLOTS_DIR, 'feature_stability_heatmap.png')
    )
    
    # ==================== 步骤 4: 源域模型训练 (任务二) ====================
    print(f"\n{'='*25} 步骤 4: 源域模型训练 (任务二) {'='*25}")
    mrmr_features_names = selection_results['mrmr_selected_features'].columns.tolist()
    
    print("\n--- 正在使用 mRMR 筛选出的特征集进行训练 ---")
    train_and_evaluate_traditional_models(
        features_df=handcrafted_features_df, selected_feature_names=mrmr_features_names, 
        model_type='xgboost', save_dir=SAVE_MODEL_DIR,
        report_save_dir=MODEL_PLOTS_DIR, text_report_dir=TEXT_REPORTS_DIR
    )
    train_and_evaluate_traditional_models(
        features_df=handcrafted_features_df, selected_feature_names=mrmr_features_names, 
        model_type='svm', save_dir=SAVE_MODEL_DIR,
        report_save_dir=MODEL_PLOTS_DIR, text_report_dir=TEXT_REPORTS_DIR
    )

    # ==================== 步骤 5: 处理目标域数据 ====================
    print(f"\n{'='*25} 步骤 5: 处理目标域数据 {'='*25}")
    # 调用我们修正过的脚本来处理目标域数据，它会自动跳过已完成的步骤
    process_target_domain_data()

    # ==================== 步骤 6: 迁移学习预备分析 (衔接任务三) ====================
    print(f"\n{'='*25} 步骤 6: 迁移学习预备分析 (衔接任务三) {'='*25}")
    
    # 加载目标域数据
    target_df = pd.read_csv(TARGET_HANDCRAFTED_FEATURES_CSV_PATH)
    
    # --- 【核心修正】加载所有预处理模型：Imputer, Scaler, PCA ---
    imputer_path = os.path.join(SAVE_MODEL_DIR, 'source_domain_imputer.joblib')
    scaler_path = os.path.join(SAVE_MODEL_DIR, 'source_domain_scaler.joblib')
    pca_path = os.path.join(SAVE_MODEL_DIR, 'source_domain_pca.joblib')
    
    # 检查所有模型是否存在
    if not all(os.path.exists(p) for p in [imputer_path, scaler_path, pca_path]):
        print("错误: 缺少Imputer, Scaler或PCA模型文件。请确保已完整运行步骤3。")
    else:
        imputer = joblib.load(imputer_path)
        scaler = joblib.load(scaler_path)
        pca = joblib.load(pca_path)
        
        # 将imputer传递给分析函数
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

    print(f"\n{'='*25} 所有传统机器学习流程执行完毕 {'='*25}")


if __name__ == '__main__':
    main_traditional()