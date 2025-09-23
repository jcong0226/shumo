# 文件名: utils_handle_outliers.py
import pandas as pd
import numpy as np

def handle_outliers_iqr(df: pd.DataFrame, train_df: pd.DataFrame = None):
    """
    使用IQR方法处理特征数据集中的异常值。
    通过“封顶”方式将超出边界的异常值替换为边界值。

    Args:
        df (pd.DataFrame): 需要处理异常值的数据集（可以是源域训练集、测试集或目标域数据集）。
        train_df (pd.DataFrame, optional): 用于计算IQR边界的训练数据集。
            - 如果为 None，则在 df 自身上计算边界（即处理源域训练集时）。
            - 如果提供，则使用 train_df 计算的边界来处理 df（即处理测试集或目标域数据时）。

    Returns:
        pd.DataFrame: 处理完异常值的数据集。
        dict: 每个特征的异常值边界（'lower_bound', 'upper_bound'）。
    """
    print("开始处理特征异常值...")
    
    # 仅对数值类型的特征列进行处理
    feature_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # 复制DataFrame以避免修改原始数据
    df_capped = df.copy()
    
    bounds = {}

    is_training_set = (train_df is None)
    
    for col in feature_cols:
        if is_training_set:
            # 如果是训练集，就在自身上计算边界
            Q1 = df_capped[col].quantile(0.25)
            Q3 = df_capped[col].quantile(0.75)
        else:
            # 如果是测试集或目标域，使用传入的训练集计算边界
            Q1 = train_df[col].quantile(0.25)
            Q3 = train_df[col].quantile(0.75)

        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        bounds[col] = {'lower_bound': lower_bound, 'upper_bound': upper_bound}
        
        # 统计异常值数量
        outliers_low = (df_capped[col] < lower_bound).sum()
        outliers_high = (df_capped[col] > upper_bound).sum()
        
        if outliers_low > 0 or outliers_high > 0:
            print(f"特征 '{col}': 发现 {outliers_low} 个下限异常值, {outliers_high} 个上限异常值。正在进行封顶处理...")
            # 执行封顶
            df_capped[col] = np.where(df_capped[col] < lower_bound, lower_bound, df_capped[col])
            df_capped[col] = np.where(df_capped[col] > upper_bound, upper_bound, df_capped[col])
    
    print("异常值处理完成。")
    return df_capped, bounds