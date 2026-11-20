import numpy as np
import pandas as pd

from sklearn.metrics import (mean_absolute_error, mean_squared_error)


def evaluate_model(y_true,y_pred) -> dict:
    """
    计算回归模型的评价指标。

    指标：
    MAE、RMSE、WAPE
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) != len(y_pred):
        raise ValueError("真实值和预测值长度不一致")

    if len(y_true) == 0:
        raise ValueError("评价数据不能为空")

    # 销量不能为负数
    y_pred = np.maximum(y_pred, 0)

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    denominator = np.sum(np.abs(y_true))

    wape = (
        np.sum(np.abs(y_true - y_pred)) / denominator
        if denominator > 0
        else np.nan
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE": wape
    }


def compare_models(results: dict) -> pd.DataFrame:
    """
    将多个模型的评价结果整理成表格。

    results 示例：
    {
        "Lag-7": {"MAE": 1.2, "RMSE": 2.1, "WAPE": 0.5},
        "LightGBM": {"MAE": 1.0, "RMSE": 1.8, "WAPE": 0.4}
    }
    """

    df = pd.DataFrame.from_dict(results, orient="index")
    df.index.name = "Model"

    return df.sort_values("WAPE")


def split_by_date(df: pd.DataFrame, validation_days: int = 28):
    """
    按时间顺序划分训练集和验证集。

    最后 validation_days 天作为验证集。
    """

    if validation_days <= 0:
        raise ValueError("validation_days 必须大于 0")

    max_date = df["date"].max()

    validation_start = (max_date - pd.Timedelta(days=validation_days - 1))

    train_df = df[df["date"] < validation_start].copy()
    valid_df = df[df["date"] >= validation_start].copy()

    if train_df.empty or valid_df.empty:
        raise ValueError("训练集或验证集为空")

    print("时间切分完成")
    print(f"训练集：{train_df['date'].min().date()} 至 {train_df['date'].max().date()}")
    print(f"验证集：{valid_df['date'].min().date()} 至 {valid_df['date'].max().date()}")

    return train_df, valid_df
