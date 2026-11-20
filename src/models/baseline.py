import numpy as np
import pandas as pd


def predict_lag7(df: pd.DataFrame) -> np.ndarray:
    """
    使用七天前的销量预测当天销量。
    """

    if "lag_7" not in df.columns:
        raise ValueError("缺少 lag_7 特征")

    return np.maximum(df["lag_7"].fillna(0).to_numpy(), 0)


def predict_rolling_mean7(df: pd.DataFrame) -> np.ndarray:
    """
    使用过去七天的平均销量预测当天销量。
    """

    if "rolling_mean_7" not in df.columns:
        raise ValueError("缺少 rolling_mean_7 特征")

    return np.maximum(df["rolling_mean_7"].fillna(0).to_numpy(), 0)
