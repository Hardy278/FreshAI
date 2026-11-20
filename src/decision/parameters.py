import math
from statistics import NormalDist

import numpy as np
import pandas as pd


def build_inventory_parameters(
    train_df: pd.DataFrame, demand_col: str = "sales",
    lead_time_days: int = 2, review_period_days: int = 1,
    service_level: float = 0.95
) -> pd.DataFrame:
    """根据训练期历史需求，为每个商品生成库存仿真参数。

    注意：
    1. 只能传入训练期数据。
    2. 每个商品必须有完整的逐日需求记录。
    3. 初始库存是仿真假设，不是真实库存。
    """
    required_cols = {"item_id", "date", demand_col}
    missing = required_cols - set(train_df.columns)

    if missing:
        raise ValueError(f"训练数据缺少字段：{sorted(missing)}")

    if lead_time_days < 0:
        raise ValueError("lead_time_days 不能小于 0")

    if review_period_days < 1:
        raise ValueError("review_period_days 必须大于等于 1")

    if not 0 < service_level < 1:
        raise ValueError("service_level 必须在 0 和 1 之间")

    df = train_df[["item_id", "date", demand_col]].copy()
    df["date"] = pd.to_datetime(df["date"])

    if df.duplicated(["item_id", "date"]).any():
        raise ValueError("存在重复的商品-日期记录")

    if df[demand_col].isna().any():
        raise ValueError("需求数据存在缺失值，请先处理")

    if (df[demand_col] < 0).any():
        raise ValueError("需求数据不能为负数")

    # 计算每个商品的训练期需求统计量
    stats = df.groupby("item_id")[demand_col].agg(
        baseline_daily_mean="mean",
        daily_demand_std="std",
        observed_days="count"
    ).reset_index()

    stats["daily_demand_std"] = stats["daily_demand_std"].fillna(0)

    # 正态分布分位数
    z_score = NormalDist().inv_cdf(service_level)
    coverage_days = lead_time_days + review_period_days

    stats["expected_demand"] = stats["baseline_daily_mean"] * coverage_days
    stats["safety_stock"] = z_score * stats["daily_demand_std"] * math.sqrt(coverage_days)

    # 初始库存按目标库存水平初始化
    stats["on_hand"] = np.ceil(stats["expected_demand"] + stats["safety_stock"]).astype(int)

    stats["on_order"] = 0
    stats["reserved"] = 0
    stats["lead_time_days"] = lead_time_days
    stats["review_period_days"] = review_period_days
    stats["min_order_qty"] = 1
    stats["pack_size"] = 1

    result_cols = [
        "item_id", "on_hand", "on_order", "reserved",
        "lead_time_days", "review_period_days", "daily_demand_std",
        "baseline_daily_mean", "min_order_qty", "pack_size",
        "observed_days", "expected_demand", "safety_stock"
    ]

    result = stats[result_cols].copy()

    print("库存参数生成完成")
    print(f"商品数量：{len(result)}")
    print(f"平均日需求：{result['baseline_daily_mean'].mean():.4f}")
    print(f"平均需求标准差：{result['daily_demand_std'].mean():.4f}")
    print(f"平均初始库存：{result['on_hand'].mean():.2f}")

    return result
