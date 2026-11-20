from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_features(sales_long: pd.DataFrame) -> pd.DataFrame:
    """
    根据每日销量长表构造预测特征。

    包含：
    1. 日期特征
    2. 节假日特征
    3. 价格可用性特征
    4. 历史滞后特征
    5. 历史滚动统计特征

    注意：
    所有滞后和滚动特征只使用当前日期之前的数据，
    避免未来信息泄漏。
    """

    required_columns = [
        "item_id",
        "store_id",
        "date",
        "sales",
        "sell_price",
        "event_name_1",
        "event_name_2"
    ]

    missing_columns = [col for col in required_columns if col not in sales_long.columns]

    if missing_columns:
        raise ValueError(f"输入数据缺少必要字段：{missing_columns}")

    df = sales_long.copy()

    # 1. 日期特征
    df["date"]         = pd.to_datetime(df["date"])
    df["day_of_week"]  = df["date"].dt.day_of_week
    df["day_of_month"] = df["date"].dt.day
    df["month"]        = df["date"].dt.month
    df["year"]         = df["date"].dt.year
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)

    # 2. 节假日特征
    df["is_holiday"] = (df["event_name_1"].notna() | df["event_name_2"].notna()).astype(int)

    # 3. 价格可用性特征
    df["has_price"] = (df["sell_price"].notna()).astype(int)

    # 4. 按商品和门店排序
    df = df.sort_values(["item_id", "store_id", "date"]).reset_index(drop=True)
    group_keys = ["item_id", "store_id"]

    # 检查同一商品、门店、日期是否重复
    duplicated = df.duplicated(subset=group_keys + ["date"])

    if duplicated.any():
        raise ValueError("发现重复的商品-门店-日期记录，请先检查预处理结果。")

    # 5. 滞后特征
    group = df.groupby(group_keys, observed=True)["sales"]

    df["lag_1"] = group.shift(1)
    df["lag_7"] = group.shift(7)
    df["lag_14"] = group.shift(14)
    df["lag_28"] = group.shift(28)

    # 6. 滚动统计特征
    group = df.groupby(group_keys, observed=True)["sales"]
    shifted_sales = group.shift(1)

    df["rolling_mean_7"] = (
        shifted_sales.groupby(
            [df["item_id"], df["store_id"]],
            observed=True
        ).rolling(7).mean()
        .reset_index(level=[0, 1], drop=True)
    )

    df["rolling_mean_28"] = (
        shifted_sales.groupby(
            [df["item_id"], df["store_id"]],
            observed=True
        ).rolling(28).mean()
        .reset_index(level=[0, 1], drop=True)
    )

    df["rolling_std_7"] = (
        shifted_sales.groupby(
            [df["item_id"], df["store_id"]],
            observed=True
        ).rolling(7).std()
        .reset_index(level=[0, 1], drop=True)
    )

    print("特征工程完成！")
    print(f"数据规模：{df.shape}")
    print(f"特征数量：{df.shape[1]}")

    return df


def save_features(df: pd.DataFrame, filename: str = "foods1_ca1_features.parquet") -> Path:
    """ 将特征数据保存为 Parquet 文件。 """

    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename

    df.to_parquet(output_path, index=False)

    print(f"特征数据已保存：{output_path}")

    return output_path
