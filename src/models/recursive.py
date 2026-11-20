import numpy as np
import pandas as pd

from src.models.predictor import DEFAULT_FEATURES, predict_lightgbm


def _build_next_day_features(history: pd.DataFrame, future_row: pd.Series) -> pd.DataFrame:
    """仅使用预测日期之前的历史销量，构造下一天的模型特征。"""

    history = history.sort_values("date")
    sales = history["sales"].to_numpy(dtype=float)

    if len(sales) < 28:
        raise ValueError("递归预测至少需要 28 天历史销量")

    date = pd.Timestamp(future_row["date"])

    row = {
        "item_id": str(future_row["item_id"]), "day_of_week": date.day_of_week,
        "day_of_month": date.day, "month": date.month, "year": date.year,
        "week_of_year": int(date.isocalendar().week), "is_weekend": int(date.day_of_week >= 5),
        "is_holiday": int(
            pd.notna(future_row["event_name_1"]) or pd.notna(future_row["event_name_2"])
        ),
        "has_price": int(pd.notna(future_row["sell_price"])),
        "sell_price": future_row["sell_price"],
        "lag_1": sales[-1], "lag_7": sales[-7], "lag_14": sales[-14], "lag_28": sales[-28],
        "rolling_mean_7": sales[-7:].mean(), "rolling_mean_28": sales[-28:].mean(),
        "rolling_std_7": sales[-7:].std(ddof=1)
    }

    return pd.DataFrame([row], columns=DEFAULT_FEATURES)


def predict_recursive(model, history_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
    """使用 LightGBM 进行递归多步预测。

    history_df:
        预测起点之前的历史数据，必须包含真实 sales。

    future_df:
        未来日期的已知信息，只需要日历、价格和商品信息。
        不允许将未来真实销量传入预测过程。

    返回：
        item_id, store_id, date, y_pred
    """

    history_required = {"item_id", "store_id", "date", "sales"}
    future_required = {
        "item_id", "store_id", "date",
        "sell_price", "event_name_1", "event_name_2"
    }

    if not history_required.issubset(history_df.columns):
        raise ValueError(f"历史数据缺少字段：{history_required - set(history_df.columns)}")

    if not future_required.issubset(future_df.columns):
        raise ValueError(f"未来数据缺少字段：{future_required - set(future_df.columns)}")

    history = history_df[["item_id", "store_id", "date", "sales"]].copy()
    future = future_df[
        ["item_id", "store_id", "date", "sell_price", "event_name_1", "event_name_2"]
    ].copy()

    history["date"] = pd.to_datetime(history["date"])
    future["date"] = pd.to_datetime(future["date"])

    if history["sales"].isna().any():
        raise ValueError("历史真实销量不能存在缺失值")

    if history.duplicated(["item_id", "store_id", "date"]).any():
        raise ValueError("历史数据存在重复日期")

    if future.duplicated(["item_id", "store_id", "date"]).any():
        raise ValueError("未来数据存在重复日期")

    # 每个商品-门店维护独立的历史序列
    histories = {}

    for key, group in history.groupby(["item_id", "store_id"], observed=True):
        histories[key] = group.sort_values("date").copy()

    results = []

    # 按日期逐日预测
    for date in sorted(future["date"].unique()):
        day_data = future[future["date"] == date]

        for _, future_row in day_data.iterrows():
            key = (future_row["item_id"], future_row["store_id"])

            if key not in histories:
                raise ValueError(f"找不到商品历史数据：{key}")

            item_history = histories[key]

            if item_history["date"].max() >= date:
                raise ValueError(f"历史数据包含预测日期或未来数据：{key}")

            X = _build_next_day_features(item_history, future_row)

            y_pred = float(predict_lightgbm(model, X)[0])
            y_pred = max(0.0, y_pred)

            results.append({
                "item_id": key[0], "store_id": key[1],
                "date": date, "y_pred": y_pred
            })

            # 关键：将预测值加入历史，而不是真实销量
            new_row = pd.DataFrame([{
                "item_id": key[0], "store_id": key[1],
                "date": date, "sales": y_pred
            }])

            histories[key] = pd.concat([item_history, new_row], ignore_index=True)

    return pd.DataFrame(results)
