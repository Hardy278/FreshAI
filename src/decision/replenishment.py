from statistics import NormalDist

import numpy as np
import pandas as pd


DEFAULT_SERVICE_LEVEL = 0.95

REQUIRED_FORECAST_COLUMNS = {"item_id", "date", "y_pred"}
REQUIRED_INVENTORY_COLUMNS = {
    "item_id",
    "on_hand",
    "on_order",
    "reserved",
    "lead_time_days",
    "review_period_days",
    "daily_demand_std"
}


def validate_forecasts(forecasts: pd.DataFrame) -> None:
    """检查销量预测数据格式。"""

    missing = REQUIRED_FORECAST_COLUMNS - set(forecasts.columns)

    if missing:
        raise ValueError(f"预测数据缺少必要字段：{sorted(missing)}")

    if forecasts.empty:
        raise ValueError("预测数据不能为空")

    if forecasts.duplicated(["item_id", "date"]).any():
        raise ValueError("同一个商品同一天存在重复预测记录")

    if forecasts["y_pred"].isna().any():
        raise ValueError("预测销量存在缺失值")

    if (forecasts["y_pred"] < 0).any():
        raise ValueError("预测销量不能为负数")


def validate_inventory(inventory: pd.DataFrame) -> None:
    """检查库存参数格式。"""

    missing = REQUIRED_INVENTORY_COLUMNS - set(inventory.columns)

    if missing:
        raise ValueError(f"库存数据缺少必要字段：{sorted(missing)}")

    if inventory["item_id"].duplicated().any():
        raise ValueError("库存数据中商品 ID 不能重复")

    numeric_columns = [
        "on_hand",
        "on_order",
        "reserved",
        "lead_time_days",
        "review_period_days",
        "daily_demand_std"
    ]

    for column in numeric_columns:
        if inventory[column].isna().any():
            raise ValueError(f"{column} 存在缺失值")

        if (inventory[column] < 0).any():
            raise ValueError(f"{column} 不能为负数")


def calculate_safety_stock(
    daily_demand_std: float,
    coverage_days: int,
    service_level: float = DEFAULT_SERVICE_LEVEL
) -> float:
    """计算安全库存。

    使用正态分布近似：

        SS = z * sigma_d * sqrt(L + R)
    """

    if not 0 < service_level < 1:
        raise ValueError("service_level 必须在 0 和 1 之间")

    if daily_demand_std < 0:
        raise ValueError("daily_demand_std 不能为负数")

    if coverage_days < 0:
        raise ValueError("coverage_days 不能为负数")

    z_score = NormalDist().inv_cdf(service_level)
    safety_stock = z_score * daily_demand_std * np.sqrt(coverage_days)

    return float(safety_stock)


def calculate_item_replenishment(
    forecast: pd.DataFrame,
    inventory: pd.Series,
    service_level: float = DEFAULT_SERVICE_LEVEL
) -> dict:
    """计算单个商品的补货建议。

    forecast:
        当前商品未来的销量预测。

    inventory:
        当前商品的库存参数。
    """

    lead_time = int(inventory["lead_time_days"])
    review_period = int(inventory["review_period_days"])

    coverage_days = lead_time + review_period

    if coverage_days <= 0:
        raise ValueError("采购提前期与检查周期之和必须大于 0")

    forecast = forecast.sort_values("date")

    if len(forecast) < coverage_days:
        raise ValueError(f"{inventory['item_id']} 的预测天数不足：需要 {coverage_days} 天，实际只有 {len(forecast)} 天")

    # 只使用库存覆盖周期内的预测销量
    forecast_window = forecast.head(coverage_days)
    expected_demand = float(forecast_window["y_pred"].sum())

    # 安全库存
    safety_stock = calculate_safety_stock(
        daily_demand_std=float(inventory["daily_demand_std"]),
        coverage_days=coverage_days,
        service_level=service_level
    )

    # 目标库存
    target_stock = expected_demand + safety_stock

    # 当前库存位置
    inventory_position = (
        float(inventory["on_hand"])
        + float(inventory["on_order"])
        - float(inventory["reserved"])
    )

    # 基础补货量
    raw_order_quantity = max(0.0, target_stock - inventory_position)

    # 最小订货量
    min_order_qty = float(inventory.get("min_order_qty", 0))

    # 包装规格，默认 1
    pack_size = float(inventory.get("pack_size", 1))

    if pack_size <= 0:
        raise ValueError("pack_size 必须大于 0")

    if raw_order_quantity > 0:
        order_quantity = max(min_order_qty, np.ceil(raw_order_quantity / pack_size) * pack_size)
    else:
        order_quantity = 0.0

    return {
        "item_id": inventory["item_id"],
        "coverage_days": coverage_days,
        "expected_demand": expected_demand,
        "safety_stock": safety_stock,
        "target_stock": target_stock,
        "inventory_position": inventory_position,
        "raw_order_quantity": raw_order_quantity,
        "order_quantity": float(order_quantity)
    }


def calculate_replenishment(
    forecasts: pd.DataFrame,
    inventory: pd.DataFrame,
    service_level: float = DEFAULT_SERVICE_LEVEL
) -> pd.DataFrame:
    """对多个商品批量计算补货建议。

    forecasts:
        item_id, date, y_pred

    inventory:
        item_id, on_hand, on_order, reserved,
        lead_time_days, review_period_days,
        daily_demand_std

    返回：
        每个商品的补货决策结果。
    """

    validate_forecasts(forecasts)
    validate_inventory(inventory)

    forecasts = forecasts.copy()
    inventory = inventory.copy()

    forecasts["date"] = pd.to_datetime(forecasts["date"])
    results = []

    forecast_groups = {item_id: group for item_id, group in forecasts.groupby("item_id")}

    for _, item_inventory in inventory.iterrows():
        item_id = item_inventory["item_id"]

        if item_id not in forecast_groups:
            raise ValueError(f"找不到商品 {item_id} 的销量预测")

        item_forecast = forecast_groups[item_id]
        result = calculate_item_replenishment(
            forecast=item_forecast,
            inventory=item_inventory,
            service_level=service_level
        )

        results.append(result)

    return pd.DataFrame(results)
