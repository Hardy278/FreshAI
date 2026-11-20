import numpy as np
import pandas as pd

from .replenishment import calculate_safety_stock


def validate_simulation_data(
    data: pd.DataFrame,
    inventory: pd.DataFrame,
    simulation_days: int,
    strategy: str
) -> None:
    required_data = {"item_id", "date", "y_pred", "y_true"}
    required_inventory = {
        "item_id", "on_hand", "lead_time_days",
        "review_period_days", "daily_demand_std"
    }

    if strategy == "fixed_baseline":
        required_inventory.add("baseline_daily_mean")

    missing_data = required_data - set(data.columns)
    missing_inventory = required_inventory - set(inventory.columns)

    if missing_data:
        raise ValueError(f"仿真数据缺少字段：{sorted(missing_data)}")

    if missing_inventory:
        raise ValueError(f"库存数据缺少字段：{sorted(missing_inventory)}")

    if simulation_days <= 0:
        raise ValueError("simulation_days 必须大于 0")

    if data.empty or inventory.empty:
        raise ValueError("输入数据不能为空")

    if data.duplicated(["item_id", "date"]).any():
        raise ValueError("同一商品同一天不能存在重复记录")

    if inventory["item_id"].duplicated().any():
        raise ValueError("库存数据中的商品 ID 不能重复")

    if not set(inventory["item_id"]).issubset(set(data["item_id"])):
        raise ValueError("部分库存商品没有对应的仿真数据")

    if (inventory["lead_time_days"] < 1).any():
        raise ValueError("采购提前期必须至少为 1 天")

    if (inventory["review_period_days"] < 1).any():
        raise ValueError("库存检查周期必须至少为 1 天")

    numeric_columns = ["on_hand", "lead_time_days", "review_period_days", "daily_demand_std"]

    if strategy == "fixed_baseline":
        numeric_columns.append("baseline_daily_mean")

    for column in numeric_columns:
        if inventory[column].isna().any():
            raise ValueError(f"{column} 存在缺失值")

        if not np.isfinite(inventory[column]).all():
            raise ValueError(f"{column} 存在无效数值")

        if (inventory[column] < 0).any():
            raise ValueError(f"{column} 不能为负数")

    if strategy not in {"ai_forecast", "fixed_baseline"}:
        raise ValueError(f"不支持的策略：{strategy}")

    for item_id, group in data.groupby("item_id"):
        group = group.sort_values("date")

        if len(group) < simulation_days:
            raise ValueError(f"{item_id} 的实际需求数据不足")

        actual = group["y_true"].iloc[:simulation_days]

        if actual.isna().any():
            raise ValueError(f"{item_id} 的仿真周期内实际销量存在缺失")

        predictions = group["y_pred"]

        if predictions.isna().any():
            raise ValueError(f"{item_id} 的预测销量存在缺失")

        if not np.isfinite(predictions).all():
            raise ValueError(f"{item_id} 的预测销量存在无效数值")

        if (predictions < 0).any():
            raise ValueError(f"{item_id} 的预测销量不能为负数")

        if (actual < 0).any():
            raise ValueError(f"{item_id} 的实际销量不能为负数")


def adjust_order_quantity(
    raw_quantity: float, min_order_qty: float = 0, pack_size: float = 1
) -> float:
    if pack_size <= 0:
        raise ValueError("pack_size 必须大于 0")

    if min_order_qty < 0:
        raise ValueError("min_order_qty 不能为负数")

    if raw_quantity <= 0:
        return 0.0

    quantity = max(min_order_qty, np.ceil(raw_quantity / pack_size) * pack_size)

    return float(quantity)


def simulate_item_inventory(
    data: pd.DataFrame,
    inventory: pd.Series,
    simulation_days: int,
    strategy: str = "ai_forecast",
    service_level: float = 0.95
) -> tuple[pd.DataFrame, dict]:
    item_id = inventory["item_id"]
    data = data.sort_values("date").reset_index(drop=True)

    lead_time = int(inventory["lead_time_days"])
    review_period = int(inventory["review_period_days"])
    coverage_days = lead_time + review_period

    if len(data) < simulation_days + coverage_days - 1:
        raise ValueError(
            f"{item_id} 的预测窗口不足：仿真需要 {simulation_days} 天，"
            f"覆盖周期需要额外 {coverage_days - 1} 天"
        )

    on_hand = float(inventory["on_hand"])
    daily_demand_std = float(inventory["daily_demand_std"])
    min_order_qty = float(inventory.get("min_order_qty", 0))
    pack_size = float(inventory.get("pack_size", 1))
    baseline_daily_mean = float(inventory.get("baseline_daily_mean", 0))

    # 记录未来到货订单
    pipeline = {}
    daily_records = []

    # 只模拟 simulation_days 天
    for day_index in range(simulation_days):
        row = data.iloc[day_index]
        date = row["date"]

        # 1. 处理当天到货
        arrivals = pipeline.pop(day_index, 0.0)
        on_hand += arrivals

        # 2. 检查是否需要补货
        ordered_quantity = 0.0

        expected_demand = np.nan
        safety_stock = np.nan
        target_stock = np.nan

        # 到达检查日才进行订货
        if day_index % review_period == 0:
            if strategy == "ai_forecast":
                forecast_window = data.iloc[day_index:day_index + coverage_days]
                expected_demand = float(forecast_window["y_pred"].sum())
            elif strategy == "fixed_baseline":
                expected_demand = baseline_daily_mean * coverage_days

            safety_stock = calculate_safety_stock(
                daily_demand_std=daily_demand_std,
                coverage_days=coverage_days, service_level=service_level
            )
            target_stock = expected_demand + safety_stock
            inventory_position = on_hand + sum(pipeline.values())
            raw_order_quantity = max(0.0, target_stock - inventory_position)

            ordered_quantity = adjust_order_quantity(
                raw_quantity=raw_order_quantity,
                min_order_qty=min_order_qty, pack_size=pack_size
            )

            # 订单在 lead_time 天后到货
            if ordered_quantity > 0:
                arrival_day = day_index + lead_time
                pipeline[arrival_day] = pipeline.get(arrival_day, 0.0) + ordered_quantity

        # 3. 发生实际销售
        actual_demand = float(row["y_true"])

        sold_quantity = min(on_hand, actual_demand)
        lost_sales = actual_demand - sold_quantity
        on_hand -= sold_quantity

        # 4. 记录库存状态
        on_order = sum(pipeline.values())

        daily_records.append({
            "item_id": item_id, "date": date, "strategy": strategy,
            "y_pred": float(row["y_pred"]), "y_true": actual_demand,
            "arrivals": arrivals, "ordered_quantity": ordered_quantity,
            "expected_demand": expected_demand, "safety_stock": safety_stock,
            "target_stock": target_stock, "sold_quantity": sold_quantity,
            "lost_sales": lost_sales, "on_hand": on_hand,
            "on_order": on_order
        })

    # 汇总
    daily_df = pd.DataFrame(daily_records)

    total_demand = float(daily_df["y_true"].sum())
    total_sold = float(daily_df["sold_quantity"].sum())
    total_lost = float(daily_df["lost_sales"].sum())
    total_ordered = float(daily_df["ordered_quantity"].sum())
    average_inventory = float(daily_df["on_hand"].mean())
    ending_on_order = float(sum(pipeline.values()))

    fill_rate = total_sold / total_demand if total_demand > 0 else 1.0

    summary = {
        "item_id": item_id, "strategy": strategy,
        "total_demand": total_demand, "total_sold": total_sold,
        "total_lost_sales": total_lost, "fill_rate": fill_rate,
        "stockout_days": int((daily_df["lost_sales"] > 0).sum()),
        "total_ordered": total_ordered,
        "ending_inventory": float(on_hand), "ending_on_order": ending_on_order,
        "average_inventory": average_inventory,
        "inventory_turnover": total_sold / average_inventory if average_inventory > 0 else np.nan
    }

    return daily_df, summary


def simulate_inventory(
    data: pd.DataFrame,
    inventory: pd.DataFrame,
    simulation_days: int,
    strategy: str = "ai_forecast",
    service_level: float = 0.95
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_simulation_data(
        data=data, inventory=inventory,
        simulation_days=simulation_days, strategy=strategy
    )

    data = data.copy()
    inventory = inventory.copy()

    data["date"] = pd.to_datetime(data["date"])

    all_daily_results = []
    all_summaries = []

    for _, item_inventory in inventory.iterrows():
        item_id = item_inventory["item_id"]
        item_data = data[data["item_id"] == item_id].copy()

        daily_result, summary = simulate_item_inventory(
            data=item_data, inventory=item_inventory,
            simulation_days=simulation_days, strategy=strategy,
            service_level=service_level
        )

        all_daily_results.append(daily_result)
        all_summaries.append(summary)

    daily_results = pd.concat(all_daily_results, ignore_index=True)
    summary_results = pd.DataFrame(all_summaries)

    return daily_results, summary_results


def compare_inventory_strategies(
    data: pd.DataFrame,
    inventory: pd.DataFrame,
    simulation_days: int,
    service_level: float = 0.95
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_daily = []
    all_summaries = []

    strategies = ["ai_forecast", "fixed_baseline"]

    for strategy in strategies:
        daily, summary = simulate_inventory(
            data=data, inventory=inventory,
            simulation_days=simulation_days, strategy=strategy,
            service_level=service_level
        )

        all_daily.append(daily)
        all_summaries.append(summary)

    daily_results = pd.concat(all_daily, ignore_index=True)
    summary_results = pd.concat(all_summaries, ignore_index=True)

    return daily_results, summary_results
