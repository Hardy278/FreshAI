import numpy as np
import pandas as pd

from src.models.recursive import predict_recursive
from src.decision.replenishment import calculate_safety_stock
from src.decision.simulation import adjust_order_quantity


def build_future_exog(
    calendar: pd.DataFrame, prices: pd.DataFrame,
    item_store_pairs: pd.DataFrame, start_date, end_date
) -> pd.DataFrame:
    """构造递归预测需要的未来外生特征。

    不包含真实销量，只包含：
    - 商品、门店
    - 日期
    - 售价
    - 日历事件

    参数：
        calendar: M5 日历数据
        prices: M5 商品门店价格数据
        item_store_pairs: 商品与门店组合，包含 item_id、store_id
        start_date: 预测起始日期
        end_date: 预测结束日期（包含当天）
    """
    required_calendar = {"date", "wm_yr_wk", "event_name_1", "event_name_2"}
    required_prices = {"item_id", "store_id", "wm_yr_wk", "sell_price"}
    required_pairs = {"item_id", "store_id"}

    if not required_calendar.issubset(calendar.columns):
        raise ValueError(f"calendar 缺少字段：{required_calendar - set(calendar.columns)}")

    if not required_prices.issubset(prices.columns):
        raise ValueError(f"prices 缺少字段：{required_prices - set(prices.columns)}")

    if not required_pairs.issubset(item_store_pairs.columns):
        raise ValueError(
            f"item_store_pairs 缺少字段：{required_pairs - set(item_store_pairs.columns)}"
        )

    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    if start_date > end_date:
        raise ValueError("start_date 不能晚于 end_date")

    # 只保留需要的商品门店组合
    pairs = item_store_pairs[["item_id", "store_id"]].drop_duplicates().copy()

    if pairs.empty:
        raise ValueError("商品门店组合为空")

    # 准备日历
    calendar_part = calendar[["date", "wm_yr_wk", "event_name_1", "event_name_2"]].copy()
    calendar_part["date"] = pd.to_datetime(calendar_part["date"])
    calendar_part = calendar_part[calendar_part["date"].between(start_date, end_date)].copy()

    if calendar_part.empty:
        raise ValueError("指定日期范围内没有日历数据")

    if calendar_part["date"].duplicated().any():
        raise ValueError("calendar 中存在重复日期")

    # 检查价格表的联合主键
    price_part = prices[["item_id", "store_id", "wm_yr_wk", "sell_price"]].copy()
    price_keys = ["item_id", "store_id", "wm_yr_wk"]

    if price_part.duplicated(price_keys).any():
        raise ValueError("prices 中存在重复的商品、门店、周组合")

    # 商品门店组合 × 预测日期
    future = pairs.merge(calendar_part, how="cross")

    # 根据商品、门店和周匹配价格
    future = future.merge(price_part, on=price_keys, how="left", validate="many_to_one")

    future = future[
        ["item_id", "store_id", "date", "sell_price", "event_name_1", "event_name_2"]
    ]

    future = future.sort_values(["item_id", "store_id", "date"]).reset_index(drop=True)

    # 检查输出
    if future.duplicated(["item_id", "store_id", "date"]).any():
        raise RuntimeError("未来外生特征存在重复商品日期记录")

    if future["date"].min() != start_date:
        raise ValueError("未来特征缺少起始日期")

    if future["date"].max() != end_date:
        raise ValueError("未来特征缺少结束日期")

    expected_rows = len(pairs) * calendar_part["date"].nunique()

    if len(future) != expected_rows:
        raise RuntimeError(f"未来特征记录数异常：预期 {expected_rows}，实际 {len(future)}")

    print("未来外生特征构造完成")
    print(f"商品门店组合：{len(pairs)}")
    print(f"日期范围：{start_date.date()} 至 {end_date.date()}")
    print(f"记录数量：{len(future)}")
    print(f"价格缺失比例：{future['sell_price'].isna().mean():.2%}")

    return future


def _validate_rolling_inputs(
    history_df, actual_df, future_exog_df, inventory, simulation_days
):
    """检查滚动回测输入数据。"""
    history_required = {"item_id", "store_id", "date", "sales"}
    actual_required = {"item_id", "store_id", "date", "sales"}
    exog_required = {
        "item_id", "store_id", "date",
        "sell_price", "event_name_1", "event_name_2"
    }
    inventory_required = {
        "item_id", "on_hand", "on_order", "reserved",
        "lead_time_days", "review_period_days",
        "daily_demand_std", "baseline_daily_mean",
        "min_order_qty", "pack_size"
    }

    datasets = [
        ("history_df", history_df, history_required),
        ("actual_df", actual_df, actual_required),
        ("future_exog_df", future_exog_df, exog_required),
        ("inventory", inventory, inventory_required)
    ]

    for name, df, required in datasets:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{name} 缺少字段：{missing}")
        if df.empty:
            raise ValueError(f"{name} 不能为空")

    if simulation_days <= 0:
        raise ValueError("simulation_days 必须大于 0")

    # 复制数据，避免修改调用方的数据
    history_df = history_df.copy()
    actual_df = actual_df.copy()
    future_exog_df = future_exog_df.copy()
    inventory = inventory.copy()

    for df in [history_df, actual_df, future_exog_df]:
        df["date"] = pd.to_datetime(df["date"])

    # 检查重复记录
    key_columns = ["item_id", "store_id", "date"]

    for name, df in [
        ("history_df", history_df), ("actual_df", actual_df),
        ("future_exog_df", future_exog_df)
    ]:
        if df.duplicated(key_columns).any():
            raise ValueError(f"{name} 存在重复的商品、门店、日期记录")

    if inventory["item_id"].duplicated().any():
        raise ValueError("inventory 存在重复商品")

    # 当前试点是单门店，因此以 item_id 作为库存键
    history_items = set(history_df["item_id"])
    actual_items = set(actual_df["item_id"])
    exog_items = set(future_exog_df["item_id"])
    inventory_items = set(inventory["item_id"])

    if not (history_items == actual_items == exog_items == inventory_items):
        raise ValueError("历史、测试、未来特征和库存中的商品集合不一致")

    # 检查测试日期
    test_dates = sorted(actual_df["date"].unique())

    if len(test_dates) != simulation_days:
        raise ValueError(f"测试日期数量应为 {simulation_days}，实际为 {len(test_dates)}")

    expected_dates = pd.date_range(start=test_dates[0], periods=simulation_days, freq="D")

    if not np.array_equal(pd.DatetimeIndex(test_dates), expected_dates):
        raise ValueError("测试日期不连续")

    # 每个测试日期必须包含所有商品
    expected_item_count = len(inventory)
    daily_counts = actual_df.groupby("date").size()

    if not (daily_counts == expected_item_count).all():
        raise ValueError("actual_df 存在商品日期记录缺失")

    # 库存参数检查
    numeric_columns = [
        "on_hand", "on_order", "reserved",
        "lead_time_days", "review_period_days",
        "daily_demand_std", "baseline_daily_mean",
        "min_order_qty", "pack_size"
    ]

    if inventory[numeric_columns].isna().any().any():
        raise ValueError("inventory 中存在数值缺失")

    if (inventory[["on_hand", "on_order", "reserved"]] < 0).any().any():
        raise ValueError("库存数量不能为负")

    if (inventory["daily_demand_std"] < 0).any():
        raise ValueError("daily_demand_std 不能为负")

    if (inventory["baseline_daily_mean"] < 0).any():
        raise ValueError("baseline_daily_mean 不能为负")

    if (inventory["lead_time_days"] < 1).any():
        raise ValueError("当前滚动仿真要求 lead_time_days 至少为 1")

    if (inventory["review_period_days"] < 1).any():
        raise ValueError("review_period_days 必须至少为 1")

    if (inventory["min_order_qty"] < 0).any():
        raise ValueError("min_order_qty 不能为负")

    if (inventory["pack_size"] < 1).any():
        raise ValueError("pack_size 必须至少为 1")

    if (inventory["on_order"] != 0).any():
        raise ValueError("当前版本不支持初始在途订单，请先将 on_order 设置为 0")

    if (inventory["reserved"] > inventory["on_hand"]).any():
        raise ValueError("初始 reserved 不能大于 on_hand")

    # 检查历史和实际销量
    if history_df["sales"].isna().any():
        raise ValueError("history_df 的 sales 存在缺失")

    if actual_df["sales"].isna().any():
        raise ValueError("actual_df 的 sales 存在缺失")

    if (history_df["sales"] < 0).any() or (actual_df["sales"] < 0).any():
        raise ValueError("销量不能为负")

    if history_df["date"].max() >= actual_df["date"].min():
        raise ValueError("历史销量必须严格早于测试期")

    # 检查未来外生特征覆盖范围
    max_coverage_days = int(
        (inventory["lead_time_days"] + inventory["review_period_days"]).max()
    )
    required_end_date = pd.Timestamp(test_dates[-1]) + pd.Timedelta(days=max_coverage_days - 1)

    if future_exog_df["date"].min() > test_dates[0]:
        raise ValueError("future_exog_df 缺少测试期起始日期")

    if future_exog_df["date"].max() < required_end_date:
        raise ValueError(f"未来特征至少需要覆盖到 {required_end_date.date()}")

    return (
        history_df, actual_df, future_exog_df, inventory,
        pd.DatetimeIndex(test_dates), max_coverage_days
    )


def run_rolling_backtest(
    model, history_df, actual_df, future_exog_df,
    inventory, simulation_days, service_level=0.95
):
    """执行逐日滚动预测与库存补货仿真。

    策略：
        ai_forecast: 使用 LightGBM 递归预测需求
        fixed_baseline: 使用训练期平均日需求估计需求

    每天使用截至前一天的真实销量进行预测。
    当天真实销量只用于销售结算，不进入当天预测。
    """
    if not 0 < service_level < 1:
        raise ValueError("service_level 必须在 0 和 1 之间")

    (
        history_df, actual_df, future_exog_df, inventory,
        test_dates, max_coverage_days
    ) = _validate_rolling_inputs(
        history_df, actual_df, future_exog_df, inventory, simulation_days
    )

    print("\n===== 开始滚动回测 =====")
    print(f"商品数量：{len(inventory)}")
    print(f"测试天数：{simulation_days}")
    print(f"预测最大覆盖天数：{max_coverage_days}")
    print(f"测试日期：{test_dates[0].date()} 至 {test_dates[-1].date()}")

    # 两种策略各自维护库存和在途订单
    strategies = ["ai_forecast", "fixed_baseline"]

    states = {}

    for strategy in strategies:
        states[strategy] = {}

        for row in inventory.itertuples(index=False):
            item_id = row.item_id

            states[strategy][item_id] = {
                "on_hand": float(row.on_hand), "reserved": float(row.reserved),
                "lead_time_days": int(row.lead_time_days),
                "review_period_days": int(row.review_period_days),
                "daily_demand_std": float(row.daily_demand_std),
                "baseline_daily_mean": float(row.baseline_daily_mean),
                "min_order_qty": int(row.min_order_qty), "pack_size": int(row.pack_size),
                # key: 到货日的仿真索引
                # value: 该日到货数量
                "pipeline": {}
            }

    inventory_lookup = inventory.set_index("item_id")

    # 建立真实销量查找表
    actual_lookup = actual_df.set_index(["date", "item_id"])["sales"]

    # 逐日结果
    daily_results = []

    # 保存每天生成的完整预测结果，便于分析预测误差
    forecast_details = []

    # 历史数据只追加真实销量
    rolling_history = history_df[["item_id", "store_id", "date", "sales"]].copy()

    for day_index, current_date in enumerate(test_dates):
        print(f"\n[{day_index + 1}/{simulation_days}] 正在模拟：{current_date.date()}")

        # 1. 使用截至昨天的真实销量，递归预测未来需求
        future_end_date = current_date + pd.Timedelta(days=max_coverage_days - 1)

        current_exog = future_exog_df[
            future_exog_df["date"].between(current_date, future_end_date)
        ].copy()

        # 只把历史销量和未来外生变量交给预测模块
        # actual_df 不会传入 predict_recursive
        forecasts = predict_recursive(
            model=model, history_df=rolling_history, future_df=current_exog
        )

        forecasts["date"] = pd.to_datetime(forecasts["date"])

        if forecasts.duplicated(["item_id", "store_id", "date"]).any():
            raise RuntimeError("递归预测结果存在重复记录")

        if forecasts["y_pred"].isna().any():
            raise RuntimeError("递归预测结果存在缺失值")

        if (forecasts["y_pred"] < 0).any():
            raise RuntimeError("递归预测结果存在负值")

        forecast_details.append(forecasts.copy())

        forecast_lookup = {
            item_id: group.sort_values("date").reset_index(drop=True)
            for item_id, group in forecasts.groupby("item_id")
        }

        # 2. 两种策略分别处理库存
        for strategy in strategies:
            strategy_states = states[strategy]

            for item_id, state in strategy_states.items():
                item_inventory = inventory_lookup.loc[item_id]

                lead_time = state["lead_time_days"]
                review_period = state["review_period_days"]
                coverage_days = lead_time + review_period

                # 2.1 处理当天到货
                arrivals = state["pipeline"].pop(day_index, 0.0)
                state["on_hand"] += arrivals

                # 2.2 获取该商品未来预测
                item_forecasts = forecast_lookup[item_id]
                item_forecasts = item_forecasts[
                    item_forecasts["date"].between(
                        current_date, current_date + pd.Timedelta(days=coverage_days - 1)
                    )
                ]

                if len(item_forecasts) != coverage_days:
                    raise RuntimeError(
                        f"{item_id} 的预测覆盖天数不足：需要 {coverage_days}，"
                        f"实际 {len(item_forecasts)}"
                    )

                ai_today_prediction = float(item_forecasts.iloc[0]["y_pred"])

                # 2.3 计算需求预测
                if strategy == "ai_forecast":
                    expected_demand = float(item_forecasts["y_pred"].sum())
                    today_prediction = ai_today_prediction
                else:
                    expected_demand = state["baseline_daily_mean"] * coverage_days
                    today_prediction = state["baseline_daily_mean"]

                # 2.4 计算安全库存
                safety_stock = calculate_safety_stock(
                    daily_demand_std=state["daily_demand_std"],
                    coverage_days=coverage_days,
                    service_level=service_level
                )
                target_stock = expected_demand + safety_stock

                # 2.5 计算库存位置
                on_order = sum(state["pipeline"].values())
                inventory_position = state["on_hand"] + on_order - state["reserved"]

                # 2.6 到达复查日时决定是否订货
                ordered_quantity = 0.0

                if day_index % review_period == 0:
                    raw_quantity = max(0.0, target_stock - inventory_position)

                    ordered_quantity = adjust_order_quantity(
                        raw_quantity,
                        min_order_qty=state["min_order_qty"],
                        pack_size=state["pack_size"]
                    )
                    ordered_quantity = float(ordered_quantity)

                    if ordered_quantity > 0:
                        arrival_day = day_index + lead_time
                        state["pipeline"][arrival_day] = (
                            state["pipeline"].get(arrival_day, 0.0) + ordered_quantity
                        )

                # 2.7 获取当天真实需求
                actual_demand = float(actual_lookup.loc[(current_date, item_id)])

                # 2.8 使用可用库存满足需求
                available_stock = max(0.0, state["on_hand"] - state["reserved"])
                sold_quantity = min(available_stock, actual_demand)

                lost_sales = actual_demand - sold_quantity
                state["on_hand"] -= sold_quantity

                if state["on_hand"] < -1e-8:
                    raise RuntimeError(f"{item_id} 出现负库存：{state['on_hand']}")

                state["on_hand"] = max(0.0, state["on_hand"])

                # 2.9 更新在途订单
                ending_on_order = sum(state["pipeline"].values())

                # 2.10 记录日度结果
                daily_results.append({
                    "date": current_date, "item_id": item_id, "strategy": strategy,
                    "y_pred": today_prediction, "ai_y_pred": ai_today_prediction,
                    "y_true": actual_demand,
                    "expected_demand": expected_demand,
                    "safety_stock": safety_stock, "target_stock": target_stock,
                    "arrivals": arrivals, "ordered_quantity": ordered_quantity,
                    "sold_quantity": sold_quantity, "lost_sales": lost_sales,
                    "on_hand": state["on_hand"], "on_order": ending_on_order,
                    "inventory_position": state["on_hand"] + ending_on_order - state["reserved"]
                })

        # 3. 当天结算完成后，将真实销量加入历史
        actual_today = actual_df[
            actual_df["date"] == current_date
        ][["item_id", "store_id", "date", "sales"]].copy()

        rolling_history = pd.concat([rolling_history, actual_today], ignore_index=True)

    # 4. 整理结果
    daily_results = pd.DataFrame(daily_results)
    forecast_details = pd.concat(forecast_details, ignore_index=True)

    # 5. 按策略汇总
    summary_rows = []

    for strategy in strategies:
        result = daily_results[daily_results["strategy"] == strategy].copy()

        total_demand = result["y_true"].sum()
        total_sold = result["sold_quantity"].sum()
        total_lost = result["lost_sales"].sum()
        average_inventory = result["on_hand"].mean()

        summary_rows.append({
            "strategy": strategy,
            "total_demand": total_demand, "total_sold": total_sold,
            "total_lost_sales": total_lost,
            "fill_rate": total_sold / total_demand if total_demand > 0 else 1.0,
            "stockout_item_days": int((result["lost_sales"] > 0).sum()),
            "stockout_dates": int(result.groupby("date")["lost_sales"].sum().gt(0).sum()),
            "total_ordered": result["ordered_quantity"].sum(),
            "ending_inventory": result.groupby("item_id")["on_hand"].last().sum(),
            "ending_on_order": result.groupby("item_id")["on_order"].last().sum(),
            "average_inventory": average_inventory,
            "inventory_turnover": (
                total_sold / average_inventory if average_inventory > 0 else np.nan
            )
        })

    summary = pd.DataFrame(summary_rows).set_index("strategy")

    print("\n===== 滚动回测完成 =====")
    print("\n策略比较结果：")
    print(summary)

    return {
        "daily_results": daily_results,
        "forecast_details": forecast_details,
        "summary": summary
    }