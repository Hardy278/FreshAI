
import pandas as pd


def build_sales_long(
    sales: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    dept_id: str = "FOODS_1",
    store_id: str = "CA_1"
) -> pd.DataFrame:
    """
    将 M5 销量宽表转换为长表，并合并日历和价格信息。

    默认使用 FOODS_1 部门、CA_1 门店作为试点。

    Returns:
        每行代表一个商品在某一天的销量记录。
    """

    # 1. 筛选目标商品和门店
    pilot_sales = sales[
        (sales["dept_id"] == dept_id) &
        (sales["store_id"] == store_id)
    ].copy()

    if pilot_sales.empty:
        raise ValueError(
            f"没有找到部门 {dept_id}、门店 {store_id} 对应的数据"
        )

    # 2. 找出每日销量列
    sales_columns = [
        col for col in pilot_sales.columns
        if col.startswith("d_")
    ]

    if not sales_columns:
        raise ValueError("没有找到 d_ 开头的每日销量列")

    # 3. 宽表转换成长表
    id_columns = [
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id"
    ]

    sales_long = pilot_sales.melt(
        id_vars=id_columns,
        value_vars=sales_columns,
        var_name="d",
        value_name="sales"
    )

    # 4. 合并日历信息
    calendar_small = calendar[
        calendar["d"].isin(sales_columns)
    ].copy()

    sales_long = sales_long.merge(
        calendar_small,
        on="d",
        how="left",
        validate="many_to_one"
    )

    # 5. 合并价格信息
    sales_long = sales_long.merge(
        prices,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
        validate="many_to_one"
    )

    # 6. 排序
    sales_long = sales_long.sort_values(
        ["item_id", "date"]
    ).reset_index(drop=True)

    print("数据预处理完成！")
    print(f"目标部门：{dept_id}")
    print(f"目标门店：{store_id}")
    print(f"商品数量：{sales_long['item_id'].nunique()}")
    print(f"记录数量：{len(sales_long)}")
    print(
        "价格缺失比例："
        f"{sales_long['sell_price'].isna().mean():.2%}"
    )

    return sales_long
