
from pathlib import Path

import pandas as pd


# 项目根目录：FreshAI/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_raw_data():
    """
    加载 M5 原始数据集。

    Returns:
        sales: 商品每日销量宽表
        prices: 商品每周价格表
        calendar: 日历信息表
    """

    sales_path = RAW_DATA_DIR / "sales_train_validation.csv"
    prices_path = RAW_DATA_DIR / "sell_prices.csv"
    calendar_path = RAW_DATA_DIR / "calendar.csv"

    # 检查文件是否存在
    for path in [sales_path, prices_path, calendar_path]:
        if not path.exists():
            raise FileNotFoundError(f"找不到数据文件：{path}")

    print("正在加载 M5 数据集...")

    sales = pd.read_csv(sales_path)
    prices = pd.read_csv(prices_path)
    calendar = pd.read_csv(calendar_path)

    calendar["date"] = pd.to_datetime(calendar["date"])

    print("数据加载完成！")
    print(f"Sales: {sales.shape}")
    print(f"Prices: {prices.shape}")
    print(f"Calendar: {calendar.shape}")

    return sales, prices, calendar
