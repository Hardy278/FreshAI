import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


def evaluate_model(y_true, y_pred) -> dict:
    """计算 MAE、RMSE 和 WAPE。"""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) != len(y_pred):
        raise ValueError("真实值和预测值长度不一致")

    if len(y_true) == 0:
        raise ValueError("评价数据不能为空")

    if not np.isfinite(y_true).all():
        raise ValueError("真实值中存在 NaN 或无穷值")

    if not np.isfinite(y_pred).all():
        raise ValueError("预测值中存在 NaN 或无穷值")

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
    """整理并比较多个模型的评价结果。"""

    df = pd.DataFrame.from_dict(results, orient="index")
    df.index.name = "Model"

    return df.sort_values("WAPE")


def split_by_date(df: pd.DataFrame, validation_days: int = 28):
    """按时间顺序划分训练集和验证集。"""

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


def split_train_valid_test(
    df: pd.DataFrame,
    validation_days: int = 28,
    test_days: int = 28
):
    """按时间顺序划分训练集、验证集和测试集。"""

    if validation_days <= 0 or test_days <= 0:
        raise ValueError("验证集和测试集天数必须大于 0")

    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])

    max_date = data["date"].max()

    test_start = (max_date - pd.Timedelta(days=test_days - 1))
    valid_start = (test_start - pd.Timedelta(days=validation_days))

    train_df = data[data["date"] < valid_start].copy()
    valid_df = data[(data["date"] >= valid_start) & (data["date"] < test_start)].copy()

    test_df = data[data["date"] >= test_start].copy()

    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError("训练集、验证集或测试集为空")

    print("三段式时间切分完成")

    for name, part in [("Train", train_df), ("Valid", valid_df), ("Test", test_df)]:
        print(f"{name}: {part['date'].min().date()} 至 {part['date'].max().date()} | {len(part)} 条记录")

    return train_df, valid_df, test_df

def analyze_item_errors(
    y_true,
    y_pred,
    item_ids
) -> pd.DataFrame:
    """
    按商品统计预测误差。

    指标：
    - Actual Sales：实际累计销量
    - Predicted Sales：预测累计销量
    - MAE：日均绝对误差
    - RMSE：均方根误差
    - WAPE：商品维度的加权绝对百分比误差
    - Bias：预测累计销量 - 实际累计销量
    - Bias Rate：偏差占实际销量的比例
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    item_ids = np.asarray(item_ids)

    if not len(y_true) == len(y_pred) == len(item_ids):
        raise ValueError("真实值、预测值和商品 ID 长度不一致")

    if len(y_true) == 0:
        raise ValueError("分析数据不能为空")

    if not np.isfinite(y_true).all():
        raise ValueError("真实销量中存在 NaN 或无穷值")

    if not np.isfinite(y_pred).all():
        raise ValueError("预测销量中存在 NaN 或无穷值")

    y_pred = np.maximum(y_pred, 0)

    error_df = pd.DataFrame({
        "item_id": item_ids,
        "sales": y_true,
        "prediction": y_pred
    })

    error_df["abs_error"] = np.abs(error_df["sales"] - error_df["prediction"])
    error_df["squared_error"] = (error_df["sales"] - error_df["prediction"]) ** 2

    result = error_df.groupby("item_id").agg(
        actual_sales=("sales", "sum"),
        predicted_sales=("prediction", "sum"),
        MAE=("abs_error", "mean"),
        RMSE=("squared_error", lambda x: np.sqrt(x.mean())),
        total_absolute_error=("abs_error", "sum")
    )

    result["WAPE"] = result["total_absolute_error"] / result["actual_sales"].abs().replace(0, np.nan)
    result["Bias"] = result["predicted_sales"] - result["actual_sales"]
    result["Bias_Rate"] = result["Bias"] / result["actual_sales"].abs().replace(0, np.nan)

    result = result.drop(columns=["total_absolute_error"])

    return result.sort_values("WAPE", ascending=False)


def analyze_demand_patterns(sales_long):
    """ 分析每个商品的历史需求模式。 """
    import numpy as np
    import pandas as pd

    results = []

    for item_id, group in sales_long.groupby("item_id"):
        sales = group["sales"].to_numpy(dtype=float)

        nonzero_sales = sales[sales > 0]
        nonzero_count = len(nonzero_sales)

        mean_sales = sales.mean()
        zero_ratio = np.mean(sales == 0)

        if nonzero_count > 0:
            adi = len(sales) / nonzero_count
            nonzero_mean = nonzero_sales.mean()

            if nonzero_count > 1 and nonzero_mean > 0:
                cv2 = (nonzero_sales.std(ddof=0) / nonzero_mean) ** 2
            else:
                cv2 = 0.0
        else:
            adi = np.inf
            cv2 = np.nan

        results.append({
            "item_id": item_id,
            "mean_sales": mean_sales,
            "zero_ratio": zero_ratio,
            "std_sales": sales.std(),
            "ADI": adi,
            "CV2": cv2
        })

    result = pd.DataFrame(results)

    def classify(row):
        if row["ADI"] < 1.32:
            if row["CV2"] < 0.49:
                return "Smooth"
            return "Erratic"
        else:
            if row["CV2"] < 0.49:
                return "Intermittent"
            return "Lumpy"

    result["demand_type"] = result.apply(classify, axis=1)

    return result.sort_values(["demand_type", "mean_sales"], ascending=[True, False])
