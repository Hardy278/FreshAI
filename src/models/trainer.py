from pathlib import Path
import time
import joblib
import pandas as pd

from src.models.predictor import (
    train_lightgbm,
    train_lightgbm_final,
    predict_lightgbm
)
from src.models.baseline import (
    predict_lag7,
    predict_rolling_mean7
)
from src.models.evaluator import (
    evaluate_model,
    compare_models,
    split_train_valid_test,
    analyze_item_errors,
    analyze_demand_patterns
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "foods1_ca1_features.parquet"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def load_feature_data(features_path=FEATURES_PATH):
    """加载特征数据。"""
    print("\n===== 1. 加载特征数据 =====")

    if not features_path.exists():
        raise FileNotFoundError(f"特征文件不存在：{features_path}")

    df = pd.read_parquet(features_path)
    df["date"] = pd.to_datetime(df["date"])

    print(f"数据规模：{df.shape}")

    return df


def analyze_demand_data(df):
    """分析商品需求模式。

    注意：
    当前分析使用完整历史数据，仅用于事后诊断，
    不作为模型输入特征。
    """
    print("\n===== 2. 商品需求模式分析 =====")

    demand_patterns = analyze_demand_patterns(df)
    type_counts = demand_patterns["demand_type"].value_counts()

    print("\n需求类型统计：")
    print(type_counts)

    print("\n商品需求模式明细 Top 20：")
    print(demand_patterns.head(20))

    return demand_patterns


def split_dataset(df):
    """按时间划分 Train、Valid、Test。"""
    print("\n===== 3. 划分数据集 =====")

    train_df, valid_df, test_df = split_train_valid_test(
        df,
        validation_days=28,
        test_days=28
    )

    print("三段式时间切分完成")
    print(
        f"Train: {train_df['date'].min().date()} "
        f"至 {train_df['date'].max().date()} "
        f"| {len(train_df)} 条记录"
    )
    print(
        f"Valid: {valid_df['date'].min().date()} "
        f"至 {valid_df['date'].max().date()} "
        f"| {len(valid_df)} 条记录"
    )
    print(
        f"Test: {test_df['date'].min().date()} "
        f"至 {test_df['date'].max().date()} "
        f"| {len(test_df)} 条记录"
    )

    return train_df, valid_df, test_df


def select_model(train_df, valid_df):
    """使用验证集进行模型选择。

    通过 Early Stopping 确定最佳迭代轮数。
    """
    print("\n===== 4. 模型选择 =====")

    print("开始训练 LightGBM...")

    model = train_lightgbm(train_df, valid_df)
    best_iteration = model.best_iteration_

    if not best_iteration or best_iteration <= 0:
        raise RuntimeError("LightGBM 未能获得有效的最佳迭代轮数")

    print("选定最佳迭代轮数：", best_iteration)

    return model, best_iteration


def train_final_model(train_df, valid_df, best_iteration):
    """合并 Train 和 Valid，使用确定的迭代轮数重新训练最终模型。"""
    print("\n===== 5. 训练最终模型 =====")

    train_valid_df = pd.concat([train_df, valid_df], ignore_index=True)

    print("开始训练最终 LightGBM 模型...")

    final_model = train_lightgbm_final(
        train_valid_df,
        n_estimators=best_iteration
    )

    print("最终模型训练完成！")

    return final_model


def generate_predictions(model, test_df):
    """使用最终模型生成测试集预测。"""
    print("\n===== 6. 测试集预测 =====")

    predictions = predict_lightgbm(model, test_df)

    print("预测完成！")

    return predictions


def evaluate_all_models(test_df, predictions):
    """比较 LightGBM 与两个基线模型。"""
    print("\n===== 7. 整体模型评估 =====")

    y_true = test_df["sales"].to_numpy()

    # LightGBM
    lightgbm_metrics = evaluate_model(y_true, predictions)

    # Rolling Mean-7
    rolling_predictions = predict_rolling_mean7(test_df)
    rolling_metrics = evaluate_model(y_true, rolling_predictions)

    # Lag-7
    lag_predictions = predict_lag7(test_df)
    lag_metrics = evaluate_model(y_true, lag_predictions)

    results = {
        "LightGBM": lightgbm_metrics,
        "Rolling Mean-7": rolling_metrics,
        "Lag-7": lag_metrics
    }

    comparison = compare_models(results)

    print("\n最终测试集结果：")
    print(comparison)

    return comparison


def analyze_prediction_errors(test_df, predictions):
    """按商品分析预测误差。"""
    print("\n===== 8. 商品级误差分析 =====")

    item_errors = analyze_item_errors(
        y_true=test_df["sales"],
        y_pred=predictions,
        item_ids=test_df["item_id"]
    )

    print("\n商品误差最大的 Top 10：")
    print(item_errors.head(10))

    print("\n商品误差最小的 Top 10：")
    print(item_errors.tail(10))

    return item_errors


def analyze_errors_by_demand_type(test_df, predictions, demand_patterns):
    """按 Smooth、Erratic、Intermittent、Lumpy 四种需求类型统计测试集误差。"""
    print("\n===== 9. 按需求类型分析预测误差 =====")

    type_mapping = demand_patterns.set_index("item_id")["demand_type"]

    type_test_df = test_df[["item_id", "sales"]].copy()
    type_test_df["prediction"] = predictions
    type_test_df["demand_type"] = type_test_df["item_id"].map(type_mapping)

    if type_test_df["demand_type"].isna().any():
        raise ValueError("部分测试集商品没有对应的需求类型")

    type_test_df["abs_error"] = (type_test_df["sales"] - type_test_df["prediction"]).abs()
    type_test_df["squared_error"] = (type_test_df["sales"] - type_test_df["prediction"]) ** 2

    type_results = type_test_df.groupby("demand_type").agg(
        item_count=("item_id", "nunique"),
        actual_sales=("sales", "sum"),
        predicted_sales=("prediction", "sum"),
        MAE=("abs_error", "mean"),
        squared_error=("squared_error", "mean"),
        total_abs_error=("abs_error", "sum")
    )

    type_results["RMSE"] = type_results["squared_error"] ** 0.5

    type_results["WAPE"] = (
        type_results["total_abs_error"]
        / type_results["actual_sales"].replace(0, float("nan"))
    )

    type_results["Bias"] = type_results["predicted_sales"] - type_results["actual_sales"]
    type_results = type_results.drop(columns=["squared_error", "total_abs_error"])

    print("\n各需求类型测试集表现：")
    print(type_results)

    return type_results


def save_experiment_artifacts(
    model,
    comparison,
    item_errors,
    demand_patterns,
    type_results,
    test_df,
    predictions,
    best_iteration,
    artifacts_dir=ARTIFACTS_DIR
):
    """保存模型、评估结果和预测明细。"""
    print("\n===== 10. 保存实验产物 =====")

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 保存最终模型
    model_path = artifacts_dir / "lightgbm_final.pkl"
    joblib.dump(model, model_path)

    # 保存整体模型比较结果
    comparison_path = artifacts_dir / "final_test_comparison.csv"
    comparison.to_csv(comparison_path, encoding="utf-8-sig")

    # 保存商品级误差
    item_errors_path = artifacts_dir / "item_error_analysis.csv"
    item_errors.to_csv(item_errors_path, encoding="utf-8-sig")

    # 保存需求模式
    demand_patterns_path = artifacts_dir / "demand_patterns.csv"
    demand_patterns.to_csv(demand_patterns_path, index=False, encoding="utf-8-sig")

    # 保存需求类型误差
    type_results_path = artifacts_dir / "demand_type_evaluation.csv"
    type_results.to_csv(type_results_path, encoding="utf-8-sig")

    # 保存测试集预测明细
    prediction_df = test_df[["item_id", "date", "sales"]].copy()
    prediction_df["prediction"] = predictions

    prediction_path = artifacts_dir / "test_predictions.csv"
    prediction_df.to_csv(prediction_path, index=False, encoding="utf-8-sig")

    # 保存实验配置
    config_path = artifacts_dir / "training_config.csv"
    pd.DataFrame([{
        "model": "LightGBM",
        "best_iteration": best_iteration,
        "validation_days": 28,
        "test_days": 28,
        "feature_file": str(FEATURES_PATH)
    }]).to_csv(config_path, index=False, encoding="utf-8-sig")

    print(f"模型保存位置：{model_path}")
    print(f"评估结果保存位置：{comparison_path}")
    print(f"商品误差分析：{item_errors_path}")
    print(f"需求模式分析：{demand_patterns_path}")
    print(f"需求类型误差：{type_results_path}")
    print(f"预测明细：{prediction_path}")
    print(f"实验配置：{config_path}")


def run_training_pipeline():
    """执行完整训练实验。

    该函数只负责组织各个步骤，
    不直接实现具体的数据处理或训练逻辑。
    """
    start_time = time.perf_counter()

    # 1. 加载数据
    df = load_feature_data()

    # 2. 需求模式分析
    demand_patterns = analyze_demand_data(df)

    # 3. 数据集划分
    train_df, valid_df, test_df = split_dataset(df)

    # 4. 模型选择
    _, best_iteration = select_model(train_df, valid_df)

    # 5. 训练最终模型
    final_model = train_final_model(train_df, valid_df, best_iteration)

    # 6. 测试集预测
    predictions = generate_predictions(final_model, test_df)

    # 7. 整体模型评估
    comparison = evaluate_all_models(test_df, predictions)

    # 8. 商品级误差分析
    item_errors = analyze_prediction_errors(test_df, predictions)

    # 9. 按需求类型分析误差
    type_results = analyze_errors_by_demand_type(
        test_df,
        predictions,
        demand_patterns
    )

    # 10. 保存实验结果
    save_experiment_artifacts(
        model=final_model,
        comparison=comparison,
        item_errors=item_errors,
        demand_patterns=demand_patterns,
        type_results=type_results,
        test_df=test_df,
        predictions=predictions,
        best_iteration=best_iteration
    )

    elapsed_time = time.perf_counter() - start_time

    print(f"\n总耗时：{elapsed_time:.2f} 秒")

    print("\n===== 实验完成 =====")
    print("最佳迭代轮数：", best_iteration)
    print(comparison)

    return {
        "model": final_model,
        "best_iteration": best_iteration,
        "comparison": comparison,
        "predictions": predictions,
        "test_df": test_df,
        "item_errors": item_errors,
        "demand_patterns": demand_patterns,
        "type_results": type_results
    }
