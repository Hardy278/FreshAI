from pathlib import Path

import time
import joblib
import pandas as pd

from src.models.predictor import train_lightgbm, train_lightgbm_final, predict_lightgbm
from src.models.evaluator import evaluate_model, compare_models, split_train_valid_test
from src.models.baseline import predict_lag7, predict_rolling_mean7


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "foods1_ca1_features.parquet"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def run_training_pipeline(
    features_path: Path = FEATURES_PATH,
    artifacts_dir: Path = ARTIFACTS_DIR,
    validation_days: int = 28,
    test_days: int = 28
):
    """运行完整的 LightGBM 销量预测训练与评估流程。"""

    pipeline_start = time.perf_counter()
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载特征数据
    print("===== 1. 加载特征数据 =====")

    df = pd.read_parquet(features_path)
    df["date"] = pd.to_datetime(df["date"])

    print(f"数据规模：{df.shape}")

    # 2. 划分数据集
    print("\n===== 2. 划分数据集 =====")

    train_df, valid_df, test_df = split_train_valid_test(
        df,
        validation_days=validation_days,
        test_days=test_days
    )

    # 3. 使用验证集确定最佳迭代轮数
    print("\n===== 3. 模型选择 =====")

    selection_model = train_lightgbm(train_df, valid_df)
    best_iteration = selection_model.best_iteration_

    if best_iteration is None or best_iteration <= 0:
        raise RuntimeError("模型没有获得有效的最佳迭代轮数")

    print(f"选定最佳迭代轮数：{best_iteration}")

    # 4. 合并训练集和验证集，重新训练最终模型
    print("\n===== 4. 训练最终模型 =====")

    train_valid_df = pd.concat( [train_df, valid_df], ignore_index=True)
    final_model = train_lightgbm_final(train_valid_df, n_estimators=best_iteration)

    # 5. 在测试集上进行最终评估
    print("\n===== 5. 测试集评估 =====")

    predictions = predict_lightgbm(final_model, test_df)
    test_result = evaluate_model(test_df["sales"], predictions)

    # 6. 与基线模型比较
    print("\n===== 6. 基线模型比较 =====")

    results = {
        "LightGBM": test_result,
        "Rolling Mean-7": evaluate_model(test_df["sales"], predict_rolling_mean7(test_df)),
        "Lag-7": evaluate_model(test_df["sales"], predict_lag7(test_df))
    }
    comparison = compare_models(results)

    print("\n最终测试集结果：")
    print(comparison)

    # 7. 保存模型和实验结果
    print("\n===== 7. 保存实验产物 =====")

    model_path = artifacts_dir / "lightgbm_final.pkl"
    comparison_path = artifacts_dir / "final_test_comparison.csv"

    joblib.dump(final_model, model_path)

    comparison.to_csv(comparison_path, encoding="utf-8-sig")

    print(f"模型保存位置：{model_path}")
    print(f"评估结果保存位置：{comparison_path}")

    total_time = time.perf_counter() - pipeline_start

    print(f"\n总耗时：{total_time:.2f} 秒")

    return {
        "model": final_model,
        "best_iteration": best_iteration,
        "comparison": comparison,
        "predictions": predictions,
        "test_df": test_df
    }
