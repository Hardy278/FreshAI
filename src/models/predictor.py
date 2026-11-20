
import lightgbm as lgb
import pandas as pd
import numpy as np


DEFAULT_FEATURES = [
    "item_id",
    "day_of_week",
    "day_of_month",
    "month",
    "year",
    "week_of_year",
    "is_weekend",
    "is_holiday",
    "has_price",
    "sell_price",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_7"
]


def prepare_model_data(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list = None
):
    """
    准备 LightGBM 所需的训练集和验证集。

    统一 item_id 的类别定义，避免训练集和验证集
    的类别编码不一致。
    """

    if feature_cols is None:
        feature_cols = DEFAULT_FEATURES

    train = train_df.copy()
    valid = valid_df.copy()

    # 删除历史特征不足的样本
    train = train.dropna(subset=["lag_28", "rolling_mean_28"])
    valid = valid.dropna(subset=["lag_28", "rolling_mean_28"])

    # 使用训练集的商品类别定义
    categories = sorted(train["item_id"].astype(str).unique())

    category_type = pd.CategoricalDtype(categories=categories)

    train["item_id"] = (train["item_id"].astype(str).astype(category_type))
    valid["item_id"] = (valid["item_id"].astype(str).astype(category_type))

    if valid["item_id"].isna().any():
        raise ValueError("验证集中出现训练集未包含的商品类别")

    X_train = train[feature_cols]
    y_train = train["sales"]

    X_valid = valid[feature_cols]
    y_valid = valid["sales"]

    return X_train, y_train, X_valid, y_valid


def train_lightgbm(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list = None
):
    """ 训练 LightGBM 销量预测模型。 """

    if feature_cols is None:
        feature_cols = DEFAULT_FEATURES

    X_train, y_train, X_valid, y_valid = (
        prepare_model_data(train_df, valid_df, feature_cols)
    )

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )

    print("开始训练 LightGBM...")
    print(f"训练样本：{len(X_train)}")
    print(f"验证样本：{len(X_valid)}")

    model.fit(
        X_train,
        y_train,
        categorical_feature=["item_id"],
        eval_set=[(X_valid, y_valid)],
        eval_metric="l1",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100)
        ]
    )

    print("LightGBM 训练完成！")
    print(f"最佳迭代轮数：{model.best_iteration_}")

    return model


def predict_lightgbm(model, df: pd.DataFrame, feature_cols: list = None) -> np.ndarray:
    """ 使用训练好的 LightGBM 模型预测销量。 """

    if feature_cols is None:
        feature_cols = DEFAULT_FEATURES

    X = df[feature_cols].copy()

    # 使用模型训练时的类别定义
    categories = model.booster_.pandas_categorical

    if categories:
        category_type = pd.CategoricalDtype(categories=categories[0])
        
        X["item_id"] = X["item_id"].astype(str).astype(category_type)

    predictions = model.predict(X, num_iteration=model.best_iteration_)

    return np.maximum(predictions, 0)
