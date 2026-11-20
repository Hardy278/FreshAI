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


def _prepare_features(df: pd.DataFrame, feature_cols: list, categories=None):
    """准备模型特征，并统一 item_id 的类别定义。"""

    X = df[feature_cols].copy()

    if "item_id" in feature_cols:
        if categories is None:
            categories = sorted(X["item_id"].astype(str).unique())

        category_type = pd.CategoricalDtype(categories=categories)
        X["item_id"] = X["item_id"].astype(str).astype(category_type)

        if X["item_id"].isna().any():
            raise ValueError("数据中出现训练集未包含的商品类别")

    return X


def prepare_model_data(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list = None
):
    """ 准备 LightGBM 所需的训练集和验证集。 """

    if feature_cols is None:
        feature_cols = DEFAULT_FEATURES

    train = train_df.dropna(subset=["lag_28", "rolling_mean_28"]).copy()
    valid = valid_df.dropna(subset=["lag_28", "rolling_mean_28"]).copy()

    categories = sorted(train["item_id"].astype(str).unique())

    X_train = _prepare_features(train, feature_cols, categories)
    X_valid = _prepare_features(valid, feature_cols, categories)

    y_train = train["sales"]
    y_valid = valid["sales"]

    return X_train, y_train, X_valid, y_valid


def train_lightgbm(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list = None
):
    """训练 LightGBM 销量预测模型，并通过 Early Stopping 选择迭代轮数。"""

    if feature_cols is None:
        feature_cols = DEFAULT_FEATURES

    X_train, y_train, X_valid, y_valid = prepare_model_data(
        train_df, valid_df, feature_cols
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


def train_lightgbm_final(
    train_df: pd.DataFrame,
    n_estimators: int,
    feature_cols: list = None
):
    """使用完整训练数据和固定迭代轮数训练最终模型。"""

    if feature_cols is None:
        feature_cols = DEFAULT_FEATURES

    if n_estimators <= 0:
        raise ValueError("n_estimators 必须大于 0")

    train = train_df.dropna(
        subset=["lag_28", "rolling_mean_28"]
    ).copy()

    categories = sorted(train["item_id"].astype(str).unique())

    X_train = _prepare_features(train, feature_cols, categories)
    y_train = train["sales"]

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=n_estimators,
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

    print("开始训练最终 LightGBM 模型...")
    print(f"训练样本：{len(X_train)}")
    print(f"固定迭代轮数：{n_estimators}")

    model.fit(
        X_train,
        y_train,
        categorical_feature=["item_id"]
    )

    print("最终模型训练完成！")

    return model


def predict_lightgbm(
    model,
    df: pd.DataFrame,
    feature_cols: list = None
) -> np.ndarray:
    """使用训练好的 LightGBM 模型预测销量。"""

    if feature_cols is None:
        feature_cols = DEFAULT_FEATURES

    categories = None

    if "item_id" in feature_cols:
        pandas_categories = model.booster_.pandas_categorical

        if pandas_categories:
            categories = pandas_categories[0]

    X = _prepare_features(df, feature_cols, categories)

    # Early Stopping 模型使用最佳迭代轮数；
    # 固定轮数训练的最终模型使用全部树。
    best_iteration = getattr(model, "best_iteration_", None)

    if best_iteration is not None and best_iteration > 0:
        predictions = model.predict(X, num_iteration=best_iteration)
    else:
        predictions = model.predict(X)

    return np.maximum(predictions, 0)
