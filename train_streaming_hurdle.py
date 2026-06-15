"""Train a Three-Stage Hurdle model to predict Week 1 album streams."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
DEFAULT_INPUT_PATH = DATA_DIR / "album_meta_features.parquet"
DEFAULT_CLASSIFIER_PATH = MODELS_DIR / "streaming_hurdle_classifier.txt"
DEFAULT_REGRESSOR_PATH = MODELS_DIR / "streaming_hurdle_regressor.joblib"
DEFAULT_SUPERSTAR_REGRESSOR_PATH = MODELS_DIR / "streaming_superstar_regressor.joblib"
DEFAULT_META_PATH = MODELS_DIR / "streaming_hurdle_meta.json"
DEFAULT_CLASSIFIER_IMPORTANCE_PATH = MODELS_DIR / "streaming_hurdle_classifier_importance.csv"
DEFAULT_REGRESSOR_IMPORTANCE_PATH = MODELS_DIR / "streaming_hurdle_regressor_lgbm_importance.csv"
DEFAULT_SUPERSTAR_IMPORTANCE_PATH = MODELS_DIR / "streaming_superstar_regressor_importance.csv"
SANDBOX_BASELINES_PATH = DATA_DIR / "artist_sandbox_baselines.csv"

TARGET_COLUMN = "ALBUM_W1_SPB"
GROUP_COLUMN = "DISPLAY_ARTIST"
FEATURE_COLUMNS: list[str] = [
    "TOTAL_TRACKS_ANALYZED",
    "CATALOG_VELOCITY_SLOPE",
    "LEAD_SINGLE_PEAK_VOLUME",
    "RETENTION_RATIO",
    "ACTIVE_SINGLE_COUNT",
    "HISTORICAL_STANDARD_TRACK_SPB",
    "HISTORICAL_MACRO_MOMENTUM",
    "IS_DEBUT_ALBUM",
    "VELOCITY_X_SINGLES",
    "SHORT_TERM_SPIKE_RATIO",
]
STAGE3_FEATURE_COLUMNS = list(FEATURE_COLUMNS)
MONOTONE_POSITIVE_FEATURES: set[str] = set()
EXCLUDED_FROM_TRAINING: list[str] = [
    "ARTIST_ID",
    "MRELG_ID",
    "DISPLAY_ARTIST",
    "FIRST_SALE_DATE",
    "MARKET_WEEK_START",
    "TOTAL_UNIVERSE_STREAMS",
    "ALBUM_TOTAL_W1_STREAMS",
    "ALBUM_W1_SPB",
    "ALBUM_STANDARD_TRACK_SPB",
    "AVG_TRACK_W1_AUDIO_STREAMS",
    "GENRE",
    "LEVEL_2_DISTRIBUTOR",
]
GATEKEEPER_PERCENTILE = 75
GATEKEEPER_PROB_THRESHOLD = 0.5
SUPERSTAR_PERCENTILE = 99
SUPERSTAR_ROUTER_RATIO = 0.8
N_SPLITS = 5
EARLY_STOPPING_ROUNDS = 50
TIME_WEIGHT_HALF_LIFE_DAYS = 365
TIME_WEIGHT_MIN = 0.05
OPTUNA_N_TRIALS = 20
TUNING_HOLDOUT_SIZE = 0.1

STAGE_LABEL_GATEKEEPER = "Gatekeeper_Rejected"
STAGE_LABEL_STANDARD = "Standard_Regressor"
STAGE_LABEL_SUPERSTAR = "Superstar_Regressor"


@dataclass(frozen=True)
class ClassifierCVResults:
    mean_logloss: float
    mean_roc_auc: float
    fold_logloss: list[float]
    fold_roc_auc: list[float]
    mean_best_iteration: int


@dataclass(frozen=True)
class RegressorCVResults:
    mean_rmse: float
    fold_rmse: list[float]
    mean_best_iteration: int
    top_1pct_wmape: float


@dataclass(frozen=True)
class SuperstarCVResults:
    mean_mae: float
    fold_mae: list[float]
    mean_best_iteration: int
    holdout_wmape: float


@dataclass(frozen=True)
class TrainingCounts:
    raw_rows: int
    clean_rows: int
    stage1_rows: int
    stage2_high_tier_rows: int
    stage3_superstar_rows: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Three-Stage Hurdle LightGBM model for Week 1 album streams."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to album_meta_features.parquet",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_DIR,
        help="Directory for saved model artifacts",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=N_SPLITS,
        help="Number of GroupKFold splits",
    )
    parser.add_argument(
        "--gatekeeper-percentile",
        type=float,
        default=GATEKEEPER_PERCENTILE,
        help="Percentile threshold for Stage 1 gatekeeper",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=OPTUNA_N_TRIALS,
        help="Number of Optuna trials for Stage 2/3 hyperparameter search",
    )
    parser.add_argument(
        "--time-weight-half-life-days",
        type=int,
        default=TIME_WEIGHT_HALF_LIFE_DAYS,
        help="Half-life in days for exponential time-decay sample weighting",
    )
    return parser.parse_args()


def engineer_stage3_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return Stage 3 inputs (pre-social baseline uses the same feature set as Stage 2)."""
    return df.copy()


def load_and_clean_data(path: Path) -> tuple[pd.DataFrame, TrainingCounts]:
    """Load feature matrix and drop invalid target rows."""
    print(f"\nLoading data from {path}...")
    df = pd.read_parquet(path)
    raw_rows = len(df)

    required_columns = {TARGET_COLUMN, GROUP_COLUMN, "FIRST_SALE_DATE", *FEATURE_COLUMNS}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Input data is missing required columns: {sorted(missing_columns)}")

    clean_df = df.dropna(subset=[TARGET_COLUMN, GROUP_COLUMN, "FIRST_SALE_DATE"]).copy()
    clean_df["FIRST_SALE_DATE"] = pd.to_datetime(clean_df["FIRST_SALE_DATE"])
    clean_df = clean_df[clean_df[TARGET_COLUMN] > 0]
    clean_df = clean_df[clean_df["TOTAL_TRACKS_ANALYZED"] > 1]

    inf_mask = np.isinf(clean_df[FEATURE_COLUMNS])
    inf_count = int(inf_mask.sum().sum())
    if inf_count:
        clean_df[FEATURE_COLUMNS] = clean_df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
        print(f"  Replaced {inf_count:,} infinite feature values with NaN")

    clean_df = engineer_stage3_features(clean_df)

    counts = TrainingCounts(
        raw_rows=raw_rows,
        clean_rows=len(clean_df),
        stage1_rows=len(clean_df),
        stage2_high_tier_rows=0,
        stage3_superstar_rows=0,
    )
    print(f"  Raw rows:   {raw_rows:,}")
    print(f"  Clean rows: {counts.clean_rows:,} (dropped missing/zero/negative targets)")
    return clean_df, counts


def calculate_dynamic_threshold(y: pd.Series, percentile: float) -> float:
    """Compute the gatekeeper SPB threshold at the given percentile."""
    threshold = float(np.percentile(y, percentile))
    print(f"\nDynamic SPB threshold ({percentile:.0f}th percentile): {threshold:,.6f}")
    return threshold


def calculate_superstar_threshold(y: pd.Series) -> float:
    """Compute the Top 1% superstar SPB threshold."""
    threshold = float(np.percentile(y, SUPERSTAR_PERCENTILE))
    print(f"Superstar threshold ({SUPERSTAR_PERCENTILE}th percentile): {threshold:,.6f}")
    return threshold


def gatekeeper_predict_proba(model: LGBMClassifier, X: pd.DataFrame) -> np.ndarray:
    """Return class-1 probabilities from the native binary classifier."""
    return model.predict_proba(X)[:, 1]


def build_monotone_constraints(feature_columns: list[str]) -> list[int]:
    """Map positive monotonic constraints onto the ordered feature column list."""
    return [1 if feature in MONOTONE_POSITIVE_FEATURES else 0 for feature in feature_columns]


def calculate_time_weights(
    dates: pd.Series,
    reference_date: pd.Timestamp,
    half_life_days: int = TIME_WEIGHT_HALF_LIFE_DAYS,
) -> np.ndarray:
    """Exponential time-decay weights: recent releases near the reference date weigh more."""
    parsed_dates = pd.to_datetime(dates, errors="coerce")
    days_diff = (reference_date - parsed_dates).dt.total_seconds() / 86_400.0
    days_diff = np.maximum(days_diff.to_numpy(dtype=float), 0.0)
    weights = 0.5 ** (days_diff / half_life_days)
    return np.clip(weights, TIME_WEIGHT_MIN, None)


def calculate_wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted MAPE: sum(|error|) / sum(|actual|) * 100."""
    actuals = np.asarray(y_true, dtype=float)
    predictions = np.asarray(y_pred, dtype=float)
    denominator = np.sum(np.abs(actuals))
    if denominator == 0:
        return float("nan")
    return float(np.sum(np.abs(actuals - predictions)) / denominator * 100.0)


def build_classifier() -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        scale_pos_weight=1.5,
        monotone_constraints=build_monotone_constraints(FEATURE_COLUMNS),
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def build_stage2_regressor(params: dict) -> LGBMRegressor:
    """Instantiate Stage 2 Tweedie regressor with Optuna-tuned hyperparameters."""
    return LGBMRegressor(
        objective="tweedie",
        metric="rmse",
        monotone_constraints=build_monotone_constraints(FEATURE_COLUMNS),
        subsample=0.8,
        n_estimators=2000,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        **params,
    )


def build_stage3_regressor(params: dict) -> LGBMRegressor:
    """Instantiate Stage 3 Huber regressor with Optuna-tuned hyperparameters."""
    return LGBMRegressor(
        objective="huber",
        monotone_constraints=build_monotone_constraints(STAGE3_FEATURE_COLUMNS),
        colsample_bytree=0.6,
        subsample=0.8,
        n_estimators=2000,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        **params,
    )


def optimize_stage2(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    train_weights: np.ndarray,
    val_weights: np.ndarray,
    n_trials: int = OPTUNA_N_TRIALS,
) -> dict:
    """Tune Stage 2 Tweedie hyperparameters minimizing validation RMSE."""
    monotone = build_monotone_constraints(FEATURE_COLUMNS)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "tweedie_variance_power": trial.suggest_float("tweedie_variance_power", 1.1, 1.9),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
        }
        model = LGBMRegressor(
            objective="tweedie",
            metric="rmse",
            monotone_constraints=monotone,
            subsample=0.8,
            n_estimators=2000,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            **params,
        )
        model.fit(
            X_train,
            y_train,
            sample_weight=train_weights,
            eval_set=[(X_val, y_val)],
            eval_sample_weight=[val_weights],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        val_preds = model.predict(X_val)
        return float(np.sqrt(mean_squared_error(y_val, val_preds)))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def optimize_stage3(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    train_weights: np.ndarray,
    val_weights: np.ndarray,
    n_trials: int = OPTUNA_N_TRIALS,
) -> dict:
    """Tune Stage 3 Huber hyperparameters minimizing validation MAE on log1p targets."""
    monotone = build_monotone_constraints(STAGE3_FEATURE_COLUMNS)
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 7, 21),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        }
        model = LGBMRegressor(
            objective="huber",
            monotone_constraints=monotone,
            colsample_bytree=0.6,
            subsample=0.8,
            n_estimators=2000,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            **params,
        )
        model.fit(
            X_train,
            y_train_log,
            sample_weight=train_weights,
            eval_set=[(X_val, y_val_log)],
            eval_sample_weight=[val_weights],
            eval_metric="mae",
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        val_preds_log = model.predict(X_val)
        return float(mean_absolute_error(y_val_log, val_preds_log))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def create_tuning_holdout_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    test_size: float = TUNING_HOLDOUT_SIZE,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return group-isolated train/validation indices for Optuna and final early stopping."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    return next(gss.split(X, y, groups))


def predict_pipeline(
    X_input: pd.DataFrame,
    X_stage3: pd.DataFrame,
    classifier: LGBMClassifier,
    regressor_stage2: LGBMRegressor,
    regressor_stage3: LGBMRegressor,
    gatekeeper_threshold: float,
    superstar_threshold: float,
    superstar_router_ratio: float = SUPERSTAR_ROUTER_RATIO,
) -> tuple[np.ndarray, np.ndarray]:
    """Execute the three-stage predictive router and return SPB forecasts with stage labels."""
    n_rows = len(X_input)
    predictions = np.full(n_rows, np.nan, dtype=float)
    stages = np.full(n_rows, STAGE_LABEL_GATEKEEPER, dtype=object)

    gatekeeper_probs = gatekeeper_predict_proba(classifier, X_input[FEATURE_COLUMNS])
    high_tier_mask = gatekeeper_probs >= gatekeeper_threshold
    if not high_tier_mask.any():
        return predictions, stages

    stage2_preds = regressor_stage2.predict(X_input.loc[high_tier_mask, FEATURE_COLUMNS])
    predictions[high_tier_mask] = stage2_preds
    stages[high_tier_mask] = STAGE_LABEL_STANDARD

    router_threshold = superstar_threshold * superstar_router_ratio
    route_mask = np.zeros(n_rows, dtype=bool)
    high_tier_indices = np.where(high_tier_mask)[0]
    route_local_mask = stage2_preds >= router_threshold
    route_mask[high_tier_indices[route_local_mask]] = True

    if route_mask.any():
        stage3_log_preds = regressor_stage3.predict(
            X_stage3.loc[route_mask, STAGE3_FEATURE_COLUMNS]
        )
        predictions[route_mask] = np.expm1(stage3_log_preds)
        stages[route_mask] = STAGE_LABEL_SUPERSTAR

    return predictions, stages


def cross_validate_classifier(
    X: pd.DataFrame,
    y_binary: pd.Series,
    groups: pd.Series,
    sample_weights: np.ndarray,
    n_splits: int,
) -> ClassifierCVResults:
    """Run grouped CV for the Stage 1 gatekeeper classifier."""
    print("\n" + "=" * 60)
    print("Stage 1: Gatekeeper Classifier (GroupKFold CV)")
    print("=" * 60)

    gkf = GroupKFold(n_splits=n_splits)
    fold_logloss: list[float] = []
    fold_roc_auc: list[float] = []
    fold_best_iters: list[int] = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        gkf.split(X, y_binary, groups), start=1
    ):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_binary.iloc[train_idx], y_binary.iloc[val_idx]
        w_train = sample_weights[train_idx]
        w_val = sample_weights[val_idx]

        model = build_classifier()
        model.fit(
            X_train,
            y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            eval_sample_weight=[w_val],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )

        y_prob = gatekeeper_predict_proba(model, X_val)
        fold_logloss.append(float(log_loss(y_val, y_prob)))
        fold_roc_auc.append(float(roc_auc_score(y_val, y_prob)))
        fold_best_iters.append(int(model.best_iteration_))

        print(
            f"  Fold {fold_idx}: logloss={fold_logloss[-1]:.4f}, "
            f"roc_auc={fold_roc_auc[-1]:.4f}, best_iter={fold_best_iters[-1]}"
        )

    mean_logloss = float(np.mean(fold_logloss))
    mean_roc_auc = float(np.mean(fold_roc_auc))
    mean_best_iteration = int(round(np.mean(fold_best_iters)))

    print(
        f"\n  CV averages: logloss={mean_logloss:.4f}, "
        f"roc_auc={mean_roc_auc:.4f}, best_iter={mean_best_iteration}"
    )

    return ClassifierCVResults(
        mean_logloss=mean_logloss,
        mean_roc_auc=mean_roc_auc,
        fold_logloss=fold_logloss,
        fold_roc_auc=fold_roc_auc,
        mean_best_iteration=mean_best_iteration,
    )


def train_final_classifier(
    X: pd.DataFrame,
    y_binary: pd.Series,
    sample_weights: np.ndarray,
    n_estimators: int,
) -> LGBMClassifier:
    """Retrain the gatekeeper on the full dataset with time-decay weighting."""
    print("\nRetraining final Stage 1 classifier on full dataset...")
    model = build_classifier()
    model.set_params(n_estimators=n_estimators)
    model.fit(X, y_binary, sample_weight=sample_weights)
    print(f"  Final classifier trained with n_estimators={n_estimators}")
    return model


def cross_validate_regressor(
    X: pd.DataFrame,
    y_target: pd.Series,
    groups: pd.Series,
    sample_weights: np.ndarray,
    stage2_params: dict,
    n_splits: int,
    superstar_threshold: float,
) -> RegressorCVResults:
    """Run grouped CV for the Stage 2 high-tier LGBM regressor."""
    print("\n" + "=" * 60)
    print("Stage 2: High-Tier Regressor (GroupKFold CV)")
    print("=" * 60)
    print(f"  High-tier subset shape: {X.shape[0]:,} rows x {X.shape[1]} features")
    print(f"  Superstar threshold (SPB): {superstar_threshold:,.6f}")

    gkf = GroupKFold(n_splits=n_splits)
    fold_rmse: list[float] = []
    fold_best_iters: list[int] = []
    oof_actuals: list[np.ndarray] = []
    oof_predictions: list[np.ndarray] = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y_target, groups), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_target.iloc[train_idx], y_target.iloc[val_idx]
        w_train = sample_weights[train_idx]
        w_val = sample_weights[val_idx]

        model = build_stage2_regressor(stage2_params)
        model.fit(
            X_train,
            y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            eval_sample_weight=[w_val],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )

        y_pred = model.predict(X_val)
        fold_rmse.append(float(np.sqrt(mean_squared_error(y_val, y_pred))))
        fold_best_iters.append(int(model.best_iteration_))
        oof_actuals.append(y_val.to_numpy(dtype=float))
        oof_predictions.append(np.asarray(y_pred, dtype=float))

        print(
            f"  Fold {fold_idx}: rmse={fold_rmse[-1]:.4f}, best_iter={fold_best_iters[-1]}"
        )

    mean_rmse = float(np.mean(fold_rmse))
    mean_best_iteration = int(round(np.mean(fold_best_iters)))

    all_actuals = np.concatenate(oof_actuals)
    all_predictions = np.concatenate(oof_predictions)
    top_tier_mask = all_actuals >= superstar_threshold
    if top_tier_mask.any():
        top_1pct_wmape = calculate_wmape(all_actuals[top_tier_mask], all_predictions[top_tier_mask])
    else:
        top_1pct_wmape = float("nan")

    print(f"\n  CV average: rmse={mean_rmse:.4f}, best_iter={mean_best_iteration}")
    print(
        f"  Stage 2 Top 1% Tier wMAPE: {top_1pct_wmape:.2f}% "
        f"({int(top_tier_mask.sum()):,} smash rows)"
    )
    return RegressorCVResults(
        mean_rmse=mean_rmse,
        fold_rmse=fold_rmse,
        mean_best_iteration=mean_best_iteration,
        top_1pct_wmape=top_1pct_wmape,
    )


def cross_validate_superstar_regressor(
    X: pd.DataFrame,
    y_target: pd.Series,
    groups: pd.Series,
    sample_weights: np.ndarray,
    stage3_params: dict,
    n_splits: int,
) -> SuperstarCVResults:
    """Run grouped CV for the Stage 3 superstar regressor on log1p targets."""
    print("\n" + "=" * 60)
    print("Stage 3: Superstar Regressor (GroupKFold CV)")
    print("=" * 60)
    print(f"  Superstar subset shape: {X.shape[0]:,} rows x {X.shape[1]} features")

    y_log = np.log1p(y_target)
    gkf = GroupKFold(n_splits=n_splits)
    fold_mae: list[float] = []
    fold_best_iters: list[int] = []
    oof_actuals: list[np.ndarray] = []
    oof_predictions: list[np.ndarray] = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y_log, groups), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]
        y_val_raw = y_target.iloc[val_idx]
        w_train = sample_weights[train_idx]
        w_val = sample_weights[val_idx]

        model = build_stage3_regressor(stage3_params)
        model.fit(
            X_train,
            y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            eval_sample_weight=[w_val],
            eval_metric="mae",
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )

        y_pred_log = model.predict(X_val)
        y_pred = np.expm1(y_pred_log)
        y_actual = y_val_raw.to_numpy(dtype=float)
        fold_mae.append(float(np.mean(np.abs(y_actual - y_pred))))
        fold_best_iters.append(int(model.best_iteration_))
        oof_actuals.append(y_actual)
        oof_predictions.append(y_pred)

        print(
            f"  Fold {fold_idx}: mae={fold_mae[-1]:.4f}, best_iter={fold_best_iters[-1]}"
        )

    mean_mae = float(np.mean(fold_mae))
    mean_best_iteration = int(round(np.mean(fold_best_iters)))
    holdout_wmape = calculate_wmape(np.concatenate(oof_actuals), np.concatenate(oof_predictions))

    print(f"\n  CV average MAE: {mean_mae:.4f}, best_iter={mean_best_iteration}")
    print(f"  Stage 3 Holdout wMAPE: {holdout_wmape:.2f}%")
    return SuperstarCVResults(
        mean_mae=mean_mae,
        fold_mae=fold_mae,
        mean_best_iteration=mean_best_iteration,
        holdout_wmape=holdout_wmape,
    )


def train_final_regressor(
    X: pd.DataFrame,
    y_target: pd.Series,
    sample_weights: np.ndarray,
    stage2_params: dict,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    val_weights: np.ndarray,
) -> LGBMRegressor:
    """Retrain the high-tier LGBMRegressor on the full subset with Optuna params."""
    print("\nRetraining final Stage 2 LGBMRegressor on full high-tier subset...")
    model = build_stage2_regressor(stage2_params)
    model.fit(
        X,
        y_target,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        eval_sample_weight=[val_weights],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    print(f"  Final regressor trained (best iteration: {model.best_iteration_})")
    return model


def train_final_superstar_regressor(
    X: pd.DataFrame,
    y_target: pd.Series,
    sample_weights: np.ndarray,
    stage3_params: dict,
    X_val: pd.DataFrame,
    y_val_log: pd.Series,
    val_weights: np.ndarray,
) -> LGBMRegressor:
    """Retrain the superstar regressor on the full Top 1% subset with Optuna params."""
    print("\nRetraining final Stage 3 Superstar Regressor on full superstar subset...")
    model = build_stage3_regressor(stage3_params)
    model.fit(
        X,
        np.log1p(y_target),
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val_log)],
        eval_sample_weight=[val_weights],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    print(f"  Final superstar regressor trained (best iteration: {model.best_iteration_})")
    return model


def extract_lgbm_importance(
    model: LGBMClassifier | LGBMRegressor,
    feature_names: list[str],
) -> pd.DataFrame:
    """Extract LightGBM feature importances by gain."""
    importances = model.booster_.feature_importance(importance_type="gain")
    return (
        pd.DataFrame({"feature": feature_names, "importance_gain": importances})
        .sort_values("importance_gain", ascending=False)
        .reset_index(drop=True)
    )


def save_feature_importances(
    classifier: LGBMClassifier,
    regressor: LGBMRegressor,
    superstar_regressor: LGBMRegressor,
    classifier_importance_path: Path,
    regressor_importance_path: Path,
    superstar_importance_path: Path,
) -> None:
    """Print and persist feature importances for all three model stages."""
    classifier_importance = extract_lgbm_importance(classifier, FEATURE_COLUMNS)
    regressor_importance = extract_lgbm_importance(regressor, FEATURE_COLUMNS)
    superstar_importance = extract_lgbm_importance(superstar_regressor, STAGE3_FEATURE_COLUMNS)

    print("\nStage 1 Classifier Top 15 Feature Importances (gain):")
    print(classifier_importance.head(15).to_string(index=False))

    print("\nStage 2 Regressor Top 15 Feature Importances (gain):")
    print(regressor_importance.head(15).to_string(index=False))

    print("\nStage 3 Superstar Regressor Top 10 Feature Importances (gain):")
    print(superstar_importance.head(10).to_string(index=False))

    classifier_importance.to_csv(classifier_importance_path, index=False)
    regressor_importance.to_csv(regressor_importance_path, index=False)
    superstar_importance.to_csv(superstar_importance_path, index=False)
    print(f"\nSaved classifier importances to {classifier_importance_path}")
    print(f"Saved regressor importances to {regressor_importance_path}")
    print(f"Saved superstar importances to {superstar_importance_path}")


def export_sandbox_routing(
    df: pd.DataFrame,
    classifier: LGBMClassifier,
    regressor: LGBMRegressor,
    superstar_regressor: LGBMRegressor,
    gatekeeper_threshold: float,
    superstar_threshold: float,
    output_path: Path,
) -> None:
    """Merge ML routing stages into the A&R sandbox baseline export for executive UI."""
    print("\n" + "=" * 60)
    print("Exporting A&R Sandbox Routing Baselines")
    print("=" * 60)

    latest_by_artist = (
        df.sort_values("FIRST_SALE_DATE", ascending=False)
        .drop_duplicates(subset=["ARTIST_ID", "DISPLAY_ARTIST"], keep="first")
        .reset_index(drop=True)
    )

    predictions, stages = predict_pipeline(
        X_input=latest_by_artist,
        X_stage3=latest_by_artist,
        classifier=classifier,
        regressor_stage2=regressor,
        regressor_stage3=superstar_regressor,
        gatekeeper_threshold=gatekeeper_threshold,
        superstar_threshold=superstar_threshold,
    )

    routing_df = pd.DataFrame(
        {
            "ARTIST_ID": latest_by_artist["ARTIST_ID"].values,
            "DISPLAY_ARTIST": latest_by_artist["DISPLAY_ARTIST"].values,
            "PREDICTED_W1_SPB": predictions,
            "PREDICTION_STAGE": np.where(
                stages == STAGE_LABEL_SUPERSTAR,
                "Superstar Regressor",
                np.where(
                    stages == STAGE_LABEL_STANDARD,
                    "Standard Regressor",
                    "Gatekeeper Rejected",
                ),
            ),
        }
    )

    if output_path.exists():
        sandbox_df = pd.read_csv(output_path)
        sandbox_df.columns = sandbox_df.columns.str.upper()
        merge_cols = [col for col in ["ARTIST_ID", "DISPLAY_ARTIST"] if col in sandbox_df.columns]
        merged = sandbox_df.merge(
            routing_df.rename(
                columns={
                    "PREDICTION_STAGE": "PREDICTION_STAGE",
                    "PREDICTED_W1_SPB": "ML_PREDICTED_W1_SPB",
                }
            ),
            on=merge_cols,
            how="left",
        )
    else:
        merged = routing_df

    merged.to_csv(output_path, index=False)
    superstar_count = int((routing_df["PREDICTION_STAGE"] == "Superstar Regressor").sum())
    standard_count = int((routing_df["PREDICTION_STAGE"] == "Standard Regressor").sum())
    print(f"  Saved sandbox routing to {output_path}")
    print(f"  Routed artists: {len(routing_df):,}")
    print(f"    Superstar Regressor: {superstar_count:,}")
    print(f"    Standard Regressor:  {standard_count:,}")


def save_metadata(
    path: Path,
    dynamic_threshold: float,
    superstar_threshold: float,
    gatekeeper_percentile: float,
    classifier_cv: ClassifierCVResults,
    regressor_cv: RegressorCVResults,
    superstar_cv: SuperstarCVResults,
    counts: TrainingCounts,
    stage2_params: dict,
    stage3_params: dict,
    time_weight_half_life_days: int,
    optuna_trials: int,
) -> None:
    """Persist training metadata as JSON."""
    metadata: dict[str, Any] = {
        "target_column": TARGET_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "stage3_feature_columns": STAGE3_FEATURE_COLUMNS,
        "group_column": GROUP_COLUMN,
        "target_transform": "none",
        "inference_inverse": "none",
        "stage3_target_transform": "log1p",
        "stage3_inference_inverse": "expm1",
        "target_metric": "SPB",
        "gatekeeper_percentile": gatekeeper_percentile,
        "optimal_prod_threshold": GATEKEEPER_PROB_THRESHOLD,
        "dynamic_threshold": dynamic_threshold,
        "superstar_threshold": superstar_threshold,
        "superstar_router_ratio": SUPERSTAR_ROUTER_RATIO,
        "cv_scores": {
            "stage1_logloss": classifier_cv.mean_logloss,
            "stage1_roc_auc": classifier_cv.mean_roc_auc,
            "stage1_fold_logloss": classifier_cv.fold_logloss,
            "stage1_fold_roc_auc": classifier_cv.fold_roc_auc,
            "stage2_rmse": regressor_cv.mean_rmse,
            "stage2_fold_rmse": regressor_cv.fold_rmse,
            "stage2_top_1pct_wmape": regressor_cv.top_1pct_wmape,
            "stage3_mae": superstar_cv.mean_mae,
            "stage3_fold_mae": superstar_cv.fold_mae,
            "stage3_holdout_wmape": superstar_cv.holdout_wmape,
        },
        "training_row_counts": asdict(counts),
        "stage1_best_iteration": classifier_cv.mean_best_iteration,
        "stage2_best_iteration": regressor_cv.mean_best_iteration,
        "stage3_best_iteration": superstar_cv.mean_best_iteration,
        "stage2_optuna_params": stage2_params,
        "stage3_optuna_params": stage3_params,
        "time_weight_half_life_days": time_weight_half_life_days,
        "time_weight_min": TIME_WEIGHT_MIN,
        "optuna_trials": optuna_trials,
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {path}")


def export_artifacts(
    classifier: LGBMClassifier,
    regressor: LGBMRegressor,
    superstar_regressor: LGBMRegressor,
    df: pd.DataFrame,
    models_dir: Path,
    classifier_path: Path,
    regressor_path: Path,
    superstar_regressor_path: Path,
    meta_path: Path,
    classifier_importance_path: Path,
    regressor_importance_path: Path,
    superstar_importance_path: Path,
    sandbox_baselines_path: Path,
    dynamic_threshold: float,
    superstar_threshold: float,
    gatekeeper_percentile: float,
    classifier_cv: ClassifierCVResults,
    regressor_cv: RegressorCVResults,
    superstar_cv: SuperstarCVResults,
    counts: TrainingCounts,
    stage2_params: dict,
    stage3_params: dict,
    time_weight_half_life_days: int,
    optuna_trials: int,
) -> None:
    """Write all trained model artifacts to disk."""
    print("\n" + "=" * 60)
    print("Exporting Model Artifacts")
    print("=" * 60)

    models_dir.mkdir(parents=True, exist_ok=True)

    classifier.booster_.save_model(str(classifier_path))
    print(f"Saved Stage 1 classifier to {classifier_path}")

    joblib.dump({"regressor": regressor, "target_metric": "SPB"}, regressor_path)
    print(f"Saved Stage 2 regressor to {regressor_path}")

    joblib.dump(
        {
            "regressor": superstar_regressor,
            "target_metric": "SPB",
            "target_transform": "log1p",
            "inference_inverse": "expm1",
            "feature_columns": STAGE3_FEATURE_COLUMNS,
        },
        superstar_regressor_path,
    )
    print(f"Saved Stage 3 superstar regressor to {superstar_regressor_path}")

    save_feature_importances(
        classifier=classifier,
        regressor=regressor,
        superstar_regressor=superstar_regressor,
        classifier_importance_path=classifier_importance_path,
        regressor_importance_path=regressor_importance_path,
        superstar_importance_path=superstar_importance_path,
    )
    export_sandbox_routing(
        df=df,
        classifier=classifier,
        regressor=regressor,
        superstar_regressor=superstar_regressor,
        gatekeeper_threshold=GATEKEEPER_PROB_THRESHOLD,
        superstar_threshold=superstar_threshold,
        output_path=sandbox_baselines_path,
    )
    save_metadata(
        path=meta_path,
        dynamic_threshold=dynamic_threshold,
        superstar_threshold=superstar_threshold,
        gatekeeper_percentile=gatekeeper_percentile,
        classifier_cv=classifier_cv,
        regressor_cv=regressor_cv,
        superstar_cv=superstar_cv,
        counts=counts,
        stage2_params=stage2_params,
        stage3_params=stage3_params,
        time_weight_half_life_days=time_weight_half_life_days,
        optuna_trials=optuna_trials,
    )


def main() -> None:
    args = parse_args()
    models_dir = args.models_dir

    classifier_path = models_dir / DEFAULT_CLASSIFIER_PATH.name
    regressor_path = models_dir / DEFAULT_REGRESSOR_PATH.name
    superstar_regressor_path = models_dir / DEFAULT_SUPERSTAR_REGRESSOR_PATH.name
    meta_path = models_dir / DEFAULT_META_PATH.name
    classifier_importance_path = models_dir / DEFAULT_CLASSIFIER_IMPORTANCE_PATH.name
    regressor_importance_path = models_dir / DEFAULT_REGRESSOR_IMPORTANCE_PATH.name
    superstar_importance_path = models_dir / DEFAULT_SUPERSTAR_IMPORTANCE_PATH.name

    df, counts = load_and_clean_data(args.input_path)

    leaked_features = set(FEATURE_COLUMNS) & set(EXCLUDED_FROM_TRAINING)
    if leaked_features:
        raise ValueError(
            f"Feature columns overlap with excluded training columns: {sorted(leaked_features)}"
        )

    X = df[FEATURE_COLUMNS]
    X_stage3 = df[STAGE3_FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    groups = df[GROUP_COLUMN]

    print("\nFeature matrix shape:", X.shape)
    print("Native NaN counts (preserved for LightGBM):")
    for col in FEATURE_COLUMNS:
        print(f"  {col}: {X[col].isna().sum():,}")

    reference_date = pd.Timestamp(df["FIRST_SALE_DATE"].max())
    sample_weights = calculate_time_weights(
        df["FIRST_SALE_DATE"],
        reference_date,
        half_life_days=args.time_weight_half_life_days,
    )
    print(
        f"\nTime-decay weights (reference={reference_date.date()}, "
        f"half-life={args.time_weight_half_life_days}d): "
        f"range={sample_weights.min():.3f}–{sample_weights.max():.3f}"
    )

    dynamic_threshold = calculate_dynamic_threshold(y, args.gatekeeper_percentile)
    superstar_threshold = calculate_superstar_threshold(y)
    y_binary = (y >= dynamic_threshold).astype(int)
    positive_rate = y_binary.mean()
    print(
        f"Stage 1 positive class rate: {positive_rate:.2%} "
        f"({y_binary.sum():,} / {len(y_binary):,})"
    )

    classifier_cv = cross_validate_classifier(
        X=X,
        y_binary=y_binary,
        groups=groups,
        sample_weights=sample_weights,
        n_splits=args.n_splits,
    )
    classifier = train_final_classifier(
        X=X,
        y_binary=y_binary,
        sample_weights=sample_weights,
        n_estimators=classifier_cv.mean_best_iteration,
    )

    high_tier_mask = y >= dynamic_threshold
    high_tier_df = df.loc[high_tier_mask]
    X_high = high_tier_df[FEATURE_COLUMNS]
    y_high = high_tier_df[TARGET_COLUMN]
    groups_high = high_tier_df[GROUP_COLUMN]
    high_tier_weights = sample_weights[high_tier_mask.to_numpy()]

    superstar_mask = y >= superstar_threshold
    superstar_df = df.loc[superstar_mask]
    X_superstar = superstar_df[STAGE3_FEATURE_COLUMNS]
    y_superstar = superstar_df[TARGET_COLUMN]
    groups_superstar = superstar_df[GROUP_COLUMN]
    superstar_weights = sample_weights[superstar_mask.to_numpy()]

    counts = TrainingCounts(
        raw_rows=counts.raw_rows,
        clean_rows=counts.clean_rows,
        stage1_rows=counts.stage1_rows,
        stage2_high_tier_rows=len(high_tier_df),
        stage3_superstar_rows=len(superstar_df),
    )
    print(f"\nStage 2 high-tier rows: {counts.stage2_high_tier_rows:,}")
    print(f"Stage 3 superstar rows: {counts.stage3_superstar_rows:,}")

    stage2_train_idx, stage2_val_idx = create_tuning_holdout_split(
        X_high, y_high, groups_high,
    )
    print(f"\nOptuna tuning Stage 2 Tweedie Regressor ({args.optuna_trials} trials)...")
    stage2_params = optimize_stage2(
        X_high.iloc[stage2_train_idx],
        X_high.iloc[stage2_val_idx],
        y_high.iloc[stage2_train_idx],
        y_high.iloc[stage2_val_idx],
        high_tier_weights[stage2_train_idx],
        high_tier_weights[stage2_val_idx],
        n_trials=args.optuna_trials,
    )
    print(f"  Stage 2 best params: {stage2_params}")

    regressor_cv = cross_validate_regressor(
        X=X_high,
        y_target=y_high,
        groups=groups_high,
        sample_weights=high_tier_weights,
        stage2_params=stage2_params,
        n_splits=args.n_splits,
        superstar_threshold=superstar_threshold,
    )
    regressor = train_final_regressor(
        X=X_high,
        y_target=y_high,
        sample_weights=high_tier_weights,
        stage2_params=stage2_params,
        X_val=X_high.iloc[stage2_val_idx],
        y_val=y_high.iloc[stage2_val_idx],
        val_weights=high_tier_weights[stage2_val_idx],
    )

    if len(superstar_df) < args.n_splits:
        raise ValueError(
            f"Superstar subset has only {len(superstar_df):,} rows; "
            f"cannot run {args.n_splits}-fold CV."
        )

    stage3_train_idx, stage3_val_idx = create_tuning_holdout_split(
        X_superstar, y_superstar, groups_superstar,
    )
    print(f"\nOptuna tuning Stage 3 Superstar Regressor ({args.optuna_trials} trials)...")
    stage3_params = optimize_stage3(
        X_superstar.iloc[stage3_train_idx],
        X_superstar.iloc[stage3_val_idx],
        y_superstar.iloc[stage3_train_idx],
        y_superstar.iloc[stage3_val_idx],
        superstar_weights[stage3_train_idx],
        superstar_weights[stage3_val_idx],
        n_trials=args.optuna_trials,
    )
    print(f"  Stage 3 best params: {stage3_params}")

    superstar_cv = cross_validate_superstar_regressor(
        X=X_superstar,
        y_target=y_superstar,
        groups=groups_superstar,
        sample_weights=superstar_weights,
        stage3_params=stage3_params,
        n_splits=args.n_splits,
    )
    superstar_regressor = train_final_superstar_regressor(
        X=X_superstar,
        y_target=y_superstar,
        sample_weights=superstar_weights,
        stage3_params=stage3_params,
        X_val=X_superstar.iloc[stage3_val_idx],
        y_val_log=np.log1p(y_superstar.iloc[stage3_val_idx]),
        val_weights=superstar_weights[stage3_val_idx],
    )

    pipeline_preds, pipeline_stages = predict_pipeline(
        X_input=df,
        X_stage3=df,
        classifier=classifier,
        regressor_stage2=regressor,
        regressor_stage3=superstar_regressor,
        gatekeeper_threshold=GATEKEEPER_PROB_THRESHOLD,
        superstar_threshold=superstar_threshold,
    )
    routed_mask = ~np.isnan(pipeline_preds)
    if routed_mask.any():
        pipeline_wmape = calculate_wmape(
            y.to_numpy(dtype=float)[routed_mask],
            pipeline_preds[routed_mask],
        )
        superstar_routed = int((pipeline_stages == STAGE_LABEL_SUPERSTAR).sum())
        print(f"\nFull-pipeline routed wMAPE: {pipeline_wmape:.2f}%")
        print(f"Full-pipeline superstar routes: {superstar_routed:,}")

    export_artifacts(
        classifier=classifier,
        regressor=regressor,
        superstar_regressor=superstar_regressor,
        df=df,
        models_dir=models_dir,
        classifier_path=classifier_path,
        regressor_path=regressor_path,
        superstar_regressor_path=superstar_regressor_path,
        meta_path=meta_path,
        classifier_importance_path=classifier_importance_path,
        regressor_importance_path=regressor_importance_path,
        superstar_importance_path=superstar_importance_path,
        sandbox_baselines_path=SANDBOX_BASELINES_PATH,
        dynamic_threshold=dynamic_threshold,
        superstar_threshold=superstar_threshold,
        gatekeeper_percentile=args.gatekeeper_percentile,
        classifier_cv=classifier_cv,
        regressor_cv=regressor_cv,
        superstar_cv=superstar_cv,
        counts=counts,
        stage2_params=stage2_params,
        stage3_params=stage3_params,
        time_weight_half_life_days=args.time_weight_half_life_days,
        optuna_trials=args.optuna_trials,
    )

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
