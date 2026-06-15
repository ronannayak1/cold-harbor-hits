"""Train Stage 4 Chartex viral multiplier (delta residual) on top of the hurdle pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit

from train_streaming_hurdle import (
    FEATURE_COLUMNS,
    GROUP_COLUMN,
    STAGE3_FEATURE_COLUMNS,
    TARGET_COLUMN,
    calculate_wmape,
    engineer_stage3_features,
    predict_pipeline,
)

DATA_DIR = Path("data")
MODELS_DIR = Path("models")

FEATURE_MATRIX_PATH = DATA_DIR / "album_meta_features.parquet"
CHARTEX_PATH = DATA_DIR / "chartex_velocity_training.csv"
CLASSIFIER_PATH = MODELS_DIR / "streaming_hurdle_classifier.txt"
REGRESSOR_PATH = MODELS_DIR / "streaming_hurdle_regressor.joblib"
SUPERSTAR_REGRESSOR_PATH = MODELS_DIR / "streaming_superstar_regressor.joblib"
META_PATH = MODELS_DIR / "streaming_hurdle_meta.json"
OUTPUT_MODEL_PATH = MODELS_DIR / "chartex_multiplier_regressor.joblib"

COHORT_START_DATE = "2026-01-12"
MULTIPLIER_MIN = 0.1
MULTIPLIER_MAX = 100.0
PEER_BENCHMARK_SPB = 15.0
VALIDATION_SIZE = 0.15
RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 50

CHARTEX_RAW_COLUMNS = [
    "TRENDING_SOUNDS_COUNT",
    "ALBUM_TOTAL_TT_VIDEOS",
    "ALBUM_TT_VIDEOS_LAST_7D",
    "ALBUM_TT_VIDEOS_LAST_24H",
    "ALBUM_TOTAL_TT_VIEWS",
    "ALBUM_TOTAL_TT_LIKES",
    "ALBUM_TOTAL_TT_SAVES",
    "ALBUM_TOTAL_TT_SHARES",
]

CHARTEX_ENGINEERED_COLUMNS = [
    "TT_VIRAL_CONCENTRATION",
    "TT_TERMINAL_ACCELERATION",
    "TT_ENGAGEMENT_DEPTH",
    "TT_LIKE_RATIO",
]

CHARTEX_FEATURES = CHARTEX_ENGINEERED_COLUMNS + [
    "TRENDING_SOUNDS_COUNT",
    "ALBUM_TOTAL_TT_VIEWS",
    "ALBUM_TOTAL_TT_VIDEOS",
]


class BoosterClassifierAdapter:
    """Sklearn-compatible wrapper for a serialized LightGBM Booster classifier."""

    def __init__(self, booster: lgb.Booster):
        self._booster = booster

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        positive_prob = np.asarray(self._booster.predict(X), dtype=float)
        return np.column_stack([1.0 - positive_prob, positive_prob])


def load_primary_models() -> tuple[LGBMClassifier, LGBMRegressor, LGBMRegressor, dict]:
    """Load trained hurdle artifacts and metadata."""
    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(f"Classifier not found: {CLASSIFIER_PATH}")
    if not REGRESSOR_PATH.exists():
        raise FileNotFoundError(f"Stage 2 regressor not found: {REGRESSOR_PATH}")
    if not SUPERSTAR_REGRESSOR_PATH.exists():
        raise FileNotFoundError(f"Stage 3 regressor not found: {SUPERSTAR_REGRESSOR_PATH}")
    if not META_PATH.exists():
        raise FileNotFoundError(f"Metadata not found: {META_PATH}")

    classifier = BoosterClassifierAdapter(lgb.Booster(model_file=str(CLASSIFIER_PATH)))

    stage2_payload = joblib.load(REGRESSOR_PATH)
    regressor = stage2_payload["regressor"] if isinstance(stage2_payload, dict) else stage2_payload

    stage3_payload = joblib.load(SUPERSTAR_REGRESSOR_PATH)
    superstar_regressor = (
        stage3_payload["regressor"] if isinstance(stage3_payload, dict) else stage3_payload
    )

    with META_PATH.open(encoding="utf-8") as handle:
        metadata = json.load(handle)

    return classifier, regressor, superstar_regressor, metadata


def load_cohort_feature_matrix(path: Path, cohort_start: str) -> pd.DataFrame:
    """Load parquet features and restrict to the 2026 Chartex training window."""
    df = pd.read_parquet(path)
    df["FIRST_SALE_DATE"] = pd.to_datetime(df["FIRST_SALE_DATE"])
    cohort = df[df["FIRST_SALE_DATE"] >= pd.Timestamp(cohort_start)].copy()
    cohort = cohort.dropna(subset=[TARGET_COLUMN, GROUP_COLUMN, "MRELG_ID"])
    cohort = cohort[cohort[TARGET_COLUMN] > 0]
    return cohort.reset_index(drop=True)


def add_primary_predictions(
    cohort_df: pd.DataFrame,
    classifier: LGBMClassifier,
    regressor: LGBMRegressor,
    superstar_regressor: LGBMRegressor,
    metadata: dict,
) -> pd.DataFrame:
    """Run the three-stage hurdle router to produce PRIMARY_PREDICTED_SPB."""
    prepared = engineer_stage3_features(cohort_df.copy())
    missing_features = set(FEATURE_COLUMNS) - set(prepared.columns)
    if missing_features:
        raise ValueError(f"Cohort data is missing hurdle features: {sorted(missing_features)}")

    primary_spb, prediction_stage = predict_pipeline(
        X_input=prepared,
        X_stage3=prepared,
        classifier=classifier,
        regressor_stage2=regressor,
        regressor_stage3=superstar_regressor,
        gatekeeper_threshold=float(metadata["optimal_prod_threshold"]),
        superstar_threshold=float(metadata["superstar_threshold"]),
        superstar_router_ratio=float(metadata.get("superstar_router_ratio", 0.8)),
    )

    enriched = prepared.copy()
    enriched["PRIMARY_PREDICTED_SPB"] = primary_spb
    enriched["PRIMARY_PREDICTION_STAGE"] = prediction_stage
    return enriched


def load_and_merge_chartex_features(cohort_df: pd.DataFrame, chartex_path: Path) -> pd.DataFrame:
    """Left-join Chartex velocity features; missing TikTok metrics remain NaN."""
    if not chartex_path.exists():
        raise FileNotFoundError(f"Chartex training file not found: {chartex_path}")

    chartex_df = pd.read_csv(chartex_path)
    chartex_df.columns = chartex_df.columns.str.upper()
    chartex_df = chartex_df.drop_duplicates(subset=["MRELG_ID"], keep="last")

    merge_columns = ["MRELG_ID", *CHARTEX_RAW_COLUMNS]
    missing_chartex_cols = set(merge_columns) - set(chartex_df.columns)
    if missing_chartex_cols:
        raise ValueError(f"Chartex CSV is missing required columns: {sorted(missing_chartex_cols)}")

    merged = cohort_df.merge(chartex_df[merge_columns], on="MRELG_ID", how="left")
    return merged


def _safe_positive_denominator(series: pd.Series) -> pd.Series:
    """Floor positive denominators at 1.0 while preserving NaN for missing TikTok data."""
    values = series.astype(float)
    safe_values = values.copy()
    valid_mask = values.notna()
    safe_values.loc[valid_mask] = np.maximum(values.loc[valid_mask], 1.0)
    return safe_values


def engineer_chartex_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build TikTok acceleration and engagement ratios for Stage 4."""
    enriched = df.copy()
    safe_total_videos = _safe_positive_denominator(enriched["ALBUM_TOTAL_TT_VIDEOS"])
    safe_total_views = _safe_positive_denominator(enriched["ALBUM_TOTAL_TT_VIEWS"])

    enriched["TT_VIRAL_CONCENTRATION"] = (
        enriched["ALBUM_TT_VIDEOS_LAST_7D"] / safe_total_videos
    )
    enriched["TT_TERMINAL_ACCELERATION"] = (
        enriched["ALBUM_TT_VIDEOS_LAST_24H"] / safe_total_videos
    )
    enriched["TT_ENGAGEMENT_DEPTH"] = (
        enriched["ALBUM_TOTAL_TT_SAVES"] + enriched["ALBUM_TOTAL_TT_SHARES"]
    ) / safe_total_views
    enriched["TT_LIKE_RATIO"] = enriched["ALBUM_TOTAL_TT_LIKES"] / safe_total_views
    return enriched


def fill_gatekeeper_spb(df: pd.DataFrame) -> pd.DataFrame:
    """Backfill gatekeeper-rejected rows with a peer-style SPB proxy before multiplier training."""
    enriched = df.copy()
    missing_mask = enriched["PRIMARY_PREDICTED_SPB"].isna()
    if not missing_mask.any():
        return enriched

    effective_tracks = 12.0 * (
        np.log(enriched["TOTAL_TRACKS_ANALYZED"].fillna(12.0) + 1.0) / np.log(13.0)
    )
    historical_spb = enriched["HISTORICAL_STANDARD_TRACK_SPB"]
    spike = enriched["SHORT_TERM_SPIKE_RATIO"].fillna(1.0).clip(0.5, 3.0)
    fallback_spb = effective_tracks * historical_spb * spike

    bad_historical = historical_spb.isna() | (historical_spb == 0)
    fallback_spb = fallback_spb.where(~bad_historical, effective_tracks * PEER_BENCHMARK_SPB)

    enriched.loc[missing_mask, "PRIMARY_PREDICTED_SPB"] = fallback_spb.loc[missing_mask]
    return enriched


def build_training_frame(cohort_df: pd.DataFrame) -> pd.DataFrame:
    """Attach Chartex features, fill gatekeeper SPB, and compute log viral multiplier target."""
    with_chartex = load_and_merge_chartex_features(cohort_df, CHARTEX_PATH)
    engineered = engineer_chartex_features(with_chartex)
    filled = fill_gatekeeper_spb(engineered)

    train_df = filled[filled["PRIMARY_PREDICTED_SPB"] > 0].copy()
    raw_ratio = train_df[TARGET_COLUMN] / train_df["PRIMARY_PREDICTED_SPB"]
    raw_ratio = raw_ratio.clip(lower=MULTIPLIER_MIN, upper=MULTIPLIER_MAX)
    train_df["VIRAL_MULTIPLIER"] = np.log(raw_ratio)
    return train_df.reset_index(drop=True)


def build_chartex_regressor() -> LGBMRegressor:
    """Highly regularized Huber regressor for small-sample TikTok residual correction."""
    return LGBMRegressor(
        objective="huber",
        num_leaves=7,
        min_child_samples=15,
        learning_rate=0.02,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def weighted_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Volume-weighted MAPE on SPB (fraction, not percent)."""
    return calculate_wmape(actual, predicted) / 100.0


def train_chartex_multiplier(train_df: pd.DataFrame) -> tuple[LGBMRegressor, dict[str, float]]:
    """Train Stage 4 with grouped holdout early stopping and report residual metrics."""
    if train_df.empty:
        raise ValueError("Training frame is empty after preparing Chartex multiplier targets.")

    X = train_df[CHARTEX_FEATURES]
    y = train_df["VIRAL_MULTIPLIER"]
    groups = train_df[GROUP_COLUMN]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
    )
    train_idx, val_idx = next(splitter.split(X, y, groups))

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = build_chartex_regressor()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )

    val_df = train_df.iloc[val_idx].copy()
    predicted_multiplier = np.exp(model.predict(X_val))
    predicted_multiplier = np.clip(predicted_multiplier, MULTIPLIER_MIN, MULTIPLIER_MAX)
    actual_multiplier = np.exp(y_val.to_numpy(dtype=float))

    actual_spb = val_df[TARGET_COLUMN].to_numpy(dtype=float)
    primary_spb = val_df["PRIMARY_PREDICTED_SPB"].to_numpy(dtype=float)
    cascaded_spb = primary_spb * predicted_multiplier

    metrics = {
        "holdout_rows": float(len(val_df)),
        "multiplier_mae": float(mean_absolute_error(actual_multiplier, predicted_multiplier)),
        "baseline_wmape": weighted_mape(actual_spb, primary_spb),
        "cascaded_wmape": weighted_mape(actual_spb, cascaded_spb),
    }
    return model, metrics


def export_model(model: LGBMRegressor, metrics: dict[str, float]) -> None:
    """Persist Stage 4 regressor and feature ordering for downstream inference."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "regressor": model,
        "chartex_features": CHARTEX_FEATURES,
        "multiplier_min": MULTIPLIER_MIN,
        "multiplier_max": MULTIPLIER_MAX,
        "target_transform": "log",
        "cohort_start_date": COHORT_START_DATE,
        "holdout_metrics": metrics,
    }
    joblib.dump(payload, OUTPUT_MODEL_PATH)


def print_summary(cohort_df: pd.DataFrame, train_df: pd.DataFrame, metrics: dict[str, float]) -> None:
    """Print training cohort and holdout evaluation summary."""
    routed_mask = cohort_df["PRIMARY_PREDICTED_SPB"].notna()
    gatekeeper_filled = int((~routed_mask).sum())
    print("\n" + "=" * 72)
    print("STAGE 4 — CHARTEX VIRAL MULTIPLIER TRAINING")
    print("=" * 72)
    print(f"Cohort start date:              {COHORT_START_DATE}")
    print(f"2026 cohort rows:               {len(cohort_df):,}")
    print(f"Hurdle-routed primary SPB:        {int(routed_mask.sum()):,}")
    print(f"Gatekeeper rows (fallback SPB):   {gatekeeper_filled:,}")
    print(f"Stage 4 training rows:          {len(train_df):,}")
    print(f"Target transform:               log (cap [{MULTIPLIER_MIN}, {MULTIPLIER_MAX}])")
    print(f"Chartex feature count:          {len(CHARTEX_FEATURES)}")
    print("-" * 72)
    print("Holdout Evaluation (15% artist-group split)")
    print(f"  Holdout rows:                 {int(metrics['holdout_rows']):,}")
    print(f"  Multiplier MAE:               {metrics['multiplier_mae']:.4f}")
    print(f"  Baseline wMAPE (SPB):         {metrics['baseline_wmape']:.2%}")
    print(f"  Cascaded wMAPE (SPB):         {metrics['cascaded_wmape']:.2%}")
    print(f"  wMAPE improvement:            {(metrics['baseline_wmape'] - metrics['cascaded_wmape']):.2%} absolute")
    print("-" * 72)
    print(f"Saved model to:                 {OUTPUT_MODEL_PATH}")
    print("=" * 72 + "\n")


def main() -> None:
    classifier, regressor, superstar_regressor, metadata = load_primary_models()
    cohort_df = load_cohort_feature_matrix(FEATURE_MATRIX_PATH, COHORT_START_DATE)
    cohort_df = add_primary_predictions(
        cohort_df,
        classifier,
        regressor,
        superstar_regressor,
        metadata,
    )
    train_df = build_training_frame(cohort_df)
    model, metrics = train_chartex_multiplier(train_df)
    export_model(model, metrics)
    print_summary(cohort_df, train_df, metrics)


if __name__ == "__main__":
    main()
