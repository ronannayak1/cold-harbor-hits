"""Train Stage 4 DAP social viral multiplier (delta residual) on top of the hurdle pipeline."""

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

from dap_social_features import (
    DAP_SOCIAL_ENGINEERED_COLUMNS,
    DAP_SOCIAL_RAW_COLUMNS,
    REAL_SOCIAL_MIN_VIEWS,
    SOCIAL_FEATURES,
    VIRAL_TERMINAL_ACCELERATION_THRESHOLD,
    annotate_social_gates,
    assess_stage4_multiplier_health,
    engineer_dap_social_features,
)
from train_streaming_hurdle import (
    FEATURE_COLUMNS,
    GROUP_COLUMN,
    STAGE_LABEL_GATEKEEPER,
    STAGE_LABEL_STANDARD,
    STAGE_LABEL_SUPERSTAR,
    TARGET_COLUMN,
    calculate_wmape,
    engineer_stage3_features,
    predict_pipeline,
)

DATA_DIR = Path("data")
MODELS_DIR = Path("models")

FEATURE_MATRIX_PATH = DATA_DIR / "album_meta_features.parquet"
DAP_SOCIAL_PATH = DATA_DIR / "dap_social_pre_release.csv"
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

# Back-compat aliases used by backtests / inference loaders.
CHARTEX_RAW_COLUMNS = DAP_SOCIAL_RAW_COLUMNS
CHARTEX_ENGINEERED_COLUMNS = DAP_SOCIAL_ENGINEERED_COLUMNS
CHARTEX_FEATURES = SOCIAL_FEATURES
CHARTEX_PATH = DAP_SOCIAL_PATH


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
    """Load parquet features and restrict to the social training window."""
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


def load_and_merge_chartex_features(cohort_df: pd.DataFrame, social_path: Path) -> pd.DataFrame:
    """Left-join DAP social features; missing metrics remain NaN for LightGBM."""
    if not social_path.exists():
        raise FileNotFoundError(
            f"DAP social training file not found: {social_path}. "
            "Run fetch_dap_social_features.py first."
        )

    social_df = pd.read_csv(social_path)
    social_df.columns = social_df.columns.str.upper()
    social_df = social_df.drop_duplicates(subset=["MRELG_ID"], keep="last")

    merge_columns = ["MRELG_ID", *DAP_SOCIAL_RAW_COLUMNS]
    missing_social_cols = set(merge_columns) - set(social_df.columns)
    if missing_social_cols:
        raise ValueError(f"DAP social CSV is missing required columns: {sorted(missing_social_cols)}")

    return cohort_df.merge(social_df[merge_columns], on="MRELG_ID", how="left")


def engineer_chartex_features(df: pd.DataFrame) -> pd.DataFrame:
    """Back-compat wrapper around DAP social feature engineering."""
    return engineer_dap_social_features(df)


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
    """Attach DAP social features and build Stage 4 targets on the gated population.

    Training / evaluation population:
      - Hurdle-routed Standard albums with real social signal, OR
      - Any row with real social signal that is not Superstar
    Superstar and no-social paths are excluded so Stage 4 cannot shrink them.
    Gatekeeper rows are included only when they have real social (eval/train residual
    on social signal), but primary SPB still uses the fallback fill for the ratio.
    """
    with_social = load_and_merge_chartex_features(cohort_df, DAP_SOCIAL_PATH)
    engineered = engineer_dap_social_features(with_social)
    filled = fill_gatekeeper_spb(engineered)
    gated = annotate_social_gates(filled)

    not_superstar = gated["PRIMARY_PREDICTION_STAGE"] != STAGE_LABEL_SUPERSTAR
    eligible = gated["HAS_REAL_SOCIAL"] & not_superstar & (gated["PRIMARY_PREDICTED_SPB"] > 0)
    train_df = gated.loc[eligible].copy()

    raw_ratio = train_df[TARGET_COLUMN] / train_df["PRIMARY_PREDICTED_SPB"]
    raw_ratio = raw_ratio.clip(lower=MULTIPLIER_MIN, upper=MULTIPLIER_MAX)
    train_df["VIRAL_MULTIPLIER"] = np.log(raw_ratio)
    return train_df.reset_index(drop=True)


def build_chartex_regressor() -> LGBMRegressor:
    """Highly regularized Huber regressor for small-sample social residual correction."""
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


def train_chartex_multiplier(
    train_df: pd.DataFrame,
) -> tuple[LGBMRegressor, dict[str, float | bool], dict[str, float | bool]]:
    """Train Stage 4 on gated rows; evaluate holdout health + wMAPE on that population."""
    if train_df.empty:
        raise ValueError("Training frame is empty after Stage 4 eligibility gating.")

    X = train_df[SOCIAL_FEATURES]
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

    health = assess_stage4_multiplier_health(predicted_multiplier)

    # Primary evaluation: eligible holdout (routed-or-social gated frame).
    metrics: dict[str, float | bool] = {
        "holdout_rows": float(len(val_df)),
        "multiplier_mae": float(mean_absolute_error(actual_multiplier, predicted_multiplier)),
        "baseline_wmape": weighted_mape(actual_spb, primary_spb),
        "cascaded_wmape": weighted_mape(actual_spb, cascaded_spb),
        "median_multiplier": float(health["median_multiplier"]),
        "p95_multiplier": float(health["p95_multiplier"]),
        "mean_multiplier": float(health["mean_multiplier"]),
        "stage4_enabled": bool(health["enabled"]),
    }

    # Secondary: viral-candidate subset within holdout.
    viral_mask = val_df["IS_VIRAL_SOCIAL"].to_numpy(dtype=bool)
    if viral_mask.any():
        metrics["viral_holdout_rows"] = float(viral_mask.sum())
        metrics["viral_baseline_wmape"] = weighted_mape(
            actual_spb[viral_mask], primary_spb[viral_mask]
        )
        metrics["viral_cascaded_wmape"] = weighted_mape(
            actual_spb[viral_mask], cascaded_spb[viral_mask]
        )
        metrics["viral_median_multiplier"] = float(np.median(predicted_multiplier[viral_mask]))
    else:
        metrics["viral_holdout_rows"] = 0.0
        metrics["viral_baseline_wmape"] = float("nan")
        metrics["viral_cascaded_wmape"] = float("nan")
        metrics["viral_median_multiplier"] = float("nan")

    # Standard-only slice (production apply population).
    standard_mask = val_df["PRIMARY_PREDICTION_STAGE"].to_numpy() == STAGE_LABEL_STANDARD
    if standard_mask.any():
        metrics["standard_holdout_rows"] = float(standard_mask.sum())
        metrics["standard_baseline_wmape"] = weighted_mape(
            actual_spb[standard_mask], primary_spb[standard_mask]
        )
        metrics["standard_cascaded_wmape"] = weighted_mape(
            actual_spb[standard_mask], cascaded_spb[standard_mask]
        )
    else:
        metrics["standard_holdout_rows"] = 0.0
        metrics["standard_baseline_wmape"] = float("nan")
        metrics["standard_cascaded_wmape"] = float("nan")

    return model, metrics, health


def export_model(
    model: LGBMRegressor,
    metrics: dict[str, float | bool],
    health: dict[str, float | bool],
) -> None:
    """Persist Stage 4 regressor, gates, and enablement flag for inference."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "regressor": model,
        "chartex_features": SOCIAL_FEATURES,
        "social_features": SOCIAL_FEATURES,
        "multiplier_min": MULTIPLIER_MIN,
        "multiplier_max": MULTIPLIER_MAX,
        "target_transform": "log",
        "cohort_start_date": COHORT_START_DATE,
        "holdout_metrics": metrics,
        "stage4_enabled": bool(health["enabled"]),
        "health": health,
        "real_social_min_views": REAL_SOCIAL_MIN_VIEWS,
        "viral_terminal_acceleration_threshold": VIRAL_TERMINAL_ACCELERATION_THRESHOLD,
        "application_policy": {
            "allowed_stages": [STAGE_LABEL_STANDARD],
            "require_real_social": True,
            "exclude_superstar": True,
            "exclude_no_social": True,
            "exclude_gatekeeper": True,
        },
    }
    joblib.dump(payload, OUTPUT_MODEL_PATH)


def print_summary(
    cohort_df: pd.DataFrame,
    train_df: pd.DataFrame,
    metrics: dict[str, float | bool],
    health: dict[str, float | bool],
) -> None:
    """Print gated training / holdout evaluation summary."""
    stage_col = "PRIMARY_PREDICTION_STAGE"
    routed_mask = cohort_df[stage_col] != STAGE_LABEL_GATEKEEPER
    print("\n" + "=" * 72)
    print("STAGE 4 — DAP SOCIAL VIRAL MULTIPLIER (GATED)")
    print("=" * 72)
    print(f"Cohort start date:              {COHORT_START_DATE}")
    print(f"2026 cohort rows:               {len(cohort_df):,}")
    print(f"Hurdle-routed rows:             {int(routed_mask.sum()):,}")
    print(f"Gatekeeper rows:                {int((~routed_mask).sum()):,}")
    print(f"Eligible Stage 4 train rows:    {len(train_df):,}")
    print(f"  Standard + social:            {int((train_df[stage_col]==STAGE_LABEL_STANDARD).sum()):,}")
    print(f"  Gatekeeper + social:          {int((train_df[stage_col]==STAGE_LABEL_GATEKEEPER).sum()):,}")
    print(f"  Viral candidates in train:    {int(train_df['IS_VIRAL_SOCIAL'].sum()):,}")
    print(f"Real social min views:          {REAL_SOCIAL_MIN_VIEWS:,.0f}")
    print(f"Viral accel threshold:          {VIRAL_TERMINAL_ACCELERATION_THRESHOLD}")
    print("-" * 72)
    print("Holdout Evaluation (eligible population only)")
    print(f"  Holdout rows:                 {int(metrics['holdout_rows']):,}")
    print(f"  Multiplier MAE:               {float(metrics['multiplier_mae']):.4f}")
    print(f"  Median multiplier:            {float(metrics['median_multiplier']):.3f}x")
    print(f"  P95 multiplier:               {float(metrics['p95_multiplier']):.3f}x")
    print(f"  Baseline wMAPE (SPB):         {float(metrics['baseline_wmape']):.2%}")
    print(f"  Cascaded wMAPE (SPB):         {float(metrics['cascaded_wmape']):.2%}")
    print(
        f"  wMAPE improvement:            "
        f"{(float(metrics['baseline_wmape']) - float(metrics['cascaded_wmape'])):.2%} absolute"
    )
    if float(metrics["standard_holdout_rows"]) > 0:
        print(
            f"  Standard-only cascaded Δ:     "
            f"{(float(metrics['standard_baseline_wmape']) - float(metrics['standard_cascaded_wmape'])):.2%} "
            f"(n={int(metrics['standard_holdout_rows'])})"
        )
    if float(metrics["viral_holdout_rows"]) > 0:
        print(
            f"  Viral-candidate cascaded Δ:   "
            f"{(float(metrics['viral_baseline_wmape']) - float(metrics['viral_cascaded_wmape'])):.2%} "
            f"(n={int(metrics['viral_holdout_rows'])})"
        )
    print("-" * 72)
    enabled = bool(health["enabled"])
    print(
        f"Stage 4 production enablement:  "
        f"{'ENABLED' if enabled else 'DISABLED'} "
        f"(need median in [0.85, 1.15] and p95 >= 1.50)"
    )
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
    model, metrics, health = train_chartex_multiplier(train_df)
    export_model(model, metrics, health)
    print_summary(cohort_df, train_df, metrics, health)


if __name__ == "__main__":
    main()
