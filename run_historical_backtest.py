"""Out-of-Time backtest for the Three-Stage LightGBM Hurdle architecture."""

from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit

from run_dual_mode_inference import (
    GATEKEEPER_PROB_THRESHOLD,
    PEER_BENCHMARK_FALLBACK_SPB,
    extract_canonical_genre,
)
from train_streaming_hurdle import (
    EXCLUDED_FROM_TRAINING,
    FEATURE_COLUMNS,
    GATEKEEPER_PERCENTILE,
    STAGE3_FEATURE_COLUMNS,
    STAGE_LABEL_GATEKEEPER,
    STAGE_LABEL_STANDARD,
    STAGE_LABEL_SUPERSTAR,
    SUPERSTAR_ROUTER_RATIO,
    OPTUNA_N_TRIALS,
    TARGET_COLUMN,
    TIME_WEIGHT_HALF_LIFE_DAYS,
    TIME_WEIGHT_MIN,
    build_classifier,
    calculate_dynamic_threshold,
    calculate_superstar_threshold,
    calculate_time_weights,
    create_tuning_holdout_split,
    engineer_stage3_features,
    gatekeeper_predict_proba,
    optimize_stage2,
    optimize_stage3,
    train_final_regressor,
    train_final_superstar_regressor,
)

DATA_DIR = Path("data")
DEFAULT_INPUT_PATH = DATA_DIR / "album_meta_features.parquet"
DEFAULT_OUTPUT_PATH = DATA_DIR / "oot_backtest_results.csv"
PEER_BENCHMARKS_PATH = DATA_DIR / "median_spb.csv"

ACTUAL_STREAMS_COLUMN = "ALBUM_TOTAL_W1_STREAMS"
SPB_DIVISOR = 1_000_000_000
HYPE_CONVERSION_DIVISOR = 2.01
RAW_STREAM_FLOOR = 500.0


def parse_args() -> argparse.Namespace:
    default_cutoff = (pd.Timestamp.today().normalize() - pd.DateOffset(months=6)).strftime(
        "%Y-%m-%d"
    )
    parser = argparse.ArgumentParser(
        description="Run an Out-of-Time backtest of the streaming hurdle model."
    )
    parser.add_argument(
        "--cutoff-date",
        type=str,
        default=default_cutoff,
        help="Temporal split date (YYYY-MM-DD); train < cutoff, test >= cutoff",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Parquet feature matrix used for the backtest",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="CSV path for detailed holdout predictions",
    )
    return parser.parse_args()


def validate_feature_leakage() -> None:
    """Ensure excluded leakage columns are not used as model features."""
    leaked_features = set(FEATURE_COLUMNS) & set(EXCLUDED_FROM_TRAINING)
    if leaked_features:
        raise ValueError(
            f"Feature columns overlap with excluded training columns: {sorted(leaked_features)}"
        )

    required_exclusions = {"ALBUM_STANDARD_TRACK_SPB", "AVG_TRACK_W1_AUDIO_STREAMS"}
    missing_exclusions = required_exclusions - set(EXCLUDED_FROM_TRAINING)
    if missing_exclusions:
        raise ValueError(
            f"EXCLUDED_FROM_TRAINING is missing leakage guards: {sorted(missing_exclusions)}"
        )


def sanitize_prediction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Defensively replace infinite SPB/feature values before model inference."""
    sanitized = df.copy()
    spb_columns = [
        col
        for col in sanitized.columns
        if "SPB" in col
        or col in FEATURE_COLUMNS
        or col in STAGE3_FEATURE_COLUMNS
        or col in EXCLUDED_FROM_TRAINING
    ]
    present_columns = [col for col in spb_columns if col in sanitized.columns]
    if present_columns:
        sanitized[present_columns] = sanitized[present_columns].replace(
            [np.inf, -np.inf], np.nan
        )
    return sanitized


def load_and_prepare_data(path: Path) -> pd.DataFrame:
    """Load, clean, and sanitize the feature matrix for backtesting."""
    validate_feature_leakage()
    print(f"\nLoading data from {path}...")
    df = pd.read_parquet(path)
    raw_rows = len(df)

    required_columns = {
        TARGET_COLUMN,
        ACTUAL_STREAMS_COLUMN,
        "TOTAL_UNIVERSE_STREAMS",
        "MARKET_WEEK_START",
        "DISPLAY_ARTIST",
        "GENRE",
        "LEVEL_2_DISTRIBUTOR",
        *FEATURE_COLUMNS,
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Input data is missing required columns: {sorted(missing_columns)}")

    df["FIRST_SALE_DATE"] = pd.to_datetime(df["FIRST_SALE_DATE"])
    clean_df = df.dropna(subset=[TARGET_COLUMN, ACTUAL_STREAMS_COLUMN, "DISPLAY_ARTIST"]).copy()
    clean_df = clean_df[clean_df[TARGET_COLUMN] > 0]
    clean_df = clean_df[clean_df[ACTUAL_STREAMS_COLUMN] > 0]
    clean_df = clean_df[clean_df["TOTAL_TRACKS_ANALYZED"] >= 2]

    clean_df = engineer_stage3_features(clean_df)
    clean_df = sanitize_prediction_features(clean_df)

    print(f"  Raw rows:   {raw_rows:,}")
    print(f"  Clean rows: {len(clean_df):,}")
    print(f"  Model target: {TARGET_COLUMN}")
    print(f"  Evaluation actuals: {ACTUAL_STREAMS_COLUMN}")
    return clean_df


def temporal_split(
    df: pd.DataFrame,
    cutoff_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset into pre-cutoff training and post-cutoff holdout sets."""
    cutoff = pd.Timestamp(cutoff_date)
    train_df = df[df["FIRST_SALE_DATE"] < cutoff].copy()
    test_df = df[df["FIRST_SALE_DATE"] >= cutoff].copy()

    print("\n" + "=" * 72)
    print("TEMPORAL SPLIT")
    print("=" * 72)
    print(f"Cutoff date: {cutoff.date()}")
    print(
        f"Train set: {len(train_df):,} rows | "
        f"{train_df['FIRST_SALE_DATE'].min().date()} to "
        f"{train_df['FIRST_SALE_DATE'].max().date()}"
        if not train_df.empty
        else f"Train set: {len(train_df):,} rows | (empty)"
    )
    print(
        f"Test set:  {len(test_df):,} rows | "
        f"{test_df['FIRST_SALE_DATE'].min().date()} to "
        f"{test_df['FIRST_SALE_DATE'].max().date()}"
        if not test_df.empty
        else f"Test set:  {len(test_df):,} rows | (empty)"
    )
    print("=" * 72)

    if train_df.empty:
        raise ValueError("Training set is empty. Choose an earlier cutoff date.")
    if test_df.empty:
        raise ValueError("Holdout set is empty. Choose a later cutoff date.")

    return train_df, test_df


def decode_spb_to_streams(spb: np.ndarray, market_size: np.ndarray) -> np.ndarray:
    """Decode Streams Per Billion into raw weekly stream volume."""
    valid = (~np.isnan(spb)) & (~np.isnan(market_size)) & (market_size > 0)
    decoded = np.full_like(spb, np.nan, dtype=float)
    decoded[valid] = (spb[valid] / SPB_DIVISOR) * market_size[valid]
    return decoded


def build_lagged_market_size_lookup(full_df: pd.DataFrame) -> pd.DataFrame:
    """Build weekly universe size lookup using the prior week's known market total."""
    weeks = full_df.dropna(subset=["MARKET_WEEK_START", "TOTAL_UNIVERSE_STREAMS"]).copy()
    weeks["MARKET_WEEK_START"] = pd.to_datetime(weeks["MARKET_WEEK_START"], errors="coerce")

    weekly = (
        weeks.dropna(subset=["MARKET_WEEK_START"])
        .groupby("MARKET_WEEK_START", as_index=False)["TOTAL_UNIVERSE_STREAMS"]
        .median()
        .sort_values("MARKET_WEEK_START")
    )
    weekly["ESTIMATED_MARKET_SIZE"] = weekly["TOTAL_UNIVERSE_STREAMS"].shift(1)
    weekly["ESTIMATED_MARKET_SIZE"] = weekly["ESTIMATED_MARKET_SIZE"].bfill()
    return weekly[["MARKET_WEEK_START", "ESTIMATED_MARKET_SIZE"]]


def estimate_market_sizes(full_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """Map each holdout row to lagged universe size via week dictionary (preserves row order)."""
    lagged_lookup = build_lagged_market_size_lookup(full_df)
    week_to_market_size = dict(
        zip(lagged_lookup["MARKET_WEEK_START"], lagged_lookup["ESTIMATED_MARKET_SIZE"])
    )

    market_weeks = pd.to_datetime(test_df["MARKET_WEEK_START"], errors="coerce")
    mapped_series = market_weeks.map(week_to_market_size)
    mapped_series = mapped_series.ffill().bfill()
    return mapped_series.to_numpy(dtype=float)


def map_peer_spb_array(
    genre: np.ndarray,
    distributor: np.ndarray,
    peer_benchmarks: pd.DataFrame,
) -> np.ndarray:
    """Vectorized cascading peer SPB lookup: exact -> distributor -> genre -> fallback."""
    if peer_benchmarks.empty:
        return np.full(len(genre), PEER_BENCHMARK_FALLBACK_SPB, dtype=float)

    lookup_df = pd.DataFrame(
        {
            "GENRE": genre,
            "LEVEL_2_DISTRIBUTOR": [
                str(d) if pd.notna(d) else "Unknown" for d in distributor
            ],
        }
    )

    exact = lookup_df.merge(
        peer_benchmarks[["GENRE", "LEVEL_2_DISTRIBUTOR", "MEDIAN_SPB"]],
        on=["GENRE", "LEVEL_2_DISTRIBUTOR"],
        how="left",
    )["MEDIAN_SPB"]

    dist_medians = (
        peer_benchmarks.groupby("LEVEL_2_DISTRIBUTOR")["MEDIAN_SPB"]
        .median()
        .rename("DIST_MEDIAN")
    )
    lookup_df = lookup_df.join(dist_medians, on="LEVEL_2_DISTRIBUTOR")

    genre_medians = (
        peer_benchmarks.groupby("GENRE")["MEDIAN_SPB"]
        .median()
        .rename("GENRE_MEDIAN")
    )
    lookup_df = lookup_df.join(genre_medians, on="GENRE")

    return np.where(
        pd.notna(exact),
        exact,
        np.where(
            pd.notna(lookup_df["DIST_MEDIAN"]),
            lookup_df["DIST_MEDIAN"],
            np.where(
                pd.notna(lookup_df["GENRE_MEDIAN"]),
                lookup_df["GENRE_MEDIAN"],
                PEER_BENCHMARK_FALLBACK_SPB,
            ),
        ),
    )


def standard_tier_forecast(
    tracks: np.ndarray,
    historical_spb: np.ndarray,
    lead_peak: np.ndarray,
    genre: np.ndarray,
    distributor: np.ndarray,
    peer_benchmarks: pd.DataFrame,
    spike_ratio: np.ndarray,
    estimated_market_size: np.ndarray,
) -> np.ndarray:
    """3-level vectorized standard-tier forecast decoded with lagged market size estimates."""
    tracks = np.where(np.isnan(tracks) | (tracks <= 0), 12.0, tracks).astype(float)
    historical_spb = historical_spb.astype(float)
    lead_peak = np.nan_to_num(lead_peak, nan=0.0).astype(float)
    spike_ratio = np.clip(np.nan_to_num(spike_ratio, nan=1.0), 0.5, 3.0).astype(float)
    estimated_market_size = estimated_market_size.astype(float)

    effective_tracks = 12.0 * (np.log(tracks + 1.0) / np.log(13.0))
    historical_raw_streams = decode_spb_to_streams(historical_spb, estimated_market_size)

    veteran_raw_streams = effective_tracks * historical_raw_streams * spike_ratio
    implied_standard_raw_streams = lead_peak / HYPE_CONVERSION_DIVISOR
    cold_start_raw_streams = effective_tracks * implied_standard_raw_streams

    peer_spb_array = map_peer_spb_array(genre, distributor, peer_benchmarks)
    peer_album_spb = effective_tracks * peer_spb_array
    peer_raw_streams = decode_spb_to_streams(peer_album_spb, estimated_market_size)

    has_history = (~np.isnan(historical_spb)) & (historical_spb > 0)
    has_lead = lead_peak > 0

    predicted_raw_streams = np.where(
        has_history,
        veteran_raw_streams,
        np.where(has_lead, cold_start_raw_streams, peer_raw_streams),
    )
    return np.maximum(predicted_raw_streams, RAW_STREAM_FLOOR)


def train_backtest_models(
    train_df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
) -> tuple[LGBMClassifier, LGBMRegressor, LGBMRegressor, float, float]:
    """Train fresh three-stage models with time-decay weighting and Optuna tuning."""
    print("\n" + "=" * 72)
    print("TRAINING BACKTEST MODELS (train_df only)")
    print("=" * 72)

    train_weights_all = calculate_time_weights(train_df["FIRST_SALE_DATE"], cutoff_date)
    print(
        f"Time-decay weights: half-life={TIME_WEIGHT_HALF_LIFE_DAYS}d, "
        f"min={TIME_WEIGHT_MIN}, range={train_weights_all.min():.3f}–{train_weights_all.max():.3f}"
    )

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    groups_train = train_df["DISPLAY_ARTIST"]

    dynamic_threshold = calculate_dynamic_threshold(y_train, GATEKEEPER_PERCENTILE)
    superstar_threshold = calculate_superstar_threshold(y_train)
    y_binary = (y_train >= dynamic_threshold).astype(int)
    positive_rate = y_binary.mean()
    print(
        f"Stage 1 positive class rate: {positive_rate:.2%} "
        f"({int(y_binary.sum()):,} / {len(y_binary):,})"
    )

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx1, val_idx1 = next(gss1.split(X_train, y_binary, groups_train))

    X_tr1, X_val1 = X_train.iloc[train_idx1], X_train.iloc[val_idx1]
    y_tr1, y_val1 = y_binary.iloc[train_idx1], y_binary.iloc[val_idx1]
    w_tr1 = train_weights_all[train_idx1]
    w_val1 = train_weights_all[val_idx1]

    classifier = build_classifier()
    print("\nFitting Stage 1 gatekeeper classifier (time-decay weighted)...")
    classifier.fit(
        X_tr1,
        y_tr1,
        sample_weight=w_tr1,
        eval_set=[(X_val1, y_val1)],
        eval_sample_weight=[w_val1],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    print(f"  Classifier trained (best iteration: {classifier.best_iteration_})")

    high_tier_mask = train_df[TARGET_COLUMN] >= dynamic_threshold
    high_tier_train = train_df[high_tier_mask].copy()
    high_tier_weights = train_weights_all[high_tier_mask.to_numpy()]

    X_high = high_tier_train[FEATURE_COLUMNS]
    y_high = high_tier_train[TARGET_COLUMN]
    groups_high = high_tier_train["DISPLAY_ARTIST"]

    stage2_train_idx, stage2_val_idx = create_tuning_holdout_split(
        X_high, y_high, groups_high,
    )

    print(f"\nOptuna tuning Stage 2 Tweedie Regressor ({OPTUNA_N_TRIALS} trials)...")
    stage2_best_params = optimize_stage2(
        X_high.iloc[stage2_train_idx],
        X_high.iloc[stage2_val_idx],
        y_high.iloc[stage2_train_idx],
        y_high.iloc[stage2_val_idx],
        high_tier_weights[stage2_train_idx],
        high_tier_weights[stage2_val_idx],
    )
    print(f"  Stage 2 best params: {stage2_best_params}")

    regressor = train_final_regressor(
        X=X_high,
        y_target=y_high,
        sample_weights=high_tier_weights,
        stage2_params=stage2_best_params,
        X_val=X_high.iloc[stage2_val_idx],
        y_val=y_high.iloc[stage2_val_idx],
        val_weights=high_tier_weights[stage2_val_idx],
    )

    superstar_mask = train_df[TARGET_COLUMN] >= superstar_threshold
    superstar_train = train_df[superstar_mask].copy()
    if len(superstar_train) < 2:
        raise ValueError(
            f"Superstar training subset has only {len(superstar_train):,} rows; "
            "cannot train Stage 3 regressor."
        )

    superstar_weights = train_weights_all[superstar_mask.to_numpy()]
    X_superstar = superstar_train[STAGE3_FEATURE_COLUMNS]
    y_superstar = superstar_train[TARGET_COLUMN]
    groups_superstar = superstar_train["DISPLAY_ARTIST"]

    stage3_train_idx, stage3_val_idx = create_tuning_holdout_split(
        X_superstar, y_superstar, groups_superstar,
    )

    print(f"\nOptuna tuning Stage 3 Superstar Regressor ({OPTUNA_N_TRIALS} trials)...")
    stage3_best_params = optimize_stage3(
        X_superstar.iloc[stage3_train_idx],
        X_superstar.iloc[stage3_val_idx],
        y_superstar.iloc[stage3_train_idx],
        y_superstar.iloc[stage3_val_idx],
        superstar_weights[stage3_train_idx],
        superstar_weights[stage3_val_idx],
    )
    print(f"  Stage 3 best params: {stage3_best_params}")

    superstar_regressor = train_final_superstar_regressor(
        X=X_superstar,
        y_target=y_superstar,
        sample_weights=superstar_weights,
        stage3_params=stage3_best_params,
        X_val=X_superstar.iloc[stage3_val_idx],
        y_val_log=np.log1p(y_superstar.iloc[stage3_val_idx]),
        val_weights=superstar_weights[stage3_val_idx],
    )

    return classifier, regressor, superstar_regressor, dynamic_threshold, superstar_threshold


def run_backtest_inference(
    test_df: pd.DataFrame,
    full_df: pd.DataFrame,
    classifier: LGBMClassifier,
    regressor: LGBMRegressor,
    superstar_regressor: LGBMRegressor,
    superstar_threshold: float,
) -> pd.DataFrame:
    """Generate holdout forecasts using the three-stage predictive router."""
    print("\n" + "=" * 72)
    print("HOLDOUT INFERENCE (test_df only — SPB targets withheld)")
    print("=" * 72)

    results = sanitize_prediction_features(test_df.copy())
    X_test = results[FEATURE_COLUMNS]
    estimated_market_size = estimate_market_sizes(full_df, results)

    peer_benchmarks = pd.read_csv(PEER_BENCHMARKS_PATH)
    peer_benchmarks.columns = peer_benchmarks.columns.str.upper()
    results["GENRE"] = results["GENRE"].apply(extract_canonical_genre)

    gatekeeper_probs = gatekeeper_predict_proba(classifier, X_test)
    is_high_tier = gatekeeper_probs >= GATEKEEPER_PROB_THRESHOLD

    predicted_spb = np.full(len(results), np.nan, dtype=float)
    predicted_raw_streams = np.zeros(len(results), dtype=float)
    prediction_stage = np.full(len(results), STAGE_LABEL_GATEKEEPER, dtype=object)

    gatekeeper_mask = ~is_high_tier
    high_tier_mask = is_high_tier

    if high_tier_mask.any():
        stage2_preds = regressor.predict(X_test.loc[high_tier_mask, FEATURE_COLUMNS])
        predicted_spb[high_tier_mask] = stage2_preds
        prediction_stage[high_tier_mask] = STAGE_LABEL_STANDARD

        router_threshold = superstar_threshold * SUPERSTAR_ROUTER_RATIO
        high_tier_indices = np.where(high_tier_mask)[0]
        route_local_mask = stage2_preds >= router_threshold

        if route_local_mask.any():
            route_global_mask = np.zeros(len(results), dtype=bool)
            route_global_mask[high_tier_indices[route_local_mask]] = True

            stage3_log_preds = superstar_regressor.predict(
                results.loc[route_global_mask, STAGE3_FEATURE_COLUMNS]
            )
            predicted_spb[route_global_mask] = np.expm1(stage3_log_preds)
            prediction_stage[route_global_mask] = STAGE_LABEL_SUPERSTAR

        decoded_streams = decode_spb_to_streams(
            predicted_spb[high_tier_mask],
            estimated_market_size[high_tier_mask],
        )
        predicted_raw_streams[high_tier_mask] = np.where(
            np.isnan(decoded_streams),
            RAW_STREAM_FLOOR,
            decoded_streams,
        )

    if gatekeeper_mask.any():
        standard_rows = results.loc[gatekeeper_mask]
        genre_array = standard_rows["GENRE"].to_numpy(dtype=object)
        distributor_array = standard_rows["LEVEL_2_DISTRIBUTOR"].to_numpy(dtype=object)
        spike_array = standard_rows["SHORT_TERM_SPIKE_RATIO"].to_numpy(dtype=float)
        predicted_raw_streams[gatekeeper_mask] = standard_tier_forecast(
            tracks=standard_rows["TOTAL_TRACKS_ANALYZED"].to_numpy(dtype=float),
            historical_spb=standard_rows["HISTORICAL_STANDARD_TRACK_SPB"].to_numpy(dtype=float),
            lead_peak=standard_rows["LEAD_SINGLE_PEAK_VOLUME"].to_numpy(dtype=float),
            genre=genre_array,
            distributor=distributor_array,
            peer_benchmarks=peer_benchmarks,
            spike_ratio=spike_array,
            estimated_market_size=estimated_market_size[gatekeeper_mask],
        )

    results["ESTIMATED_MARKET_SIZE"] = estimated_market_size
    results["PREDICTED_W1_SPB"] = predicted_spb
    results["GATEKEEPER_PROBABILITY"] = gatekeeper_probs
    results["PREDICTED_IS_HIGH_TIER"] = is_high_tier
    results["PREDICTION_STAGE"] = prediction_stage
    results["PREDICTED_W1_STREAMS"] = predicted_raw_streams

    superstar_count = int((prediction_stage == STAGE_LABEL_SUPERSTAR).sum())
    standard_count = int((prediction_stage == STAGE_LABEL_STANDARD).sum())
    gatekeeper_count = int((prediction_stage == STAGE_LABEL_GATEKEEPER).sum())

    print(f"  Scored {len(results):,} holdout albums")
    print(f"  Lagged market size range: {estimated_market_size.min():,.0f} – {estimated_market_size.max():,.0f}")
    print(f"  Routed Superstar:  {superstar_count:,}")
    print(f"  Routed Standard:   {standard_count:,}")
    print(f"  Gatekeeper Reject: {gatekeeper_count:,}")
    return results


def format_pct(value: float) -> str:
    return f"{value:.1%}"


def format_number(value: float) -> str:
    if np.isnan(value):
        return "N/A"
    return f"{value:,.2f}"


def weighted_mape(actual: pd.Series, predicted: pd.Series) -> float:
    """Volume-weighted MAPE: sum(|error|) / sum(actual)."""
    actual_sum = float(actual.sum())
    if actual_sum == 0:
        return np.nan
    return float(np.abs(actual - predicted).sum() / actual_sum)


def stage_stream_metrics(tier_df: pd.DataFrame) -> dict[str, float]:
    """Compute raw-stream accuracy metrics for a prediction-stage subset."""
    if tier_df.empty:
        return {"wmape": np.nan, "mae": np.nan, "rmse": np.nan}

    tier_actual = tier_df[ACTUAL_STREAMS_COLUMN].astype(float)
    tier_predicted = tier_df["PREDICTED_W1_STREAMS"].astype(float)
    return {
        "wmape": weighted_mape(tier_actual, tier_predicted),
        "mae": mean_absolute_error(tier_actual, tier_predicted),
        "rmse": float(np.sqrt(mean_squared_error(tier_actual, tier_predicted))),
    }


def evaluate_holdout(
    results: pd.DataFrame,
    dynamic_threshold: float,
) -> pd.DataFrame:
    """Score decoded raw-stream predictions against actual album streams by routing stage."""
    actual_spb = results[TARGET_COLUMN].astype(float)
    actual_streams = results[ACTUAL_STREAMS_COLUMN].astype(float)
    predicted_streams = results["PREDICTED_W1_STREAMS"].astype(float)

    actual_is_high_tier = (actual_spb >= dynamic_threshold).astype(int)
    predicted_is_high_tier = results["PREDICTED_IS_HIGH_TIER"].astype(int)

    results["ACTUAL_IS_HIGH_TIER"] = actual_is_high_tier.astype(bool)
    results["ABSOLUTE_ERROR"] = np.abs(actual_streams - predicted_streams)
    results["APE"] = results["ABSOLUTE_ERROR"] / actual_streams

    precision = precision_score(actual_is_high_tier, predicted_is_high_tier, zero_division=0)
    recall = recall_score(actual_is_high_tier, predicted_is_high_tier, zero_division=0)
    f1 = f1_score(actual_is_high_tier, predicted_is_high_tier, zero_division=0)

    overall_wmape = weighted_mape(actual_streams, predicted_streams)
    overall_mae = mean_absolute_error(actual_streams, predicted_streams)
    overall_rmse = float(np.sqrt(mean_squared_error(actual_streams, predicted_streams)))

    superstar_results = results[results["PREDICTION_STAGE"] == STAGE_LABEL_SUPERSTAR]
    standard_results = results[results["PREDICTION_STAGE"] == STAGE_LABEL_STANDARD]
    gatekeeper_results = results[results["PREDICTION_STAGE"] == STAGE_LABEL_GATEKEEPER]

    superstar_metrics = stage_stream_metrics(superstar_results)
    standard_metrics = stage_stream_metrics(standard_results)
    gatekeeper_metrics = stage_stream_metrics(gatekeeper_results)

    print("\n" + "=" * 72)
    print("OUT-OF-TIME BACKTEST — EXECUTIVE SUMMARY")
    print("=" * 72)
    print(f"Holdout albums evaluated:       {len(results):,}")
    print(f"Training-derived SPB threshold: {dynamic_threshold:,.6f}")
    print(f"Actual high-tier rate:          {actual_is_high_tier.mean():.2%}")
    print(f"Predicted high-tier rate:       {predicted_is_high_tier.mean():.2%}")
    print("-" * 72)
    print("Gatekeeper Classification (Stage 1)")
    print(f"  Precision:                    {format_pct(precision)}")
    print(f"  Recall:                       {format_pct(recall)}")
    print(f"  F1 Score:                     {format_pct(f1)}")
    print("-" * 72)
    print("Raw Stream Volume Accuracy (All Holdout Albums)")
    print(f"  Overall wMAPE:                {format_pct(overall_wmape)}")
    print(f"  MAE:                          {format_number(overall_mae)} streams")
    print(f"  RMSE:                         {format_number(overall_rmse)} streams")
    print("-" * 72)
    print("Raw Stream Volume Accuracy by Prediction Stage")
    print(
        f"  Superstar Regressor wMAPE:    {format_pct(superstar_metrics['wmape'])} "
        f"({len(superstar_results):,} albums)"
        if not np.isnan(superstar_metrics["wmape"])
        else "  Superstar Regressor wMAPE:    N/A"
    )
    print(
        f"  Superstar Regressor MAE:      {format_number(superstar_metrics['mae'])} streams"
        if not np.isnan(superstar_metrics["mae"])
        else "  Superstar Regressor MAE:      N/A"
    )
    print(
        f"  Superstar Regressor RMSE:     {format_number(superstar_metrics['rmse'])} streams"
        if not np.isnan(superstar_metrics["rmse"])
        else "  Superstar Regressor RMSE:     N/A"
    )
    print(
        f"  Standard Regressor wMAPE:     {format_pct(standard_metrics['wmape'])} "
        f"({len(standard_results):,} albums)"
        if not np.isnan(standard_metrics["wmape"])
        else "  Standard Regressor wMAPE:     N/A"
    )
    print(
        f"  Standard Regressor MAE:       {format_number(standard_metrics['mae'])} streams"
        if not np.isnan(standard_metrics["mae"])
        else "  Standard Regressor MAE:       N/A"
    )
    print(
        f"  Standard Regressor RMSE:      {format_number(standard_metrics['rmse'])} streams"
        if not np.isnan(standard_metrics["rmse"])
        else "  Standard Regressor RMSE:      N/A"
    )
    print(
        f"  Gatekeeper Fallback wMAPE:    {format_pct(gatekeeper_metrics['wmape'])} "
        f"({len(gatekeeper_results):,} albums)"
        if not np.isnan(gatekeeper_metrics["wmape"])
        else "  Gatekeeper Fallback wMAPE:    N/A"
    )
    print(
        f"  Gatekeeper Fallback MAE:      {format_number(gatekeeper_metrics['mae'])} streams"
        if not np.isnan(gatekeeper_metrics["mae"])
        else "  Gatekeeper Fallback MAE:      N/A"
    )
    print(
        f"  Gatekeeper Fallback RMSE:     {format_number(gatekeeper_metrics['rmse'])} streams"
        if not np.isnan(gatekeeper_metrics["rmse"])
        else "  Gatekeeper Fallback RMSE:     N/A"
    )
    print("=" * 72)

    return results


def main() -> None:
    args = parse_args()

    df = load_and_prepare_data(args.input_file)
    train_df, test_df = temporal_split(df, args.cutoff_date)
    cutoff = pd.Timestamp(args.cutoff_date)
    classifier, regressor, superstar_regressor, dynamic_threshold, superstar_threshold = (
        train_backtest_models(train_df, cutoff)
    )
    results = run_backtest_inference(
        test_df,
        df,
        classifier,
        regressor,
        superstar_regressor,
        superstar_threshold,
    )
    results = evaluate_holdout(results, dynamic_threshold)

    print(f"\nExporting holdout results to {args.output_file}...")
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_file, index=False)
    print("Backtest complete.")


if __name__ == "__main__":
    main()
