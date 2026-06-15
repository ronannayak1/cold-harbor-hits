"""Dual-mode inference engine for Week 1 album stream forecasts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from build_social_features import (
    extract_social_features,
    filter_pre_release_entries,
    format_release_datetime,
    parse_timeline_json,
)
from train_streaming_hurdle import (
    FEATURE_COLUMNS,
    SOCIAL_RATIO_COLUMNS,
    SOCIAL_VOLUME_COLUMNS,
    STAGE3_FEATURE_COLUMNS,
    STAGE_LABEL_GATEKEEPER,
    STAGE_LABEL_STANDARD,
    STAGE_LABEL_SUPERSTAR,
    engineer_stage3_features,
    gatekeeper_predict_proba,
    predict_pipeline,
)

DATA_DIR = Path("data")
MODELS_DIR = Path("models")

SANDBOX_BASELINES_PATH = DATA_DIR / "artist_sandbox_baselines.csv"
PEER_BENCHMARKS_PATH = DATA_DIR / "median_spb.csv"
FEATURE_MATRIX_PATH = DATA_DIR / "album_meta_features.parquet"
CLASSIFIER_PATH = MODELS_DIR / "streaming_hurdle_classifier.txt"
REGRESSOR_PATH = MODELS_DIR / "streaming_hurdle_regressor.joblib"
SUPERSTAR_REGRESSOR_PATH = MODELS_DIR / "streaming_superstar_regressor.joblib"
META_PATH = MODELS_DIR / "streaming_hurdle_meta.json"

ALBUM_SKEW_MULTIPLIER = 3.78
HYPE_CONVERSION_DIVISOR = 2.01
PEER_BENCHMARK_FALLBACK_SPB = 15.0
COLD_START_BASELINE_STREAMS = 50_000.0
SPB_DIVISOR = 1_000_000_000
GATEKEEPER_PROB_THRESHOLD = 0.5 #default that should get replaced by optimal from metadata
RAW_STREAM_FLOOR = 500.0
DEFAULT_TRACKS = 12
DEFAULT_HITS = 1

_peer_benchmarks_cache: pd.DataFrame | None = None

STAGE_DISPLAY_LABELS = {
    STAGE_LABEL_GATEKEEPER: "Gatekeeper Rejected",
    STAGE_LABEL_STANDARD: "Standard Regressor",
    STAGE_LABEL_SUPERSTAR: "Superstar Regressor",
}


class BoosterClassifierAdapter:
    """Sklearn-compatible wrapper for a serialized LightGBM Booster classifier."""

    def __init__(self, booster: lgb.Booster):
        self._booster = booster

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        positive_prob = np.asarray(self._booster.predict(X), dtype=float)
        return np.column_stack([1.0 - positive_prob, positive_prob])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict Week 1 album streams via A&R Sandbox or ML hurdle inference."
    )
    parser.add_argument(
        "--ar-sandbox-mode",
        action="store_true",
        help="Use deterministic A&R Sandbox calculation instead of ML models",
    )
    parser.add_argument(
        "--ml-single-shot",
        action="store_true",
        help="Run ML inference for a single artist using historical state lookup",
    )
    parser.add_argument(
        "--artist",
        type=str,
        default=None,
        help="Artist display name (required for A&R Sandbox or ML single-shot mode)",
    )
    parser.add_argument(
        "--tracks",
        type=int,
        default=DEFAULT_TRACKS,
        help="Total track count for sandbox forecast",
    )
    parser.add_argument(
        "--hits",
        type=int,
        default=DEFAULT_HITS,
        help="Expected hit track count for sandbox forecast",
    )
    parser.add_argument(
        "--lead-single-volume",
        type=float,
        default=0.0,
        help="Hypothetical lead single W1 volume for ML single-shot mode",
    )
    parser.add_argument(
        "--active-singles",
        type=int,
        default=0,
        help="Count of active pre-release singles for ML single-shot mode",
    )
    parser.add_argument(
        "--retention-ratio",
        type=float,
        default=0.5,
        help="W2/W1 retention ratio for ML single-shot mode",
    )
    parser.add_argument(
        "--raw-ig-hype",
        type=float,
        default=None,
        help="Current raw IG hype multiplier for ML single-shot mode",
    )
    parser.add_argument(
        "--tt-outlier-reach",
        type=float,
        default=None,
        help="Current TikTok outlier reach for ML single-shot mode",
    )
    parser.add_argument(
        "--social-csv",
        type=Path,
        default=None,
        help=(
            "Path to a Snowflake CSV export containing POSTS_TIMELINE_DATA and "
            "TIKTOK_VIDEO_TIMELINE_DATA JSON arrays."
        ),
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Parquet file of active rollouts (required for ML inference mode)",
    )
    parser.add_argument(
        "--album-skew-multiplier",
        type=float,
        default=ALBUM_SKEW_MULTIPLIER,
        help="Album skew multiplier applied to hit tracks in Sandbox mode (default: 3.78)",
    )
    parser.add_argument(
        "--current-market-size",
        type=int,
        required=True,
        help="Total global tracking streams for the current week (universe denominator)",
    )
    return parser.parse_args()


def format_streams(value: float) -> str:
    """Format a stream count for executive-friendly terminal output."""
    if np.isnan(value):
        return "N/A"
    return f"{value:,.0f}"


def format_percent(value: float) -> str:
    if np.isnan(value):
        return "N/A"
    return f"{value:.1%}"


def format_spb(value: float) -> str:
    if np.isnan(value):
        return "N/A"
    return f"{value:,.6f}"


def decode_spb_to_streams(spb: float, market_size: int) -> float:
    """Convert Streams Per Billion into raw weekly stream volume."""
    return (spb / SPB_DIVISOR) * market_size


def streams_to_spb(streams: float, market_size: int) -> float:
    """Convert raw stream volume into Streams Per Billion."""
    return (streams / market_size) * SPB_DIVISOR


def load_sandbox_baselines(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.upper()
    return df


def lookup_artist_baseline(
    baselines: pd.DataFrame,
    artist: str,
    current_market_size: int,
) -> tuple[float, bool]:
    """Return momentum-adjusted SPB baseline and whether this is a cold start."""
    baseline_col = (
        "SANDBOX_STANDARD_TRACK_SPB"
        if "SANDBOX_STANDARD_TRACK_SPB" in baselines.columns
        else "ALBUM_STANDARD_TRACK_SPB"
    )
    if baseline_col == "ALBUM_STANDARD_TRACK_SPB":
        print(
            "Warning: SANDBOX_STANDARD_TRACK_SPB not found in baselines CSV. "
            "Falling back to ALBUM_STANDARD_TRACK_SPB. Re-run build_streaming_features.py "
            "to refresh momentum-adjusted baselines."
        )

    artist_lower = artist.strip().lower()
    valid_artists = baselines.dropna(subset=["DISPLAY_ARTIST"])
    matches = baselines[
        baselines["DISPLAY_ARTIST"].str.lower() == artist_lower
    ]

    cold_start_spb = streams_to_spb(COLD_START_BASELINE_STREAMS, current_market_size)

    if matches.empty:
        print(
            f"\nWarning: Artist '{artist}' not found in sandbox baselines. "
            f"Cold Start — defaulting to {format_spb(cold_start_spb)} SPB per track."
        )
        return cold_start_spb, True

    valid = matches.dropna(subset=[baseline_col])
    if valid.empty:
        print(
            f"\nWarning: Artist '{artist}' found but has no baseline history. "
            f"Cold Start — defaulting to {format_spb(cold_start_spb)} SPB per track."
        )
        return cold_start_spb, True

    baseline = float(valid[baseline_col].iloc[0])
    return baseline, False


def calculate_sandbox_forecast(
    tracks: int,
    hits: int,
    sandbox_baseline_spb: float,
    album_skew_multiplier: float,
    current_market_size: int,
) -> dict[str, float]:
    """Deterministic additive sandbox forecast in SPB, decoded to raw streams."""
    standard_tracks = max(0, tracks - hits)
    hit_volume_spb = hits * (sandbox_baseline_spb * album_skew_multiplier)
    standard_volume_spb = standard_tracks * sandbox_baseline_spb
    total_forecast_spb = standard_volume_spb + hit_volume_spb
    total_forecast_streams = decode_spb_to_streams(total_forecast_spb, current_market_size)

    return {
        "standard_tracks": float(standard_tracks),
        "hit_volume_spb": hit_volume_spb,
        "standard_volume_spb": standard_volume_spb,
        "total_forecast_spb": total_forecast_spb,
        "total_forecast": total_forecast_streams,
    }


def print_sandbox_report(
    artist: str,
    tracks: int,
    hits: int,
    sandbox_baseline_spb: float,
    is_cold_start: bool,
    forecast: dict[str, float],
    album_skew_multiplier: float,
    current_market_size: int,
) -> None:
    print("\n" + "=" * 72)
    print("A&R SANDBOX — WEEK 1 ALBUM STREAM FORECAST")
    print("=" * 72)
    print(f"Artist:                  {artist}")
    print(f"Album Track Count:       {tracks}")
    print(f"Expected Hit Tracks:     {hits}")
    print(f"Album Skew Multiplier:   {album_skew_multiplier:.2f}x")
    print(f"Current Market Size:     {current_market_size:,} streams")
    print(
        f"Baseline (per track):    {format_spb(sandbox_baseline_spb)} SPB"
        + ("  [COLD START]" if is_cold_start else "  [momentum-adjusted decay]")
    )
    print("-" * 72)
    print("SPB Volume Split")
    print(
        f"  Standard tracks:       {int(forecast['standard_tracks'])} × "
        f"{format_spb(sandbox_baseline_spb)} = {format_spb(forecast['standard_volume_spb'])} SPB"
    )
    print(
        f"  Hit tracks:            {hits} × "
        f"{format_spb(sandbox_baseline_spb * album_skew_multiplier)} = "
        f"{format_spb(forecast['hit_volume_spb'])} SPB"
    )
    print(f"  Total Forecast SPB:    {format_spb(forecast['total_forecast_spb'])}")
    print("-" * 72)
    print(f"WEEK 1 FORECAST:         {format_streams(forecast['total_forecast'])} streams")
    print("=" * 72 + "\n")


def run_sandbox_mode(
    artist: str,
    tracks: int,
    hits: int,
    album_skew_multiplier: float,
    current_market_size: int,
) -> None:
    if not artist:
        print("Error: --artist is required when --ar-sandbox-mode is set.", file=sys.stderr)
        sys.exit(1)
    if tracks < 0 or hits < 0:
        print("Error: --tracks and --hits must be non-negative.", file=sys.stderr)
        sys.exit(1)
    if hits > tracks:
        print(
            f"Warning: hits ({hits}) exceeds tracks ({tracks}); "
            "standard track count will be floored at 0."
        )

    baselines = load_sandbox_baselines(SANDBOX_BASELINES_PATH)
    sandbox_baseline_spb, is_cold_start = lookup_artist_baseline(
        baselines, artist, current_market_size
    )
    forecast = calculate_sandbox_forecast(
        tracks, hits, sandbox_baseline_spb, album_skew_multiplier, current_market_size
    )
    print_sandbox_report(
        artist,
        tracks,
        hits,
        sandbox_baseline_spb,
        is_cold_start,
        forecast,
        album_skew_multiplier,
        current_market_size,
    )

def load_model_metadata(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_regressor_payload(path: Path) -> LGBMRegressor:
    if not path.exists():
        raise FileNotFoundError(f"Regressor not found: {path}")

    payload = joblib.load(path)
    if isinstance(payload, dict):
        return payload["regressor"]
    return payload


def load_ml_artifacts() -> tuple[BoosterClassifierAdapter, LGBMRegressor, LGBMRegressor, dict[str, Any]]:
    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(f"Classifier not found: {CLASSIFIER_PATH}")
    if not REGRESSOR_PATH.exists():
        raise FileNotFoundError(f"Stage 2 regressor not found: {REGRESSOR_PATH}")
    if not SUPERSTAR_REGRESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Stage 3 superstar regressor not found: {SUPERSTAR_REGRESSOR_PATH}. "
            "Run train_streaming_hurdle.py to export production artifacts."
        )
    if not META_PATH.exists():
        raise FileNotFoundError(f"Metadata not found: {META_PATH}")

    booster = lgb.Booster(model_file=str(CLASSIFIER_PATH))
    classifier = BoosterClassifierAdapter(booster)
    regressor = load_regressor_payload(REGRESSOR_PATH)
    superstar_regressor = load_regressor_payload(SUPERSTAR_REGRESSOR_PATH)

    metadata = load_model_metadata(META_PATH)
    metadata["optimal_prod_threshold"] = float(
        metadata.get("optimal_prod_threshold", GATEKEEPER_PROB_THRESHOLD)
    )
    metadata["superstar_threshold"] = float(metadata["superstar_threshold"])
    metadata["superstar_router_ratio"] = float(
        metadata.get("superstar_router_ratio", 0.8)
    )
    return classifier, regressor, superstar_regressor, metadata


def sanitize_features(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Replace infinities with NaN so tree models and fallbacks behave consistently."""
    features = df[feature_columns].copy()
    return features.replace([np.inf, -np.inf], np.nan)


def prepare_rollout_features(df: pd.DataFrame) -> pd.DataFrame:
    """Impute social features, engineer Stage 3 columns, and sanitize SPB inputs."""
    prepared = df.copy()

    present_volume_cols = [col for col in SOCIAL_VOLUME_COLUMNS if col in prepared.columns]
    present_ratio_cols = [col for col in SOCIAL_RATIO_COLUMNS if col in prepared.columns]
    if present_volume_cols:
        prepared[present_volume_cols] = prepared[present_volume_cols].fillna(0.0)
    if present_ratio_cols:
        prepared[present_ratio_cols] = prepared[present_ratio_cols].fillna(1.0)

    if "RAW_IG_LATE_STAGE_HYPE" not in prepared.columns:
        prepared["RAW_IG_LATE_STAGE_HYPE"] = 1.0
    else:
        prepared["RAW_IG_LATE_STAGE_HYPE"] = prepared["RAW_IG_LATE_STAGE_HYPE"].fillna(1.0)

    prepared = engineer_stage3_features(prepared)

    spb_columns = [
        col
        for col in prepared.columns
        if "SPB" in col or col in FEATURE_COLUMNS or col in STAGE3_FEATURE_COLUMNS
    ]
    if spb_columns:
        prepared[spb_columns] = prepared[spb_columns].replace([np.inf, -np.inf], np.nan)

    return prepared


def effective_tracks(tracks: float) -> float:
    """Logarithmic track penalty anchored to a standard 12-track release."""
    return 12.0 * (np.log(tracks + 1.0) / np.log(13.0))


def load_peer_benchmarks(path: Path = PEER_BENCHMARKS_PATH) -> pd.DataFrame:
    """Load peer debut SPB benchmarks (cached)."""
    global _peer_benchmarks_cache
    if _peer_benchmarks_cache is not None:
        return _peer_benchmarks_cache

    if not path.exists():
        _peer_benchmarks_cache = pd.DataFrame()
        return _peer_benchmarks_cache

    benchmarks = pd.read_csv(path)
    benchmarks.columns = benchmarks.columns.str.upper()
    _peer_benchmarks_cache = benchmarks
    return benchmarks


def extract_canonical_genre(genre_raw: Any) -> str:
    """Safely extract the Luminate MAIN_GENRE from a nested JSON string or array."""
    if pd.isna(genre_raw) or not str(genre_raw).strip():
        return "Unknown"

    if isinstance(genre_raw, str) and (genre_raw.startswith("[") or genre_raw.startswith("{")):
        try:
            parsed = json.loads(genre_raw)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and item.get("CLIENT_DOMAIN") == "Luminate":
                        return str(item.get("MAIN_GENRE", "Unknown"))
                if len(parsed) > 0 and isinstance(parsed[0], dict):
                    return str(parsed[0].get("MAIN_GENRE", "Unknown"))
        except json.JSONDecodeError:
            pass

    return str(genre_raw)


def standard_release_fallback(
    row: pd.Series,
    current_market_size: int,
    peer_benchmarks: pd.DataFrame,
) -> float:
    """3-level SPB fallback hierarchy decoded to raw streams for standard releases."""
    tracks = row.get("TOTAL_TRACKS_ANALYZED", 12.0)
    if pd.isna(tracks) or tracks <= 0:
        tracks = 12.0

    historical_spb = row.get("HISTORICAL_STANDARD_TRACK_SPB", np.nan)
    lead_peak = float(row.get("LEAD_SINGLE_PEAK_VOLUME", 0.0))
    spike_ratio = max(0.5, min(float(row.get("SHORT_TERM_SPIKE_RATIO", 1.0)), 3.0))
    track_multiplier = effective_tracks(float(tracks))

    if pd.notna(historical_spb) and float(historical_spb) > 0:
        fallback_spb = track_multiplier * float(historical_spb) * spike_ratio
        raw_streams = decode_spb_to_streams(fallback_spb, current_market_size)
        return max(raw_streams, RAW_STREAM_FLOOR)

    if lead_peak > 0:
        implied_spb = streams_to_spb(lead_peak, current_market_size) / HYPE_CONVERSION_DIVISOR
        fallback_spb = track_multiplier * implied_spb
        raw_streams = decode_spb_to_streams(fallback_spb, current_market_size)
        return max(raw_streams, RAW_STREAM_FLOOR)

    genre = extract_canonical_genre(row.get("GENRE"))
    distributor = str(row.get("LEVEL_2_DISTRIBUTOR", "Unknown"))
    peer_spb = PEER_BENCHMARK_FALLBACK_SPB

    if not peer_benchmarks.empty:
        exact = peer_benchmarks[
            (peer_benchmarks["GENRE"] == genre)
            & (peer_benchmarks["LEVEL_2_DISTRIBUTOR"] == distributor)
        ]
        dist_match = peer_benchmarks[peer_benchmarks["LEVEL_2_DISTRIBUTOR"] == distributor]
        genre_match = peer_benchmarks[peer_benchmarks["GENRE"] == genre]

        if not exact.empty and pd.notna(exact["MEDIAN_SPB"].iloc[0]):
            peer_spb = float(exact["MEDIAN_SPB"].iloc[0])
        elif not dist_match.empty:
            peer_spb = float(dist_match["MEDIAN_SPB"].median())
        elif not genre_match.empty:
            peer_spb = float(genre_match["MEDIAN_SPB"].median())

    fallback_spb = track_multiplier * peer_spb
    raw_streams = decode_spb_to_streams(fallback_spb, current_market_size)
    return max(raw_streams, RAW_STREAM_FLOOR)


def album_label(row: pd.Series, index: int) -> str:
    for col in ("DISPLAY_ARTIST", "MRELG_ID", "ALBUM_MRELG_ID"):
        if col in row.index and pd.notna(row[col]):
            return str(row[col])
    return f"Album #{index + 1}"


def print_ml_report(
    rollout_df: pd.DataFrame,
    probabilities: np.ndarray,
    prediction_stages: np.ndarray,
    predicted_spb: np.ndarray,
    forecasts: np.ndarray,
) -> None:
    print("\n" + "=" * 72)
    print("ACTIVE ROLLOUT ML INFERENCE — WEEK 1 FORECASTS")
    print("=" * 72)

    for idx, row in rollout_df.iterrows():
        pos = rollout_df.index.get_loc(idx)
        label = album_label(row, pos)
        stage = prediction_stages[pos]
        stage_label = STAGE_DISPLAY_LABELS.get(stage, str(stage))

        active_singles = row.get("ACTIVE_SINGLE_COUNT", np.nan)
        lead_peak = row.get("LEAD_SINGLE_PEAK_VOLUME", np.nan)
        retention = row.get("RETENTION_RATIO", np.nan)
        ig_hype = row.get("IG_LATE_STAGE_HYPE", np.nan)
        raw_ig_hype = row.get("RAW_IG_LATE_STAGE_HYPE", np.nan)
        tt_reach = row.get("TT_OUTLIER_REACH", np.nan)

        print(f"\n{label}")
        print("-" * 72)
        print(f"  Active Singles:        {int(active_singles) if pd.notna(active_singles) else 'N/A'}")
        print(f"  Lead Single W1 Volume: {format_streams(float(lead_peak)) if pd.notna(lead_peak) else 'N/A'}")
        print(f"  Retention Ratio:       {format_percent(float(retention)) if pd.notna(retention) else 'N/A'}")
        print(f"  IG Late-Stage Hype:    {float(ig_hype):.2f}" if pd.notna(ig_hype) else "  IG Late-Stage Hype:    N/A")
        print(
            f"  Raw IG Hype:           {float(raw_ig_hype):.2f}"
            if pd.notna(raw_ig_hype)
            else "  Raw IG Hype:           N/A"
        )
        print(f"  TT Outlier Reach:      {float(tt_reach):.2f}" if pd.notna(tt_reach) else "  TT Outlier Reach:      N/A")
        print(f"  Gatekeeper Prob:       {probabilities[pos]:.1%}")
        print(f"  Prediction Stage:      {stage_label}")
        if not np.isnan(predicted_spb[pos]):
            print(f"  Predicted SPB:         {format_spb(float(predicted_spb[pos]))}")
        print(f"  WEEK 1 FORECAST:       {format_streams(float(forecasts[pos]))} streams")

    superstar_count = int((prediction_stages == STAGE_LABEL_SUPERSTAR).sum())
    standard_count = int((prediction_stages == STAGE_LABEL_STANDARD).sum())
    gatekeeper_count = int((prediction_stages == STAGE_LABEL_GATEKEEPER).sum())

    print("\n" + "=" * 72)
    print(
        f"Summary: {len(rollout_df)} album(s) scored — "
        f"{superstar_count} Superstar, {standard_count} Standard, "
        f"{gatekeeper_count} Gatekeeper Fallback"
    )
    print("=" * 72 + "\n")


def extract_social_multipliers_from_csv(csv_path: Path) -> tuple[float | None, float | None]:
    """Compute RAW_IG_LATE_STAGE_HYPE and TT_OUTLIER_REACH from a Snowflake social export."""
    if not csv_path.exists():
        print(f"Warning: Social CSV not found: {csv_path}", file=sys.stderr)
        return None, None

    try:
        social_df = pd.read_csv(csv_path)
    except (OSError, pd.errors.ParserError, ValueError) as exc:
        print(f"Warning: Failed to read social CSV ({csv_path}): {exc}", file=sys.stderr)
        return None, None

    if social_df.empty:
        print(f"Warning: Social CSV is empty: {csv_path}", file=sys.stderr)
        return None, None

    social_df.columns = social_df.columns.str.upper()
    row_series = social_df.iloc[0]

    if "FIRST_SALE_DATE" not in social_df.columns:
        print(
            "Warning: Social CSV is missing FIRST_SALE_DATE; cannot compute social multipliers.",
            file=sys.stderr,
        )
        return None, None

    first_sale_date = pd.to_datetime(row_series["FIRST_SALE_DATE"], errors="coerce")
    if pd.isna(first_sale_date):
        print(
            "Warning: Social CSV has an invalid FIRST_SALE_DATE; cannot compute social multipliers.",
            file=sys.stderr,
        )
        return None, None

    row_df = social_df.iloc[[0]].copy()
    row_df["FIRST_SALE_DATE"] = first_sale_date
    row = next(row_df.itertuples(index=False))
    features = extract_social_features(row)

    first_sale_date_str = format_release_datetime(first_sale_date)
    raw_ig_hype: float | None = None
    tt_outlier_reach: float | None = None

    if "POSTS_TIMELINE_DATA" in social_df.columns:
        valid_ig_posts = filter_pre_release_entries(
            parse_timeline_json(row_series.get("POSTS_TIMELINE_DATA")),
            first_sale_date_str,
        )
        if valid_ig_posts:
            raw_ig_hype = float(features["RAW_IG_LATE_STAGE_HYPE"])

    if "TIKTOK_VIDEO_TIMELINE_DATA" in social_df.columns:
        valid_tt_videos = filter_pre_release_entries(
            parse_timeline_json(row_series.get("TIKTOK_VIDEO_TIMELINE_DATA")),
            first_sale_date_str,
        )
        if valid_tt_videos:
            tt_value = features.get("TT_OUTLIER_REACH", np.nan)
            if pd.notna(tt_value):
                tt_outlier_reach = float(tt_value)

    return raw_ig_hype, tt_outlier_reach


def build_single_shot_df(
    artist: str,
    tracks: int,
    lead_volume: float,
    active_singles: int,
    retention: float,
    raw_ig_hype: float | None,
    tt_outlier_reach: float | None,
) -> pd.DataFrame:
    """Hydrate historical features for an artist and inject hypothetical rollout data."""
    if not FEATURE_MATRIX_PATH.exists():
        raise FileNotFoundError(
            f"Feature matrix required for single-shot lookup: {FEATURE_MATRIX_PATH}"
        )

    history_df = pd.read_parquet(FEATURE_MATRIX_PATH)
    artist_lower = artist.strip().lower()
    matches = history_df[history_df["DISPLAY_ARTIST"].str.lower() == artist_lower]

    if not matches.empty:
        matches = matches.sort_values("FIRST_SALE_DATE", ascending=False)
        base_row = matches.iloc[0].copy()
        is_debut = 0
    else:
        print(
            f"\nWarning: Artist '{artist}' not found in historical data. Treating as True Debut."
        )
        base_row = pd.Series(
            {
                "DISPLAY_ARTIST": artist,
                "GENRE": "Unknown",
                "LEVEL_2_DISTRIBUTOR": "Unknown",
                "CATALOG_VELOCITY_SLOPE": 0.0,
                "HISTORICAL_STANDARD_TRACK_SPB": np.nan,
                "HISTORICAL_MACRO_MOMENTUM": 0.0,
                "SHORT_TERM_SPIKE_RATIO": 1.0,
                "RAW_IG_LATE_STAGE_HYPE": 1.0,
            }
        )
        is_debut = 1

    for col in SOCIAL_VOLUME_COLUMNS:
        if col not in base_row.index or pd.isna(base_row.get(col)):
            base_row[col] = 0.0
    for col in SOCIAL_RATIO_COLUMNS:
        if col not in base_row.index or pd.isna(base_row.get(col)):
            base_row[col] = 1.0
    if pd.isna(base_row.get("RAW_IG_LATE_STAGE_HYPE")):
        base_row["RAW_IG_LATE_STAGE_HYPE"] = 1.0

    base_row["TOTAL_TRACKS_ANALYZED"] = tracks
    base_row["LEAD_SINGLE_PEAK_VOLUME"] = lead_volume
    base_row["ACTIVE_SINGLE_COUNT"] = active_singles
    base_row["RETENTION_RATIO"] = retention if lead_volume > 0 else np.nan
    base_row["IS_DEBUT_ALBUM"] = is_debut

    velocity = float(base_row.get("CATALOG_VELOCITY_SLOPE", 0.0))
    if pd.isna(velocity):
        velocity = 0.0
    base_row["VELOCITY_X_SINGLES"] = velocity * active_singles

    if raw_ig_hype is not None:
        base_row["RAW_IG_LATE_STAGE_HYPE"] = raw_ig_hype
        base_row["IG_LATE_STAGE_HYPE"] = min(raw_ig_hype, 10.0)
    if tt_outlier_reach is not None:
        base_row["TT_OUTLIER_REACH"] = tt_outlier_reach

    return prepare_rollout_features(pd.DataFrame([base_row]))


def run_ml_mode(rollout_df: pd.DataFrame, current_market_size: int) -> None:
    peer_benchmarks = load_peer_benchmarks()
    classifier, regressor, superstar_regressor, metadata = load_ml_artifacts()

    feature_columns: list[str] = metadata.get("feature_columns", FEATURE_COLUMNS)
    stage3_feature_columns: list[str] = metadata.get(
        "stage3_feature_columns", STAGE3_FEATURE_COLUMNS
    )

    prepared_df = prepare_rollout_features(rollout_df)
    missing_features = [
        col
        for col in [*feature_columns, *stage3_feature_columns, "GENRE", "LEVEL_2_DISTRIBUTOR"]
        if col not in prepared_df.columns
    ]
    if missing_features:
        print(
            f"Error: rollout data is missing required feature columns: {missing_features}",
            file=sys.stderr,
        )
        sys.exit(1)

    features = sanitize_features(prepared_df, feature_columns)
    probabilities = gatekeeper_predict_proba(classifier, features)

    predicted_spb, prediction_stages = predict_pipeline(
        X_input=prepared_df,
        X_stage3=prepared_df,
        classifier=classifier,
        regressor_stage2=regressor,
        regressor_stage3=superstar_regressor,
        gatekeeper_threshold=metadata["optimal_prod_threshold"],
        superstar_threshold=metadata["superstar_threshold"],
        superstar_router_ratio=metadata["superstar_router_ratio"],
    )

    forecasts = np.zeros(len(prepared_df), dtype=float)
    gatekeeper_mask = prediction_stages == STAGE_LABEL_GATEKEEPER
    routed_mask = ~gatekeeper_mask

    if gatekeeper_mask.any():
        for pos in np.where(gatekeeper_mask)[0]:
            forecasts[pos] = standard_release_fallback(
                prepared_df.iloc[pos], current_market_size, peer_benchmarks
            )

    if routed_mask.any():
        decoded_streams = np.array(
            [
                decode_spb_to_streams(float(spb), current_market_size)
                for spb in predicted_spb[routed_mask]
            ],
            dtype=float,
        )
        forecasts[routed_mask] = np.maximum(decoded_streams, RAW_STREAM_FLOOR)

    print_ml_report(
        prepared_df,
        probabilities,
        prediction_stages,
        predicted_spb,
        forecasts,
    )


def main() -> None:
    args = parse_args()

    if args.current_market_size <= 0:
        print("Error: --current-market-size must be a positive integer.", file=sys.stderr)
        sys.exit(1)

    if args.ar_sandbox_mode:
        run_sandbox_mode(
            args.artist or "",
            args.tracks,
            args.hits,
            args.album_skew_multiplier,
            args.current_market_size,
        )
        return

    if args.ml_single_shot:
        if not args.artist:
            print(
                "Error: --artist is required when using --ml-single-shot.",
                file=sys.stderr,
            )
            sys.exit(1)

        raw_ig_hype = args.raw_ig_hype
        tt_outlier_reach = args.tt_outlier_reach

        if args.social_csv is not None:
            if not args.social_csv.exists():
                print(f"Error: Social CSV not found: {args.social_csv}", file=sys.stderr)
                sys.exit(1)

            csv_raw_ig_hype, csv_tt_outlier_reach = extract_social_multipliers_from_csv(
                args.social_csv
            )
            print("\nExtracted Social Multipliers from CSV:")
            print("-" * 72)
            print(
                f"  RAW_IG_LATE_STAGE_HYPE: "
                f"{csv_raw_ig_hype if csv_raw_ig_hype is not None else 'N/A (historical fallback)'}"
            )
            print(
                f"  TT_OUTLIER_REACH:       "
                f"{csv_tt_outlier_reach if csv_tt_outlier_reach is not None else 'N/A (historical fallback)'}"
            )

            if raw_ig_hype is None:
                raw_ig_hype = csv_raw_ig_hype
            if tt_outlier_reach is None:
                tt_outlier_reach = csv_tt_outlier_reach

        synthetic_df = build_single_shot_df(
            artist=args.artist,
            tracks=args.tracks,
            lead_volume=args.lead_single_volume,
            active_singles=args.active_singles,
            retention=args.retention_ratio,
            raw_ig_hype=raw_ig_hype,
            tt_outlier_reach=tt_outlier_reach,
        )

        print("\nGenerated Synthetic Rollout Features:")
        print("-" * 72)
        for col in [
            "HISTORICAL_STANDARD_TRACK_SPB",
            "CATALOG_VELOCITY_SLOPE",
            "VELOCITY_X_SINGLES",
            "SHORT_TERM_SPIKE_RATIO",
            "IG_LATE_STAGE_HYPE",
            "RAW_IG_LATE_STAGE_HYPE",
            "SUPERSTAR_MOMENTUM_INDEX",
            "TT_OUTLIER_REACH",
        ]:
            print(f"  {col}: {synthetic_df.iloc[0].get(col, 'N/A')}")

        run_ml_mode(synthetic_df, args.current_market_size)
        return

    if args.input_file is None:
        print(
            "Error: --input-file is required when not using --ar-sandbox-mode or --ml-single-shot.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not args.input_file.exists():
        print(f"Error: input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    run_ml_mode(pd.read_parquet(args.input_file), args.current_market_size)


if __name__ == "__main__":
    main()
