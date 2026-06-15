"""Dual-mode inference engine for Week 1 album stream forecasts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import linregress

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from train_streaming_hurdle import (
    FEATURE_COLUMNS,
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
CHARTEX_MULTIPLIER_PATH = MODELS_DIR / "chartex_multiplier_regressor.joblib"

ALBUM_SKEW_MULTIPLIER = 3.78
HYPE_CONVERSION_DIVISOR = 2.01
PEER_BENCHMARK_FALLBACK_SPB = 15.0
COLD_START_BASELINE_STREAMS = 50_000.0
SPB_DIVISOR = 1_000_000_000
GATEKEEPER_PROB_THRESHOLD = 0.5 #default that should get replaced by optimal from metadata
RAW_STREAM_FLOOR = 500.0
DEFAULT_TRACKS = 12
DEFAULT_HITS = 1
SNOWFLAKE_ENV_PATH = Path("secrets/amg_research.env")
SNOWFLAKE_BASE_ENV_VARS = (
    "SNOWFLAKE_USER",
    "SNOWFLAKE_ACCOUNT",
    "SNOWFLAKE_WAREHOUSE",
    "SNOWFLAKE_ROLE",
)
LIVE_CATALOG_SQL = """
SELECT
    FLOOR(DATEDIFF(DAY, asd.report_date, %(anchor_date)s) / 7) AS weeks_prior_to_release,
    COALESCE(SUM(IFF(
        asd.metric_category = 'Streams'
        AND asd.service_type = 'OnDemand'
        AND asd.content_type = 'Audio',
        asd.quantity, 0
    )), 0) AS weekly_catalog_streams
FROM LUMINATE_PROD.EXTRACT_S.VW_DAILY_FACT_ARTIST_SUMMARY_DS asd
WHERE asd.COUNTRY_CODE = 'AA'
    AND asd.ARTIST_ID = %(artist_id)s
    AND asd.report_date >= DATEADD(DAY, -84, %(anchor_date)s)
    AND asd.report_date < %(anchor_date)s
GROUP BY ALL
ORDER BY weeks_prior_to_release ASC
"""
LIVE_CHARTEX_SQL = """
WITH artist_meta AS (
    SELECT artist_name
    FROM CURRENT_DEV.DATA.ARTIST_METADATA
    WHERE luminate_artist_id = %(artist_id)s
    LIMIT 1
),
chartex_snapshots AS (
    SELECT ctd.*
    FROM CURRENT_DEV.DATA.CHARTEX_TIKTOK_DATA ctd
    JOIN artist_meta am ON LOWER(ctd.artists) = LOWER(am.artist_name)
    WHERE DATE(ctd.created_at) <= %(anchor_date)s
),
pre_release_velocity AS (
    SELECT *
    FROM chartex_snapshots
    QUALIFY ROW_NUMBER() OVER(
        PARTITION BY tiktok_sound_id
        ORDER BY created_at DESC
    ) = 1
)
SELECT
    COUNT(DISTINCT tiktok_sound_id) AS trending_sounds_count,
    SUM(tiktok_total_video_count) AS album_total_tt_videos,
    SUM(tiktok_last_7_days_video_count) AS album_tt_videos_last_7d,
    SUM(tiktok_last_24_hours_video_count) AS album_tt_videos_last_24h,
    SUM(total_video_views) AS album_total_tt_views,
    SUM(total_video_likes) AS album_total_tt_likes,
    SUM(total_video_saves) AS album_total_tt_saves,
    SUM(total_video_shares) AS album_total_tt_shares
FROM pre_release_velocity
"""

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
        "--artist-id",
        type=str,
        default=None,
        help="Luminate Artist ID for real-time Snowflake catalog and Chartex extraction",
    )
    parser.add_argument(
        "--first-sale-date",
        type=str,
        default=None,
        help=(
            "Target release date (YYYY-MM-DD). Data extraction anchors to the earlier "
            "of this date and today."
        ),
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


def load_snowflake_env() -> None:
    """Load Snowflake credentials from secrets/amg_research.env."""
    if not SNOWFLAKE_ENV_PATH.exists():
        raise EnvironmentError(
            f"Snowflake env file not found: {SNOWFLAKE_ENV_PATH}. "
            "Copy secrets/amg_research.env.example and fill in credentials."
        )

    if load_dotenv is not None:
        load_dotenv(SNOWFLAKE_ENV_PATH)


def _load_snowflake_private_key() -> bytes:
    """Load a PKCS#8 private key for Snowflake key-pair authentication."""
    try:
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import serialization
    except ImportError as exc:
        raise ImportError(
            "cryptography is required for Snowflake key-pair auth. "
            "Install with: pip install cryptography"
        ) from exc

    key_path = os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH")
    key_pem = os.getenv("SNOWFLAKE_PRIVATE_KEY")
    passphrase = os.getenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE")

    if key_path:
        key_data = Path(key_path).expanduser().read_bytes()
    elif key_pem:
        key_data = key_pem.replace("\\n", "\n").encode()
    else:
        raise EnvironmentError(
            "Key-pair auth requires SNOWFLAKE_PRIVATE_KEY_PATH or SNOWFLAKE_PRIVATE_KEY."
        )

    private_key = serialization.load_pem_private_key(
        key_data,
        password=passphrase.encode() if passphrase else None,
        backend=default_backend(),
    )
    return private_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def get_snowflake_connection():
    """Open a Snowflake connection using secrets/amg_research.env credentials."""
    try:
        import snowflake.connector
    except ImportError as exc:
        raise ImportError(
            "snowflake-connector-python is required for live catalog extraction. "
            f"Install with: {sys.executable} -m pip install snowflake-connector-python "
            "(current interpreter: "
            f"{sys.executable})"
        ) from exc

    load_snowflake_env()
    missing = [var for var in SNOWFLAKE_BASE_ENV_VARS if not os.getenv(var)]
    if missing:
        raise EnvironmentError(
            f"Missing Snowflake environment variables: {', '.join(missing)}"
        )

    connect_kwargs: dict[str, Any] = {
        "user": os.environ["SNOWFLAKE_USER"],
        "account": os.environ["SNOWFLAKE_ACCOUNT"],
        "warehouse": os.environ["SNOWFLAKE_WAREHOUSE"],
        "role": os.environ["SNOWFLAKE_ROLE"],
    }

    database = os.getenv("SNOWFLAKE_DATABASE")
    schema = os.getenv("SNOWFLAKE_SCHEMA")
    if database:
        connect_kwargs["database"] = database
    if schema:
        connect_kwargs["schema"] = schema

    auth_method = os.getenv("SNOWFLAKE_AUTH_METHOD", "password").lower()
    if auth_method == "key_pair":
        connect_kwargs["private_key"] = _load_snowflake_private_key()
    else:
        password = os.getenv("SNOWFLAKE_PASSWORD")
        if not password:
            raise EnvironmentError(
                "Password auth requires SNOWFLAKE_PASSWORD in secrets/amg_research.env."
            )
        connect_kwargs["password"] = password

    return snowflake.connector.connect(**connect_kwargs)


def compute_catalog_velocity_slope(weekly_catalog: pd.DataFrame) -> float:
    """Mirror build_streaming_features.compute_catalog_velocity on a weekly frame."""
    sorted_weeks = weekly_catalog.sort_values("WEEKS_PRIOR_TO_RELEASE", ascending=False)
    if len(sorted_weeks) < 2:
        return 0.0

    x = np.arange(len(sorted_weeks))
    y = sorted_weeks["WEEKLY_CATALOG_STREAMS"].to_numpy(dtype=float)
    try:
        slope, _, _, _, _ = linregress(x, y)
    except ValueError:
        return 0.0
    return float(slope)


def compute_short_term_spike_ratio(weekly_catalog: pd.DataFrame) -> float:
    """Mirror build_streaming_features.compute_short_term_spike on a weekly frame."""
    sorted_weeks = weekly_catalog.sort_values("WEEKS_PRIOR_TO_RELEASE", ascending=True)
    if len(sorted_weeks) < 2:
        return 1.0

    w1_streams = float(sorted_weeks.iloc[0]["WEEKLY_CATALOG_STREAMS"])
    w2_w4_streams = float(sorted_weeks.iloc[1:4]["WEEKLY_CATALOG_STREAMS"].mean())
    if w2_w4_streams == 0:
        return 1.0 if w1_streams == 0 else 2.0

    return float(np.clip(w1_streams / w2_w4_streams, 0.5, 4.0))


def resolve_anchor_date(first_sale_date: str | None) -> str:
    """Return YYYY-MM-DD anchor: min(today, release date) when provided, else today."""
    today = datetime.today().date()
    if first_sale_date:
        parsed_date = datetime.strptime(first_sale_date, "%Y-%m-%d").date()
        anchor_date_obj = min(today, parsed_date)
    else:
        anchor_date_obj = today
    return anchor_date_obj.strftime("%Y-%m-%d")


def _chartex_row_is_empty(chartex_row: pd.Series | None) -> bool:
    """True when Chartex returned no usable TikTok aggregates before the anchor date."""
    if chartex_row is None or chartex_row.empty:
        return True
    values = chartex_row.reindex(CHARTEX_RAW_COLUMNS)
    return bool(values.isna().all())


def _safe_ratio(numerator: Any, denominator: float) -> float:
    """Divide safely, preserving NaN when the numerator is missing."""
    if pd.isna(numerator):
        return np.nan
    return float(numerator) / denominator


def engineer_live_chartex_features(chartex_row: pd.Series) -> dict[str, float]:
    """Engineer TikTok ratios for inference; empty rows propagate NaN to LightGBM."""
    engineered: dict[str, float] = {col: np.nan for col in CHARTEX_ENGINEERED_COLUMNS}
    if _chartex_row_is_empty(chartex_row):
        return engineered

    safe_total_videos = max(float(chartex_row["ALBUM_TOTAL_TT_VIDEOS"] or 0.0), 1.0)
    safe_total_views = max(float(chartex_row["ALBUM_TOTAL_TT_VIEWS"] or 0.0), 1.0)

    engineered["TT_VIRAL_CONCENTRATION"] = _safe_ratio(
        chartex_row["ALBUM_TT_VIDEOS_LAST_7D"], safe_total_videos
    )
    engineered["TT_TERMINAL_ACCELERATION"] = _safe_ratio(
        chartex_row["ALBUM_TT_VIDEOS_LAST_24H"], safe_total_videos
    )
    engineered["TT_ENGAGEMENT_DEPTH"] = _safe_ratio(
        chartex_row["ALBUM_TOTAL_TT_SAVES"] + chartex_row["ALBUM_TOTAL_TT_SHARES"],
        safe_total_views,
    )
    engineered["TT_LIKE_RATIO"] = _safe_ratio(
        chartex_row["ALBUM_TOTAL_TT_LIKES"], safe_total_views
    )
    return engineered


def fetch_live_snowflake_metrics(artist_id: str, anchor_date: str) -> dict[str, Any]:
    """Pull live catalog velocity and Chartex TikTok aggregates anchored to a release date."""
    connection = get_snowflake_connection()
    try:
        cursor = connection.cursor()
        cursor.execute(
            LIVE_CATALOG_SQL,
            {"artist_id": artist_id, "anchor_date": anchor_date},
        )
        catalog_rows = cursor.fetchall()
        catalog_columns = [col[0] for col in cursor.description]

        cursor.execute(
            LIVE_CHARTEX_SQL,
            {"artist_id": artist_id, "anchor_date": anchor_date},
        )
        chartex_rows = cursor.fetchall()
        chartex_columns = [col[0] for col in cursor.description]
    finally:
        connection.close()

    if not catalog_rows:
        raise ValueError(
            f"No catalog stream rows returned from Snowflake for ARTIST_ID={artist_id}"
        )

    weekly_catalog = pd.DataFrame(catalog_rows, columns=catalog_columns)
    weekly_catalog.columns = weekly_catalog.columns.str.upper()
    weekly_catalog["WEEKS_PRIOR_TO_RELEASE"] = weekly_catalog["WEEKS_PRIOR_TO_RELEASE"].astype(
        int
    )
    weekly_catalog["WEEKLY_CATALOG_STREAMS"] = weekly_catalog["WEEKLY_CATALOG_STREAMS"].astype(
        float
    )

    catalog_velocity_slope = compute_catalog_velocity_slope(weekly_catalog)
    short_term_spike_ratio = compute_short_term_spike_ratio(weekly_catalog)

    chartex_row: pd.Series | None = None
    if chartex_rows:
        chartex_df = pd.DataFrame(chartex_rows, columns=chartex_columns)
        chartex_df.columns = chartex_df.columns.str.upper()
        chartex_row = chartex_df.iloc[0]

    raw_chartex: dict[str, float] = {col: np.nan for col in CHARTEX_RAW_COLUMNS}
    if not _chartex_row_is_empty(chartex_row):
        assert chartex_row is not None
        for col in CHARTEX_RAW_COLUMNS:
            value = chartex_row.get(col, np.nan)
            raw_chartex[col] = np.nan if pd.isna(value) else float(value)

    engineered_chartex = engineer_live_chartex_features(
        chartex_row if chartex_row is not None else pd.Series(dtype=float)
    )

    return {
        "anchor_date": anchor_date,
        "catalog_velocity_slope": catalog_velocity_slope,
        "short_term_spike_ratio": short_term_spike_ratio,
        **raw_chartex,
        **engineered_chartex,
    }


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


def load_stage4_payload() -> dict[str, Any] | None:
    """Load Stage 4 Chartex multiplier artifacts when exported."""
    if not CHARTEX_MULTIPLIER_PATH.exists():
        return None
    return joblib.load(CHARTEX_MULTIPLIER_PATH)


def apply_stage4_multiplier(
    primary_spb: float,
    synthetic_row: pd.Series,
    stage4_payload: dict[str, Any],
) -> tuple[float, float]:
    """Exponentiate the log-multiplier prediction and cascade onto primary SPB."""
    chartex_features: list[str] = stage4_payload["chartex_features"]
    stage4_model: LGBMRegressor = stage4_payload["regressor"]
    multiplier_min = float(stage4_payload.get("multiplier_min", 0.1))
    multiplier_max = float(stage4_payload.get("multiplier_max", 100.0))

    X_stage4 = pd.DataFrame([synthetic_row[chartex_features].to_dict()])
    predicted_log_multiplier = float(stage4_model.predict(X_stage4)[0])
    viral_multiplier = float(np.clip(np.exp(predicted_log_multiplier), multiplier_min, multiplier_max))
    final_spb = primary_spb * viral_multiplier
    return final_spb, viral_multiplier


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
    """Sanitize rollout features before three-stage inference."""
    prepared = engineer_stage3_features(df.copy())

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

        print(f"\n{label}")
        print("-" * 72)
        print(f"  Active Singles:        {int(active_singles) if pd.notna(active_singles) else 'N/A'}")
        print(f"  Lead Single W1 Volume: {format_streams(float(lead_peak)) if pd.notna(lead_peak) else 'N/A'}")
        print(f"  Retention Ratio:       {format_percent(float(retention)) if pd.notna(retention) else 'N/A'}")
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


def build_single_shot_df(
    artist: str,
    tracks: int,
    lead_volume: float,
    active_singles: int,
    retention: float,
    live_metrics: dict[str, Any] | None = None,
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
            }
        )
        is_debut = 1

    base_row["TOTAL_TRACKS_ANALYZED"] = tracks
    base_row["LEAD_SINGLE_PEAK_VOLUME"] = lead_volume
    base_row["ACTIVE_SINGLE_COUNT"] = active_singles
    base_row["RETENTION_RATIO"] = retention if lead_volume > 0 else np.nan
    base_row["IS_DEBUT_ALBUM"] = is_debut

    for col in [*CHARTEX_RAW_COLUMNS, *CHARTEX_ENGINEERED_COLUMNS]:
        base_row[col] = np.nan

    if live_metrics:
        if "catalog_velocity_slope" in live_metrics:
            base_row["CATALOG_VELOCITY_SLOPE"] = live_metrics["catalog_velocity_slope"]
        if "short_term_spike_ratio" in live_metrics:
            base_row["SHORT_TERM_SPIKE_RATIO"] = live_metrics["short_term_spike_ratio"]

        for col in [*CHARTEX_RAW_COLUMNS, *CHARTEX_ENGINEERED_COLUMNS]:
            value = live_metrics.get(col, np.nan)
            base_row[col] = np.nan if value is None or pd.isna(value) else value

    velocity = float(base_row.get("CATALOG_VELOCITY_SLOPE", 0.0))
    if pd.isna(velocity):
        velocity = 0.0
    base_row["VELOCITY_X_SINGLES"] = velocity * active_singles

    return prepare_rollout_features(pd.DataFrame([base_row]))


def print_single_shot_report(
    artist: str,
    row: pd.Series,
    probability: float,
    prediction_stage: str,
    primary_spb: float,
    final_spb: float,
    final_streams: float,
    viral_multiplier: float | None,
    current_market_size: int,
) -> None:
    """Print a single-artist ML forecast with optional Stage 4 cascade details."""
    stage_label = STAGE_DISPLAY_LABELS.get(prediction_stage, prediction_stage)
    active_singles = row.get("ACTIVE_SINGLE_COUNT", np.nan)
    lead_peak = row.get("LEAD_SINGLE_PEAK_VOLUME", np.nan)
    retention = row.get("RETENTION_RATIO", np.nan)

    print("\n" + "=" * 72)
    print("ML SINGLE-SHOT INFERENCE — WEEK 1 ALBUM STREAM FORECAST")
    print("=" * 72)
    print(f"Artist:                  {artist}")
    print(f"Current Market Size:     {current_market_size:,} streams")
    print("-" * 72)
    print(f"  Active Singles:        {int(active_singles) if pd.notna(active_singles) else 'N/A'}")
    print(
        f"  Lead Single W1 Volume: {format_streams(float(lead_peak)) if pd.notna(lead_peak) else 'N/A'}"
    )
    print(f"  Retention Ratio:       {format_percent(float(retention)) if pd.notna(retention) else 'N/A'}")
    print(f"  Gatekeeper Prob:       {probability:.1%}")
    print(f"  Prediction Stage:      {stage_label}")
    if not np.isnan(primary_spb):
        print(f"  Primary Predicted SPB: {format_spb(primary_spb)}")
    if viral_multiplier is not None:
        print(
            f"  [Cascaded Stage 4 Multiplier Applied: {viral_multiplier:.2f}x]"
        )
    if not np.isnan(final_spb):
        print(f"  Final Predicted SPB:   {format_spb(final_spb)}")
    print(f"  WEEK 1 FORECAST:       {format_streams(final_streams)} streams")
    print("=" * 72 + "\n")


def run_single_shot_inference(
    synthetic_df: pd.DataFrame,
    artist: str,
    current_market_size: int,
    stage4_payload: dict[str, Any] | None = None,
) -> None:
    """Score one synthetic rollout row, optionally cascading Stage 4 Chartex multiplier."""
    peer_benchmarks = load_peer_benchmarks()
    classifier, regressor, superstar_regressor, metadata = load_ml_artifacts()

    feature_columns: list[str] = metadata.get("feature_columns", FEATURE_COLUMNS)
    stage3_feature_columns: list[str] = metadata.get(
        "stage3_feature_columns", STAGE3_FEATURE_COLUMNS
    )

    prepared_df = prepare_rollout_features(synthetic_df)
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

    row = prepared_df.iloc[0]
    stage = str(prediction_stages[0])
    probability = float(probabilities[0])
    primary_spb = float(predicted_spb[0]) if not np.isnan(predicted_spb[0]) else np.nan
    viral_multiplier: float | None = None
    final_spb = primary_spb

    if stage == STAGE_LABEL_GATEKEEPER or np.isnan(primary_spb):
        final_streams = standard_release_fallback(row, current_market_size, peer_benchmarks)
        final_spb = np.nan
    else:
        if stage4_payload is not None:
            final_spb, viral_multiplier = apply_stage4_multiplier(
                primary_spb, row, stage4_payload
            )

        final_streams = max(
            decode_spb_to_streams(float(final_spb), current_market_size),
            RAW_STREAM_FLOOR,
        )

    print_single_shot_report(
        artist=artist,
        row=row,
        probability=probability,
        prediction_stage=stage,
        primary_spb=primary_spb,
        final_spb=float(final_spb) if not np.isnan(final_spb) else np.nan,
        final_streams=final_streams,
        viral_multiplier=viral_multiplier,
        current_market_size=current_market_size,
    )


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

        if args.first_sale_date:
            try:
                datetime.strptime(args.first_sale_date, "%Y-%m-%d")
            except ValueError:
                print(
                    "Error: --first-sale-date must be formatted as YYYY-MM-DD.",
                    file=sys.stderr,
                )
                sys.exit(1)

        anchor_date = resolve_anchor_date(args.first_sale_date)
        live_metrics: dict[str, Any] | None = None
        if args.artist_id:
            try:
                live_metrics = fetch_live_snowflake_metrics(args.artist_id, anchor_date)
            except (EnvironmentError, ImportError, ValueError) as exc:
                print(f"Error: Failed to fetch live Snowflake metrics: {exc}", file=sys.stderr)
                sys.exit(1)

            print("\nLive Snowflake Metrics:")
            print("-" * 72)
            print(f"  ARTIST_ID:                {args.artist_id}")
            print(f"  Anchor Date:              {anchor_date}")
            print(f"  CATALOG_VELOCITY_SLOPE:   {live_metrics['catalog_velocity_slope']:,.2f}")
            print(f"  SHORT_TERM_SPIKE_RATIO:   {live_metrics['short_term_spike_ratio']:.4f}")
            for col in CHARTEX_RAW_COLUMNS:
                value = live_metrics.get(col, np.nan)
                print(f"  {col}: {value if pd.isna(value) else f'{value:,.0f}'}")

        synthetic_df = build_single_shot_df(
            artist=args.artist,
            tracks=args.tracks,
            lead_volume=args.lead_single_volume,
            active_singles=args.active_singles,
            retention=args.retention_ratio,
            live_metrics=live_metrics,
        )

        print("\nGenerated Synthetic Rollout Features:")
        print("-" * 72)
        for col in [
            "HISTORICAL_STANDARD_TRACK_SPB",
            "CATALOG_VELOCITY_SLOPE",
            "VELOCITY_X_SINGLES",
            "SHORT_TERM_SPIKE_RATIO",
            *CHARTEX_ENGINEERED_COLUMNS,
        ]:
            print(f"  {col}: {synthetic_df.iloc[0].get(col, 'N/A')}")

        stage4_payload = load_stage4_payload()
        run_single_shot_inference(
            synthetic_df,
            artist=args.artist,
            current_market_size=args.current_market_size,
            stage4_payload=stage4_payload,
        )
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
