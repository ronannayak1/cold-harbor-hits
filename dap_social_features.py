"""DAP pre-release social feature schema and engineering for Stage 4.

Source table: US_LABELS_SANDBOX.RONAN_N.DAP_SOCIAL_PRE_RELEASE
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

DAP_SOCIAL_SOURCE_TABLE = "US_LABELS_SANDBOX.RONAN_N.DAP_SOCIAL_PRE_RELEASE"

DAP_SOCIAL_RAW_COLUMNS = [
    "TT_TOTAL_VIEWS",
    "TT_TOTAL_LIKES",
    "TT_TOTAL_SHARES",
    "TT_TOTAL_COMMENTS",
    "TT_VIEWS_LAST_7D",
    "TT_LIKES_LAST_7D",
    "TT_SHARES_LAST_7D",
    "TT_COMMENTS_LAST_7D",
    "TT_VIEWS_LAST_24H",
    "TT_LIKES_LAST_24H",
    "TT_SHARES_LAST_24H",
    "TT_COMMENTS_LAST_24H",
    "YT_TOTAL_VIEWS",
    "YT_TOTAL_LIKES",
    "YT_TOTAL_SHARES",
    "YT_TOTAL_COMMENTS",
    "YT_VIEWS_LAST_7D",
    "YT_LIKES_LAST_7D",
    "YT_SHARES_LAST_7D",
    "YT_COMMENTS_LAST_7D",
    "YT_VIEWS_LAST_24H",
    "YT_LIKES_LAST_24H",
    "YT_SHARES_LAST_24H",
    "YT_COMMENTS_LAST_24H",
]

DAP_SOCIAL_FETCH_COLUMNS = ["MRELG_ID", "FIRST_SALE_DATE", *DAP_SOCIAL_RAW_COLUMNS]

DAP_SOCIAL_ENGINEERED_COLUMNS = [
    "TT_VIRAL_CONCENTRATION",
    "TT_TERMINAL_ACCELERATION",
    "TT_ENGAGEMENT_DEPTH",
    "TT_LIKE_RATIO",
    "YT_VIRAL_CONCENTRATION",
    "YT_TERMINAL_ACCELERATION",
    "YT_ENGAGEMENT_DEPTH",
    "YT_LIKE_RATIO",
]

# Model inputs: engineered velocity/engagement ratios + raw platform totals.
SOCIAL_FEATURES = DAP_SOCIAL_ENGINEERED_COLUMNS + [
    "TT_TOTAL_VIEWS",
    "TT_TOTAL_LIKES",
    "TT_TOTAL_SHARES",
    "TT_TOTAL_COMMENTS",
    "YT_TOTAL_VIEWS",
    "YT_TOTAL_LIKES",
    "YT_TOTAL_SHARES",
    "YT_TOTAL_COMMENTS",
]

LIVE_DAP_SOCIAL_SQL = f"""
SELECT
    {", ".join(DAP_SOCIAL_FETCH_COLUMNS)}
FROM {DAP_SOCIAL_SOURCE_TABLE}
WHERE MRELG_ID = %(mrelg_id)s
LIMIT 1
"""

# Real social presence: ignore tiny/noisy view counts.
# Chosen at ~1M max(TT,YT) views — below this, social coverage is sparse noise.
REAL_SOCIAL_MIN_VIEWS = 1_000_000.0

# Virality: among albums with real social presence, top-decile 24h view acceleration
# (p90 of max(TT,YT) terminal acceleration ≈ 0.0026 on the 2026 routed cohort).
# Empirically lifts P(actual/primary >= 2) from ~4% → ~8.5% and median residual toward 1.0.
VIRAL_TERMINAL_ACCELERATION_THRESHOLD = 0.0026

# Stage 4 may only cascade in production when holdout multipliers are healthy.
STAGE4_HEALTH_MEDIAN_MIN = 0.85
STAGE4_HEALTH_MEDIAN_MAX = 1.15
STAGE4_HEALTH_P95_MIN = 1.50  # requires a real right tail for virality


def max_platform_views(tt_views: Any, yt_views: Any) -> float:
    """Return max non-negative TT/YT total views, treating missing as 0."""
    tt = 0.0 if pd.isna(tt_views) else max(float(tt_views), 0.0)
    yt = 0.0 if pd.isna(yt_views) else max(float(yt_views), 0.0)
    return max(tt, yt)


def max_terminal_acceleration(tt_acc: Any, yt_acc: Any) -> float:
    """Return max non-negative TT/YT 24h view acceleration."""
    tt = 0.0 if pd.isna(tt_acc) else max(float(tt_acc), 0.0)
    yt = 0.0 if pd.isna(yt_acc) else max(float(yt_acc), 0.0)
    return max(tt, yt)


def has_real_social_signal(
    tt_views: Any,
    yt_views: Any,
    min_views: float = REAL_SOCIAL_MIN_VIEWS,
) -> bool:
    """True when an album has meaningful pre-release TT or YT view volume."""
    return max_platform_views(tt_views, yt_views) >= min_views


def is_viral_social_candidate(
    tt_views: Any,
    yt_views: Any,
    tt_terminal_acc: Any,
    yt_terminal_acc: Any,
    min_views: float = REAL_SOCIAL_MIN_VIEWS,
    acc_threshold: float = VIRAL_TERMINAL_ACCELERATION_THRESHOLD,
) -> bool:
    """True when real social presence coincides with top-decile terminal acceleration."""
    if not has_real_social_signal(tt_views, yt_views, min_views=min_views):
        return False
    return max_terminal_acceleration(tt_terminal_acc, yt_terminal_acc) >= acc_threshold


def annotate_social_gates(df: pd.DataFrame) -> pd.DataFrame:
    """Attach HAS_REAL_SOCIAL / IS_VIRAL_SOCIAL boolean columns."""
    enriched = df.copy()
    max_views = np.maximum(
        enriched["TT_TOTAL_VIEWS"].fillna(0.0).clip(lower=0.0),
        enriched["YT_TOTAL_VIEWS"].fillna(0.0).clip(lower=0.0),
    )
    max_acc = np.maximum(
        enriched["TT_TERMINAL_ACCELERATION"].fillna(0.0).clip(lower=0.0),
        enriched["YT_TERMINAL_ACCELERATION"].fillna(0.0).clip(lower=0.0),
    )
    enriched["HAS_REAL_SOCIAL"] = max_views >= REAL_SOCIAL_MIN_VIEWS
    enriched["IS_VIRAL_SOCIAL"] = enriched["HAS_REAL_SOCIAL"] & (
        max_acc >= VIRAL_TERMINAL_ACCELERATION_THRESHOLD
    )
    return enriched


def assess_stage4_multiplier_health(
    predicted_multipliers: np.ndarray,
    median_min: float = STAGE4_HEALTH_MEDIAN_MIN,
    median_max: float = STAGE4_HEALTH_MEDIAN_MAX,
    p95_min: float = STAGE4_HEALTH_P95_MIN,
) -> dict[str, float | bool]:
    """Require multipliers centered near 1.0 with a viral right tail before enabling cascade."""
    values = np.asarray(predicted_multipliers, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            "enabled": False,
            "median_multiplier": float("nan"),
            "p95_multiplier": float("nan"),
            "mean_multiplier": float("nan"),
        }

    median_mult = float(np.median(values))
    p95_mult = float(np.quantile(values, 0.95))
    mean_mult = float(np.mean(values))
    enabled = (median_min <= median_mult <= median_max) and (p95_mult >= p95_min)
    return {
        "enabled": enabled,
        "median_multiplier": median_mult,
        "p95_multiplier": p95_mult,
        "mean_multiplier": mean_mult,
    }


def should_apply_stage4(
    prediction_stage: str,
    row: pd.Series,
    stage4_payload: dict[str, Any] | None,
    standard_stage_label: str,
) -> tuple[bool, str]:
    """Gate Stage 4 cascade: Standard + real social + healthy multiplier model only."""
    if stage4_payload is None:
        return False, "stage4_model_missing"
    if not bool(stage4_payload.get("stage4_enabled", False)):
        return False, "stage4_health_gate_failed"
    if prediction_stage != standard_stage_label:
        return False, f"stage_excluded:{prediction_stage}"
    if not has_real_social_signal(row.get("TT_TOTAL_VIEWS"), row.get("YT_TOTAL_VIEWS")):
        return False, "no_real_social_signal"
    return True, "applied"


def _safe_positive_denominator(series: pd.Series) -> pd.Series:
    """Floor positive denominators at 1.0 while preserving NaN for missing social data."""
    values = series.astype(float)
    safe_values = values.copy()
    valid_mask = values.notna()
    safe_values.loc[valid_mask] = np.maximum(values.loc[valid_mask], 1.0)
    return safe_values


def _nonnegative(series: pd.Series) -> pd.Series:
    """Clip negative velocity windows to 0 while preserving NaN."""
    values = series.astype(float)
    clipped = values.copy()
    valid_mask = values.notna()
    clipped.loc[valid_mask] = np.maximum(values.loc[valid_mask], 0.0)
    return clipped


def engineer_dap_social_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build TT/YT acceleration and engagement ratios for Stage 4."""
    enriched = df.copy()

    tt_views = _safe_positive_denominator(enriched["TT_TOTAL_VIEWS"])
    yt_views = _safe_positive_denominator(enriched["YT_TOTAL_VIEWS"])

    enriched["TT_VIRAL_CONCENTRATION"] = _nonnegative(enriched["TT_VIEWS_LAST_7D"]) / tt_views
    enriched["TT_TERMINAL_ACCELERATION"] = _nonnegative(enriched["TT_VIEWS_LAST_24H"]) / tt_views
    enriched["TT_ENGAGEMENT_DEPTH"] = (
        enriched["TT_TOTAL_SHARES"] + enriched["TT_TOTAL_COMMENTS"]
    ) / tt_views
    enriched["TT_LIKE_RATIO"] = enriched["TT_TOTAL_LIKES"] / tt_views

    enriched["YT_VIRAL_CONCENTRATION"] = _nonnegative(enriched["YT_VIEWS_LAST_7D"]) / yt_views
    enriched["YT_TERMINAL_ACCELERATION"] = _nonnegative(enriched["YT_VIEWS_LAST_24H"]) / yt_views
    enriched["YT_ENGAGEMENT_DEPTH"] = (
        enriched["YT_TOTAL_SHARES"] + enriched["YT_TOTAL_COMMENTS"]
    ) / yt_views
    enriched["YT_LIKE_RATIO"] = enriched["YT_TOTAL_LIKES"] / yt_views

    return enriched


def _safe_ratio(numerator: Any, denominator: float) -> float:
    if pd.isna(numerator):
        return np.nan
    return float(numerator) / denominator


def _nonnegative_scalar(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    return float(max(float(value), 0.0))


def dap_social_row_is_empty(row: pd.Series | None) -> bool:
    """True when no usable DAP social aggregates are present."""
    if row is None or row.empty:
        return True
    values = row.reindex(DAP_SOCIAL_RAW_COLUMNS)
    return bool(values.isna().all())


def engineer_dap_social_row(row: pd.Series) -> dict[str, float]:
    """Engineer TT/YT ratios for a single inference row; empty rows stay NaN."""
    engineered: dict[str, float] = {col: np.nan for col in DAP_SOCIAL_ENGINEERED_COLUMNS}
    if dap_social_row_is_empty(row):
        return engineered

    safe_tt_views = max(float(row.get("TT_TOTAL_VIEWS") or 0.0), 1.0)
    safe_yt_views = max(float(row.get("YT_TOTAL_VIEWS") or 0.0), 1.0)

    engineered["TT_VIRAL_CONCENTRATION"] = _safe_ratio(
        _nonnegative_scalar(row.get("TT_VIEWS_LAST_7D")), safe_tt_views
    )
    engineered["TT_TERMINAL_ACCELERATION"] = _safe_ratio(
        _nonnegative_scalar(row.get("TT_VIEWS_LAST_24H")), safe_tt_views
    )
    engineered["TT_ENGAGEMENT_DEPTH"] = _safe_ratio(
        (row.get("TT_TOTAL_SHARES") or 0.0) + (row.get("TT_TOTAL_COMMENTS") or 0.0),
        safe_tt_views,
    )
    engineered["TT_LIKE_RATIO"] = _safe_ratio(row.get("TT_TOTAL_LIKES"), safe_tt_views)

    engineered["YT_VIRAL_CONCENTRATION"] = _safe_ratio(
        _nonnegative_scalar(row.get("YT_VIEWS_LAST_7D")), safe_yt_views
    )
    engineered["YT_TERMINAL_ACCELERATION"] = _safe_ratio(
        _nonnegative_scalar(row.get("YT_VIEWS_LAST_24H")), safe_yt_views
    )
    engineered["YT_ENGAGEMENT_DEPTH"] = _safe_ratio(
        (row.get("YT_TOTAL_SHARES") or 0.0) + (row.get("YT_TOTAL_COMMENTS") or 0.0),
        safe_yt_views,
    )
    engineered["YT_LIKE_RATIO"] = _safe_ratio(row.get("YT_TOTAL_LIKES"), safe_yt_views)

    return engineered
