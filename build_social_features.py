"""Engineer pre-release Instagram and TikTok velocity features from nested JSON timelines.

DEPRECATED: This module is not used by the Cold Harbor Hits production pipeline.
The Three-Stage Hurdle architecture reverted to the pre-social streaming baseline.
Do not run this script as part of build_streaming_features.py or model training.
"""

from __future__ import annotations

import json
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
INPUT_PATH = DATA_DIR / "ig_tt_pre_album.csv"
OUTPUT_PATH = DATA_DIR / "social_features.parquet"
CHUNK_SIZE = 10_000
DATETIME_FMT = "%Y-%m-%d %H:%M:%S"
SUPERSTAR_ENGAGEMENT_FLOOR = 500_000.0

FEATURE_COLUMNS = [
    "TOTAL_PRE_RELEASE_IG_FAVES",
    "TOTAL_PRE_RELEASE_IG_COMMENTS",
    "IG_AVG_ENGAGEMENT_RATE",
    "IG_COMMENT_DENSITY",
    "IG_LATE_STAGE_HYPE",
    "RAW_IG_LATE_STAGE_HYPE",
    "TOTAL_PRE_RELEASE_TT_PLAYS",
    "TT_SHARE_VELOCITY",
    "TT_OUTLIER_REACH",
]


def parse_timeline_json(raw_value: Any) -> list[dict[str, Any]]:
    """Parse a JSON timeline column into a list of post/video dicts."""
    if raw_value is None or (isinstance(raw_value, float) and np.isnan(raw_value)):
        return []
    if isinstance(raw_value, list):
        return raw_value
    if not str(raw_value).strip():
        return []

    try:
        parsed = json.loads(raw_value)
    except (json.JSONDecodeError, TypeError):
        return []

    return parsed if isinstance(parsed, list) else []


def filter_pre_release_entries(
    entries: list[dict[str, Any]],
    first_sale_date_str: str,
) -> list[dict[str, Any]]:
    """Keep only timeline entries posted strictly before the album release date."""
    valid: list[dict[str, Any]] = []
    for entry in entries:
        if str(entry.get("posted_at", "")) < first_sale_date_str:
            valid.append(entry)
    return valid


def safe_float(value: Any, default: float = 0.0) -> float:
    """Coerce a metric to float, clamping negative API artifacts to 0.0."""
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        parsed_val = float(value)
        return max(0.0, parsed_val)
    except (TypeError, ValueError):
        return default


def default_features(mrelg_id: Any) -> dict[str, float | str]:
    """Return the baseline feature dict for rows with missing release dates."""
    features: dict[str, float | str] = {"MRELG_ID": mrelg_id}
    for col in FEATURE_COLUMNS:
        features[col] = 0.0
    features["IG_LATE_STAGE_HYPE"] = 1.0
    features["TT_SHARE_VELOCITY"] = np.nan
    features["TT_OUTLIER_REACH"] = np.nan
    return features


def format_release_datetime(value: pd.Timestamp) -> str:
    """Format a release timestamp for lexicographic ISO string comparisons."""
    return value.strftime(DATETIME_FMT)


def extract_social_features(row: Any) -> dict[str, float | str]:
    """Engineer leakage-safe social velocity features for a single album row."""
    mrelg_id = row.MRELG_ID
    first_sale_date = row.FIRST_SALE_DATE

    features = default_features(mrelg_id)
    if pd.isna(first_sale_date):
        return features

    first_sale_date_str = format_release_datetime(first_sale_date)
    t_30 = first_sale_date - timedelta(days=30)
    t_180 = first_sale_date - timedelta(days=180)
    t_30_str = format_release_datetime(t_30)
    t_180_str = format_release_datetime(t_180)

    valid_ig_posts = filter_pre_release_entries(
        parse_timeline_json(row.POSTS_TIMELINE_DATA),
        first_sale_date_str,
    )
    valid_tt_videos = filter_pre_release_entries(
        parse_timeline_json(row.TIKTOK_VIDEO_TIMELINE_DATA),
        first_sale_date_str,
    )

    ig_faves = sum(safe_float(post.get("favorite_count")) for post in valid_ig_posts)
    ig_comments = sum(safe_float(post.get("comment_count")) for post in valid_ig_posts)
    ig_engagements = [
        safe_float(post.get("favorite_count")) + safe_float(post.get("comment_count"))
        for post in valid_ig_posts
    ]

    features["TOTAL_PRE_RELEASE_IG_FAVES"] = ig_faves
    features["TOTAL_PRE_RELEASE_IG_COMMENTS"] = ig_comments
    features["IG_AVG_ENGAGEMENT_RATE"] = (
        float(np.mean(ig_engagements)) if ig_engagements else 0.0
    )
    features["IG_COMMENT_DENSITY"] = ig_comments / max(ig_faves, 1.0)

    if valid_ig_posts:
        engagements_30d = 0.0
        engagements_150d_prior = 0.0
        late_stage_engagements: list[float] = []

        for post in valid_ig_posts:
            posted_at_str = str(post.get("posted_at", ""))
            engagement = safe_float(post.get("favorite_count")) + safe_float(
                post.get("comment_count")
            )
            if t_30_str <= posted_at_str < first_sale_date_str:
                engagements_30d += engagement
                late_stage_engagements.append(engagement)
            elif t_180_str <= posted_at_str < t_30_str:
                engagements_150d_prior += engagement

        raw_hype_ratio = (engagements_30d + 1.0) / (engagements_150d_prior + 1.0)
        if late_stage_engagements:
            late_stage_avg = float(np.mean(late_stage_engagements))
            if late_stage_avg >= SUPERSTAR_ENGAGEMENT_FLOOR:
                raw_hype_ratio = max(1.0, raw_hype_ratio)

        features["RAW_IG_LATE_STAGE_HYPE"] = float(raw_hype_ratio)
        features["IG_LATE_STAGE_HYPE"] = min(float(raw_hype_ratio), 10.0)

    total_tt_plays = sum(safe_float(video.get("play_count")) for video in valid_tt_videos)
    features["TOTAL_PRE_RELEASE_TT_PLAYS"] = total_tt_plays

    if valid_tt_videos:
        total_tt_shares = sum(safe_float(video.get("share_count")) for video in valid_tt_videos)
        features["TT_SHARE_VELOCITY"] = total_tt_shares / max(total_tt_plays, 1.0)

        per_video_reach_ratios = [
            safe_float(video.get("play_count"))
            / max(safe_float(video.get("user_follower_count")), 1.0)
            for video in valid_tt_videos
        ]
        features["TT_OUTLIER_REACH"] = float(max(per_video_reach_ratios))

    return features


def process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Process one CSV chunk into engineered social features."""
    chunk = chunk.copy()
    chunk["FIRST_SALE_DATE"] = pd.to_datetime(chunk["FIRST_SALE_DATE"], errors="coerce")
    records = [extract_social_features(row) for row in chunk.itertuples(index=False)]
    return pd.DataFrame(records)


def main() -> None:
    print(
        "WARNING: build_social_features.py is deprecated and ignored by the production pipeline."
    )
    print("Reverting to pre-social baseline — no social_features.parquet will be consumed.")
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    print(f"Loading social timelines from {INPUT_PATH}")
    print(f"Chunk size: {CHUNK_SIZE:,} rows")
    start_time = time.time()

    processed_chunks: list[pd.DataFrame] = []
    total_rows = 0

    for chunk_idx, chunk in enumerate(
        pd.read_csv(INPUT_PATH, chunksize=CHUNK_SIZE),
        start=1,
    ):
        chunk.columns = chunk.columns.str.upper()
        chunk_start = time.time()
        feature_chunk = process_chunk(chunk)
        processed_chunks.append(feature_chunk)
        total_rows += len(chunk)
        chunk_elapsed = time.time() - chunk_start
        elapsed = time.time() - start_time

        print(
            f"  Chunk {chunk_idx:,}: processed {len(chunk):,} rows "
            f"({chunk_elapsed:.1f}s chunk, {elapsed:.1f}s total, {total_rows:,} cumulative)"
        )

    if not processed_chunks:
        print("No rows found in input file.")
        return

    social_features = pd.concat(processed_chunks, ignore_index=True)
    social_features = social_features[["MRELG_ID", *FEATURE_COLUMNS]]

    print(f"\nFinal feature matrix: {social_features.shape[0]:,} rows x {social_features.shape[1]} columns")
    print("\nFeature summary:")
    for col in FEATURE_COLUMNS:
        non_zero = int((social_features[col] != 0).sum())
        print(f"  {col}: non-zero={non_zero:,}, mean={social_features[col].mean():.4f}")

    print(f"\nSaving to {OUTPUT_PATH}...")
    social_features.to_parquet(OUTPUT_PATH, index=False)

    total_elapsed = time.time() - start_time
    print(f"Done in {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min).")


if __name__ == "__main__":
    main()
