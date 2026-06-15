"""Build album-level streaming features for Two-Stage LightGBM and A&R Sandbox lookup."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

DATA_DIR = Path("data")
BASE_TABLE_PATH = Path("/Users/ronannayak/WMG/cold-harbor-hits/data/spb_marketshare.csv")
Z_SCORE_HITS_PATH = DATA_DIR / "z_score_hits.csv"
LEAD_SINGLES_PATH = DATA_DIR / "lead_single_album_map_streams.csv"
CATALOG_STREAMS_PATH = DATA_DIR / "artist_catalog_streams_12w.csv"
FEATURE_MATRIX_PATH = DATA_DIR / "album_meta_features.parquet"
SANDBOX_BASELINES_PATH = DATA_DIR / "artist_sandbox_baselines.csv"
SOCIAL_FEATURES_PATH = DATA_DIR / "social_features.parquet"

SOCIAL_VOLUME_COLUMNS = [
    "TOTAL_PRE_RELEASE_IG_FAVES",
    "TOTAL_PRE_RELEASE_IG_COMMENTS",
    "TOTAL_PRE_RELEASE_TT_PLAYS",
]
SOCIAL_RATIO_COLUMNS = [
    "IG_AVG_ENGAGEMENT_RATE",
    "IG_COMMENT_DENSITY",
    "IG_LATE_STAGE_HYPE",
    "RAW_IG_LATE_STAGE_HYPE",
    "TT_SHARE_VELOCITY",
    "TT_OUTLIER_REACH",
]
SOCIAL_FEATURE_COLUMNS = SOCIAL_VOLUME_COLUMNS + SOCIAL_RATIO_COLUMNS


def load_csv_uppercase(path: Path) -> pd.DataFrame:
    """Load a CSV and normalize column names to uppercase."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.upper()
    return df


def compute_catalog_velocity(catalog_df: pd.DataFrame) -> pd.DataFrame:
    """Compute catalog_velocity_slope via linear regression per album."""
    def _slope_for_album(group: pd.DataFrame) -> float:
        # Sort ascending (e.g., 12 down to 0, which is oldest to newest chronologically)
        sorted_group = group.sort_values("WEEKS_PRIOR_TO_RELEASE", ascending=False)
        if len(sorted_group) < 2:
            return np.nan

        # Create a simple chronological time index (0, 1, 2, 3...)
        # This ensures a positive slope ALWAYS means chronological growth
        x = np.arange(len(sorted_group))
        y = sorted_group["WEEKLY_CATALOG_STREAMS"].values

        try:
            slope, _, _, _, _ = linregress(x, y)
        except ValueError:
            return np.nan
        return slope

    slopes = (
        catalog_df.groupby("ALBUM_MRELG_ID", group_keys=False)
        .apply(_slope_for_album)
        .rename("CATALOG_VELOCITY_SLOPE")
        .reset_index()
    )
    return slopes


def compute_short_term_spike(catalog_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the ratio of the most recent catalog week vs the previous 3 weeks."""
    def _spike_ratio(group: pd.DataFrame) -> float:
        group = group.sort_values("WEEKS_PRIOR_TO_RELEASE", ascending=True)
        if len(group) < 2:
            return 1.0

        w1_streams = group.iloc[0]["WEEKLY_CATALOG_STREAMS"]
        w2_w4_streams = group.iloc[1:4]["WEEKLY_CATALOG_STREAMS"].mean()

        if w2_w4_streams == 0:
            return 1.0 if w1_streams == 0 else 2.0

        ratio = w1_streams / w2_w4_streams
        return float(np.clip(ratio, 0.5, 4.0))

    spikes = (
        catalog_df.groupby("ALBUM_MRELG_ID", group_keys=False)
        .apply(_spike_ratio)
        .rename("SHORT_TERM_SPIKE_RATIO")
        .reset_index()
    )
    return spikes


def compute_lead_single_features(singles_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate lead-single hype, retention, and active single count per album."""
    agg = (
        singles_df.groupby("MRELG_ID_ALBUM")
        .agg(
            LEAD_SINGLE_PEAK_VOLUME=("SINGLE_W1_AUDIO_STREAMS", "sum"),
            SINGLE_W2_SUM=("SINGLE_W2_AUDIO_STREAMS", "sum"),
            ACTIVE_SINGLE_COUNT=("SINGLE_W1_AUDIO_STREAMS", "count"),
        )
        .reset_index()
    )

    agg["RETENTION_RATIO"] = np.where(
        agg["LEAD_SINGLE_PEAK_VOLUME"] == 0,
        np.nan,
        agg["SINGLE_W2_SUM"] / agg["LEAD_SINGLE_PEAK_VOLUME"],
    )
    return agg.drop(columns=["SINGLE_W2_SUM"])


def compute_historical_features(base_df: pd.DataFrame, z_score_df: pd.DataFrame) -> pd.DataFrame:
    """Engineer artist-level historical baselines and macro-momentum (leakage-safe)."""
    z_score_cols = ["MRELG_ID", "HITS_1_5_SD"]
    if "ALBUM_STANDARD_TRACK_SPB" in z_score_df.columns:
        z_score_cols.append("ALBUM_STANDARD_TRACK_SPB")
    z_score_subset = (
        z_score_df[z_score_cols]
        .drop_duplicates(subset="MRELG_ID", keep="last")
    )
    merged = base_df.merge(
        z_score_subset,
        on="MRELG_ID",
        how="left",
        suffixes=("", "_ZSCORE"),
    )
    if "ALBUM_STANDARD_TRACK_SPB_ZSCORE" in merged.columns:
        merged["ALBUM_STANDARD_TRACK_SPB"] = merged["ALBUM_STANDARD_TRACK_SPB"].fillna(
            merged["ALBUM_STANDARD_TRACK_SPB_ZSCORE"]
        )
        merged = merged.drop(columns=["ALBUM_STANDARD_TRACK_SPB_ZSCORE"])
    merged["FIRST_SALE_DATE"] = pd.to_datetime(merged["FIRST_SALE_DATE"])
    merged = merged.sort_values(by=["ARTIST_ID", "FIRST_SALE_DATE"]).reset_index(drop=True)

    merged["ARTIST_HISTORICAL_HIT_AVG"] = (
        merged.groupby("ARTIST_ID")["HITS_1_5_SD"]
        .transform(lambda s: s.expanding().mean().shift(1))
    )
    merged["HISTORICAL_STANDARD_TRACK_SPB"] = (
        merged.groupby("ARTIST_ID")["ALBUM_STANDARD_TRACK_SPB"]
        .transform(lambda s: s.ewm(alpha=0.5, adjust=False).mean().shift(1))
    )
    growth = merged.groupby("ARTIST_ID")["ALBUM_STANDARD_TRACK_SPB"].pct_change()
    merged["ALBUM_OVER_ALBUM_GROWTH"] = growth.replace([np.inf, -np.inf], np.nan)
    merged["HISTORICAL_MACRO_MOMENTUM"] = (
        merged.groupby("ARTIST_ID")["ALBUM_OVER_ALBUM_GROWTH"].shift(1)
    )
    return merged


def build_feature_matrix(
    base_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    singles_features: pd.DataFrame,
    velocity_features: pd.DataFrame,
    spike_features: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join all engineered features onto the base album table."""
    features = historical_df.merge(
        singles_features,
        left_on="MRELG_ID",
        right_on="MRELG_ID_ALBUM",
        how="left",
    ).merge(
        velocity_features,
        left_on="MRELG_ID",
        right_on="ALBUM_MRELG_ID",
        how="left",
    ).merge(
        spike_features,
        left_on="MRELG_ID",
        right_on="ALBUM_MRELG_ID",
        how="left",
    )

    features["LEAD_SINGLE_PEAK_VOLUME"] = features["LEAD_SINGLE_PEAK_VOLUME"].fillna(0)
    features["ACTIVE_SINGLE_COUNT"] = features["ACTIVE_SINGLE_COUNT"].fillna(0)
    features["SHORT_TERM_SPIKE_RATIO"] = features["SHORT_TERM_SPIKE_RATIO"].fillna(1.0)
    features["IS_DEBUT_ALBUM"] = (
        features["HISTORICAL_STANDARD_TRACK_SPB"].isna() | 
        (features["HISTORICAL_STANDARD_TRACK_SPB"] <= 0)
    ).astype(int)
    features["VELOCITY_X_SINGLES"] = (
        features["CATALOG_VELOCITY_SLOPE"].fillna(0) * features["ACTIVE_SINGLE_COUNT"]
    )
    return features


def merge_social_features(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    """Left-join social velocity features and apply targeted imputation for missing albums."""
    if not SOCIAL_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Social features not found: {SOCIAL_FEATURES_PATH}. Run build_social_features.py first."
        )

    social_df = pd.read_parquet(SOCIAL_FEATURES_PATH)
    if "MRELG_ID" not in social_df.columns:
        raise ValueError("social_features.parquet is missing required join key: MRELG_ID")

    missing_social_cols = set(SOCIAL_FEATURE_COLUMNS) - set(social_df.columns)
    if missing_social_cols:
        raise ValueError(
            f"social_features.parquet is missing required columns: {sorted(missing_social_cols)}"
        )

    social_df = (
        social_df[["MRELG_ID", *SOCIAL_FEATURE_COLUMNS]]
        .drop_duplicates(subset="MRELG_ID", keep="last")
    )

    merged = feature_matrix.merge(social_df, on="MRELG_ID", how="left")

    imputation_map = {col: 0.0 for col in SOCIAL_VOLUME_COLUMNS}
    imputation_map.update({col: 1.0 for col in SOCIAL_RATIO_COLUMNS})
    return merged.fillna(imputation_map)


def build_sandbox_baselines(
    feature_matrix: pd.DataFrame, lambda_decay: float = 0.5
) -> pd.DataFrame:
    """Build per-artist Sandbox lookup using time-decay adjusted for macro-momentum."""
    df = feature_matrix.copy()

    df["YEARS_ELAPSED"] = (
        pd.to_datetime("today") - pd.to_datetime(df["FIRST_SALE_DATE"])
    ).dt.days / 365.25
    df["WEIGHT"] = np.exp(-lambda_decay * df["YEARS_ELAPSED"])

    def calculate_weighted_avg(group: pd.DataFrame, col_name: str) -> float:
        weight_sum = group["WEIGHT"].sum()
        if weight_sum == 0:
            return group[col_name].iloc[-1]
        return np.average(group[col_name], weights=group["WEIGHT"])

    df = df.sort_values("FIRST_SALE_DATE", ascending=True)

    baselines = df.groupby(["ARTIST_ID", "DISPLAY_ARTIST"]).apply(
        lambda g: pd.Series(
            {
                "decayed_standard_track_spb": calculate_weighted_avg(
                    g, "ALBUM_STANDARD_TRACK_SPB"
                ),
                "decayed_hit_rate": calculate_weighted_avg(g, "HITS_1_5_SD"),
                "latest_album_growth": (
                    g["ALBUM_OVER_ALBUM_GROWTH"].iloc[-1]
                    if not g["ALBUM_OVER_ALBUM_GROWTH"].isna().all()
                    else 0.0
                ),
            }
        )
    ).reset_index()

    baselines["latest_album_growth_capped"] = baselines["latest_album_growth"].clip(
        lower=-0.5, upper=0.5
    )
    baselines["SANDBOX_STANDARD_TRACK_SPB"] = baselines["decayed_standard_track_spb"] * (
        1 + baselines["latest_album_growth_capped"]
    )
    baselines["SANDBOX_STANDARD_TRACK_SPB"] = baselines["SANDBOX_STANDARD_TRACK_SPB"].round(2)

    return baselines[
        ["ARTIST_ID", "DISPLAY_ARTIST", "SANDBOX_STANDARD_TRACK_SPB", "decayed_hit_rate"]
    ]


def summarize_dataframe(df: pd.DataFrame, label: str) -> None:
    """Print shape and null counts for key engineered columns."""
    print(f"\n{'=' * 60}")
    print(f"{label}")
    print(f"{'=' * 60}")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    key_cols = [
        "CATALOG_VELOCITY_SLOPE",
        "LEAD_SINGLE_PEAK_VOLUME",
        "RETENTION_RATIO",
        "ACTIVE_SINGLE_COUNT",
        "VELOCITY_X_SINGLES",
        "ARTIST_HISTORICAL_HIT_AVG",
        "HISTORICAL_STANDARD_TRACK_SPB",
        "ALBUM_OVER_ALBUM_GROWTH",
        "HISTORICAL_MACRO_MOMENTUM",
        "IS_DEBUT_ALBUM",
        "SHORT_TERM_SPIKE_RATIO",
        *SOCIAL_FEATURE_COLUMNS,
    ]
    present_cols = [c for c in key_cols if c in df.columns]
    if present_cols:
        null_counts = df[present_cols].isna().sum()
        print("\nNull counts (engineered features):")
        for col, count in null_counts.items():
            pct = 100 * count / len(df) if len(df) else 0
            print(f"  {col}: {count:,} ({pct:.1f}%)")


def main() -> None:
    print("Loading raw data pulls...")
    base_df = load_csv_uppercase(BASE_TABLE_PATH)
    z_score_df = load_csv_uppercase(Z_SCORE_HITS_PATH)
    singles_df = load_csv_uppercase(LEAD_SINGLES_PATH)
    catalog_df = load_csv_uppercase(CATALOG_STREAMS_PATH)

    print(f"  Base table:        {base_df.shape}")
    print(f"  Z-score hits:      {z_score_df.shape}")
    print(f"  Lead singles:      {singles_df.shape}")
    print(f"  Catalog streams:   {catalog_df.shape}")

    print("\nEngineering features...")
    velocity_features = compute_catalog_velocity(catalog_df)
    spike_features = compute_short_term_spike(catalog_df)
    singles_features = compute_lead_single_features(singles_df)
    historical_df = compute_historical_features(base_df, z_score_df)
    feature_matrix = build_feature_matrix(
        base_df, historical_df, singles_features, velocity_features, spike_features
    )

    print("\nMerging social velocity features...")
    feature_matrix = merge_social_features(feature_matrix)
    print(f"  Social features joined: {len(SOCIAL_FEATURE_COLUMNS)} columns")

    summarize_dataframe(feature_matrix, "Album Meta Feature Matrix")

    print(f"\nSaving feature matrix to {FEATURE_MATRIX_PATH}...")
    feature_matrix.to_parquet(FEATURE_MATRIX_PATH, index=False)

    sandbox_baselines = build_sandbox_baselines(feature_matrix)
    summarize_dataframe(sandbox_baselines, "A&R Sandbox Baselines")

    print(f"\nSaving sandbox baselines to {SANDBOX_BASELINES_PATH}...")
    sandbox_baselines.to_csv(SANDBOX_BASELINES_PATH, index=False)

    print("\nDone.")


if __name__ == "__main__":
    main()
