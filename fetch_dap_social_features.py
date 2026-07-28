"""Fetch DAP pre-release social features from Snowflake for Stage 4 training.

Replaces the legacy Chartex extract (`data/chartex_velocity_training.csv`) with:
  US_LABELS_SANDBOX.RONAN_N.DAP_SOCIAL_PRE_RELEASE

Auth: secrets/amg_research.env (SSO externalbrowser supported).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dap_social_features import DAP_SOCIAL_FETCH_COLUMNS, DAP_SOCIAL_SOURCE_TABLE
from snowflake_client import get_snowflake_connection

DATA_DIR = Path("data")
DEFAULT_OUTPUT_PATH = DATA_DIR / "dap_social_pre_release.csv"

FETCH_SQL = f"""
SELECT
    {", ".join(DAP_SOCIAL_FETCH_COLUMNS)}
FROM {DAP_SOCIAL_SOURCE_TABLE}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull DAP_SOCIAL_PRE_RELEASE from Snowflake into a local CSV."
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Destination CSV path (default: {DEFAULT_OUTPUT_PATH})",
    )
    return parser.parse_args()


def fetch_dap_social_pre_release() -> pd.DataFrame:
    """Query the full DAP social pre-release table into a DataFrame."""
    connection = get_snowflake_connection()
    try:
        cursor = connection.cursor()
        print(f"Querying {DAP_SOCIAL_SOURCE_TABLE} ...")
        cursor.execute(FETCH_SQL)
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
    finally:
        connection.close()

    df = pd.DataFrame(rows, columns=columns)
    df.columns = df.columns.str.upper()
    return df


def main() -> None:
    args = parse_args()
    df = fetch_dap_social_pre_release()

    missing = [col for col in DAP_SOCIAL_FETCH_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Fetched frame is missing expected columns: {missing}")

    df = df[DAP_SOCIAL_FETCH_COLUMNS].copy()
    if "FIRST_SALE_DATE" in df.columns:
        df["FIRST_SALE_DATE"] = pd.to_datetime(df["FIRST_SALE_DATE"], errors="coerce")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_file, index=False)

    print(f"Rows fetched:     {len(df):,}")
    print(f"Columns:          {len(df.columns)}")
    if df["FIRST_SALE_DATE"].notna().any():
        print(
            f"FIRST_SALE_DATE:  {df['FIRST_SALE_DATE'].min().date()} → "
            f"{df['FIRST_SALE_DATE'].max().date()}"
        )
    print(f"Saved to:         {args.output_file}")


if __name__ == "__main__":
    main()
