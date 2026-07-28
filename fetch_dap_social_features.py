"""Fetch / rebuild DAP pre-release social features from Snowflake.

Training extract destination: data/dap_social_pre_release.csv
Source rebuild: US_LABELS_SANDBOX.RONAN_N.DAP_SOCIAL_PRE_RELEASE

Cutoff rule per album: LEAST(first_sale_date, CURRENT_DATE())
  - past releases: social mass stops at release date
  - future releases: social mass stops at today
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dap_social_features import (
    DAP_SOCIAL_FETCH_COLUMNS,
    DAP_SOCIAL_REBUILD_SQL,
    DAP_SOCIAL_SOURCE_TABLE,
)
from snowflake_client import get_dap_snowflake_connection, get_snowflake_connection

DATA_DIR = Path("data")
DEFAULT_OUTPUT_PATH = DATA_DIR / "dap_social_pre_release.csv"

SELECT_SQL = f"""
SELECT
    {", ".join(DAP_SOCIAL_FETCH_COLUMNS)}
FROM {DAP_SOCIAL_SOURCE_TABLE}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild DAP_SOCIAL_PRE_RELEASE from FACT_SOCIAL (date-anchored) "
            "and optionally export a local CSV."
        )
    )
    parser.add_argument(
        "--rebuild-table",
        action="store_true",
        help=(
            "Run CREATE OR REPLACE on US_LABELS_SANDBOX.RONAN_N.DAP_SOCIAL_PRE_RELEASE "
            "using LEAST(first_sale_date, CURRENT_DATE()) cutoffs."
        ),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Destination CSV path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--skip-csv",
        action="store_true",
        help="Rebuild the Snowflake table only; do not write a local CSV.",
    )
    return parser.parse_args()


def rebuild_dap_social_table() -> None:
    """Materialize the date-anchored DAP social training table in Snowflake.

    Uses SSO via secrets/amg_research_password.env so both DF_PROD_DAP_MISC
    (TikTok) and DF_PROD (YouTube + DIM_ARTIST) are reachable; the CREATE OR
    REPLACE target remains the fully-qualified sandbox table.
    """
    connection = get_dap_snowflake_connection()
    try:
        cursor = connection.cursor()
        print(f"Rebuilding {DAP_SOCIAL_SOURCE_TABLE} with proper date cutoffs ...")
        cursor.execute(DAP_SOCIAL_REBUILD_SQL)
        print("Rebuild complete.")
    finally:
        connection.close()


def fetch_dap_social_pre_release() -> pd.DataFrame:
    """Query the DAP social pre-release table into a DataFrame."""
    connection = get_snowflake_connection()
    try:
        cursor = connection.cursor()
        print(f"Querying {DAP_SOCIAL_SOURCE_TABLE} ...")
        cursor.execute(SELECT_SQL)
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
    finally:
        connection.close()

    df = pd.DataFrame(rows, columns=columns)
    df.columns = df.columns.str.upper()
    return df


def main() -> None:
    args = parse_args()

    if args.rebuild_table:
        rebuild_dap_social_table()

    if args.skip_csv:
        return

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
