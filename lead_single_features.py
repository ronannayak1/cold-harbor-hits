"""Live lead-single feature extraction for ML single-shot inference.

Mirrors the training `album_artist_lead_singles` export:
- Singles with Main Artist credit for the target artist
- Released in the 6 months before the album release date
- Only singles whose *next* qualifying album is the target release
  (training rn=1 assignment)
- W1/W2 OnDemand Audio streams (country AA), with W1 matching the
  training definition (all streams in the first 14 days under the outer
  date filter; W2 = days 7–14)

Leakage guards:
- Single first_sale_date < album release date
- Single first_sale_date <= anchor_date (= min(release, today))
- Stream report_date <= anchor_date
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from snowflake_client import get_snowflake_connection

# Candidate singles in the 6-month window + W1/W2 streams.
LIVE_LEAD_SINGLES_SQL = """
WITH target AS (
    SELECT
        %(artist_id)s::STRING AS artist_id,
        %(release_date)s::DATE AS album_release_date,
        %(anchor_date)s::DATE AS anchor_date
),
target_singles AS (
    SELECT
        mrg.mrelg_id,
        mrg.title,
        mrg.display_artist,
        MIN(mrg.first_sale_date) AS first_sale_date
    FROM luminate_prod.extract_s.vw_musical_release_group_ds mrg
    JOIN luminate_prod.extract_s.vw_mrel_mrelg_map_ds ml
        ON ml.mrelg_id = mrg.mrelg_id
    JOIN luminate_prod.extract_s.vw_mp_mrel_map_ds mp
        ON mp.mrel_id = ml.mrel_id
    JOIN current_dev.data.marketshare_map_icpns i
        ON i.mp_id = mp.mp_id,
    LATERAL FLATTEN(input => mrg.artists) a,
    target t
    WHERE mrg.release_type = 'Single'
        AND a.value:ARTIST_ID::STRING = t.artist_id
        AND a.value:ROLE::STRING = 'Main Artist'
        AND mrg.first_sale_date >= '2018-01-01'
        AND mrg.display_artist IS NOT NULL
        AND TRIM(UPPER(mrg.display_artist)) NOT IN (
            'VARIOUS ARTISTS', 'UNKNOWN', 'VARIOS ARTISTAS', 'VARIOUS'
        )
        AND mrg.first_sale_date >= DATEADD(MONTH, -6, t.album_release_date)
        AND mrg.first_sale_date < t.album_release_date
        AND mrg.first_sale_date <= t.anchor_date
    GROUP BY ALL
)
SELECT
    ts.mrelg_id AS MRELG_ID,
    ts.title AS TITLE,
    ts.display_artist AS DISPLAY_ARTIST,
    ts.first_sale_date AS FIRST_SALE_DATE,
    -- Training-compatible W1: all OnDemand Audio in the outer 14-day window
    COALESCE(SUM(IFF(
        s.metric_category = 'Streams'
        AND s.service_type = 'OnDemand'
        AND s.content_type = 'Audio',
        s.quantity, 0
    )), 0) AS SINGLE_W1_AUDIO_STREAMS,
    COALESCE(SUM(IFF(
        s.metric_category = 'Streams'
        AND s.service_type = 'OnDemand'
        AND s.content_type = 'Audio'
        AND s.report_date >= DATEADD(DAY, 7, ts.first_sale_date)
        AND s.report_date < DATEADD(DAY, 14, ts.first_sale_date),
        s.quantity, 0
    )), 0) AS SINGLE_W2_AUDIO_STREAMS
FROM target_singles ts
CROSS JOIN target t
JOIN luminate_prod.extract_s.vw_daily_fact_mrelg_summary_ds s
    ON s.mrelg_id = ts.mrelg_id
WHERE s.country_code = 'AA'
    AND s.report_date >= ts.first_sale_date
    AND s.report_date < DATEADD(DAY, 14, ts.first_sale_date)
    AND s.report_date <= t.anchor_date
GROUP BY ALL
ORDER BY ts.first_sale_date ASC, SINGLE_W1_AUDIO_STREAMS DESC
"""

# Lightweight album release dates for this artist (training valid_albums filters).
LIVE_ARTIST_ALBUMS_SQL = """
WITH artist_albums AS (
    SELECT
        mrg.mrelg_id,
        MIN(mrg.first_sale_date) AS first_sale_date
    FROM luminate_prod.extract_s.vw_musical_release_group_ds mrg,
    LATERAL FLATTEN(input => mrg.artists) a
    WHERE mrg.release_type = 'Album'
        AND a.value:ARTIST_ID::STRING = %(artist_id)s
        AND a.value:ROLE::STRING = 'Main Artist'
        AND mrg.display_artist IS NOT NULL
        AND TRIM(UPPER(mrg.display_artist)) NOT IN (
            'VARIOUS ARTISTS', 'UNKNOWN', 'VARIOS ARTISTAS', 'VARIOUS'
        )
        AND UPPER(mrg.title) NOT LIKE '%%SOUNDTRACK%%'
        AND UPPER(mrg.title) NOT LIKE '%%OST%%'
        AND UPPER(mrg.title) NOT LIKE '%%LIVE%%'
        AND UPPER(mrg.title) NOT LIKE '%%REMIX%%'
        AND UPPER(mrg.title) NOT LIKE '%%ACOUSTIC%%'
        AND UPPER(mrg.title) NOT LIKE '%%INSTRUMENTAL%%'
    GROUP BY mrg.mrelg_id
)
SELECT
    aa.mrelg_id AS MRELG_ID,
    aa.first_sale_date AS FIRST_SALE_DATE
FROM artist_albums aa
JOIN luminate_prod.extract_s.vw_mrel_mrelg_map_ds ml
    ON ml.mrelg_id = aa.mrelg_id
GROUP BY aa.mrelg_id, aa.first_sale_date
HAVING COUNT(DISTINCT ml.mrel_id) >= 5
ORDER BY aa.first_sale_date
"""


def _query_df(sql: str, params: dict[str, Any]) -> pd.DataFrame:
    connection = get_snowflake_connection()
    try:
        cursor = connection.cursor()
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
    finally:
        connection.close()
    frame = pd.DataFrame(rows, columns=columns)
    if not frame.empty:
        frame.columns = frame.columns.str.upper()
    return frame


def filter_next_album_singles(
    singles_df: pd.DataFrame,
    albums_df: pd.DataFrame,
    release_date: str,
) -> pd.DataFrame:
    """Keep singles whose next qualifying album date equals the target release.

    Matches training rn=1 assignment: drop singles that already have an
    intervening album between the single and the target release date.
    """
    if singles_df.empty:
        return singles_df

    release_ts = pd.Timestamp(release_date)
    album_dates = (
        pd.to_datetime(albums_df["FIRST_SALE_DATE"], errors="coerce").dropna().sort_values()
        if not albums_df.empty
        else pd.Series(dtype="datetime64[ns]")
    )

    keep_mask = []
    for single_date in pd.to_datetime(singles_df["FIRST_SALE_DATE"], errors="coerce"):
        if pd.isna(single_date):
            keep_mask.append(False)
            continue
        intervening = album_dates[(album_dates > single_date) & (album_dates < release_ts)]
        keep_mask.append(intervening.empty)

    return singles_df.loc[keep_mask].reset_index(drop=True)


def aggregate_lead_single_metrics(singles_df: pd.DataFrame) -> dict[str, Any]:
    """Aggregate per-single W1/W2 rows into model lead-single features."""
    if singles_df.empty:
        return {
            "LEAD_SINGLE_PEAK_VOLUME": 0.0,
            "ACTIVE_SINGLE_COUNT": 0,
            "RETENTION_RATIO": np.nan,
            "lead_singles": [],
        }

    work = singles_df.copy()
    work.columns = work.columns.str.upper()
    w1 = work["SINGLE_W1_AUDIO_STREAMS"].astype(float)
    w2 = work["SINGLE_W2_AUDIO_STREAMS"].astype(float)
    peak = float(w1.sum())
    count = int(len(work))
    retention = float(w2.sum() / peak) if peak > 0 else np.nan

    lead_singles = [
        {
            "mrelg_id": str(row["MRELG_ID"]),
            "title": str(row.get("TITLE", "")),
            "first_sale_date": str(row["FIRST_SALE_DATE"])[:10],
            "w1": float(row["SINGLE_W1_AUDIO_STREAMS"]),
            "w2": float(row["SINGLE_W2_AUDIO_STREAMS"]),
        }
        for _, row in work.iterrows()
    ]

    return {
        "LEAD_SINGLE_PEAK_VOLUME": peak,
        "ACTIVE_SINGLE_COUNT": count,
        "RETENTION_RATIO": retention,
        "lead_singles": lead_singles,
    }


def fetch_live_lead_singles(
    artist_id: str,
    release_date: str,
    anchor_date: str,
) -> dict[str, Any]:
    """Pull lead singles for an artist in the 6 months before release_date.

    ``anchor_date`` should be ``min(release_date, today)`` and caps both
    eligible singles and stream fact dates (leakage guard).
    """
    params = {
        "artist_id": artist_id,
        "release_date": release_date,
        "anchor_date": anchor_date,
    }
    singles_df = _query_df(LIVE_LEAD_SINGLES_SQL, params)
    albums_df = _query_df(LIVE_ARTIST_ALBUMS_SQL, {"artist_id": artist_id})
    filtered = filter_next_album_singles(singles_df, albums_df, release_date)
    return aggregate_lead_single_metrics(filtered)
