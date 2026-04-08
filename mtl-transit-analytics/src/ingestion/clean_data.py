"""
Clean and standardise raw STM ridership and GTFS data.

Outputs cleaned Parquet files to data/processed/.
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import DATA_RAW, DATA_PROCESSED

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ridership cleaning
# ---------------------------------------------------------------------------

def clean_ridership(src: Path | None = None) -> pd.DataFrame:
    """
    Load, validate, and clean the raw ridership CSV.

    Steps:
        1. Parse dates.
        2. Normalise column names (snake_case).
        3. Drop duplicates and rows with null ridership.
        4. Clip negative ridership to 0.
        5. Add derived columns: year, month, week, day_of_week, is_weekend,
           is_peak, season.

    Returns the cleaned DataFrame and saves it to data/processed/ridership_clean.parquet.
    """
    if src is None:
        src = DATA_RAW / "ridership_metro.csv"

    logger.info("Loading raw ridership from %s", src)
    df = pd.read_csv(src, parse_dates=["date"], dtype={"line_id": str})

    # Normalise column names
    df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]

    # Ensure required columns exist
    required = {"date", "line_id", "line_name", "station_id", "station_name", "period", "ridership"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in ridership CSV: {missing}")

    # Basic quality gates
    before = len(df)
    df = df.drop_duplicates()
    df = df.dropna(subset=["date", "ridership"])
    df["ridership"] = df["ridership"].clip(lower=0).astype(int)
    logger.info("Dropped %d rows during cleaning (kept %d)", before - len(df), len(df))

    # Derived time columns
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["day_of_week"] = df["date"].dt.dayofweek       # 0 = Monday
    df["is_weekend"] = df["day_of_week"] >= 5

    # Peak classification
    peak_periods = {"Heure de pointe AM", "Heure de pointe PM"}
    df["is_peak"] = df["period"].isin(peak_periods)

    # Season
    df["season"] = df["month"].map(_month_to_season)

    # Save
    out = DATA_PROCESSED / "ridership_clean.parquet"
    df.to_parquet(out, index=False)
    logger.info("Clean ridership saved to %s (%d rows)", out, len(df))
    return df


def _month_to_season(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    elif month in (3, 4, 5):
        return "Spring"
    elif month in (6, 7, 8):
        return "Summer"
    else:
        return "Fall"


# ---------------------------------------------------------------------------
# GTFS stops cleaning
# ---------------------------------------------------------------------------

def clean_gtfs_stops(gtfs_dir: Path | None = None) -> pd.DataFrame:
    """
    Load and clean GTFS stops.txt.

    Returns cleaned stops DataFrame saved to data/processed/stops_clean.parquet.
    """
    if gtfs_dir is None:
        gtfs_dir = DATA_RAW / "gtfs"

    stops_file = gtfs_dir / "stops.txt"
    logger.info("Loading GTFS stops from %s", stops_file)
    df = pd.read_csv(stops_file, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]

    df["stop_lat"] = pd.to_numeric(df["stop_lat"], errors="coerce")
    df["stop_lon"] = pd.to_numeric(df["stop_lon"], errors="coerce")
    df = df.dropna(subset=["stop_lat", "stop_lon"])

    out = DATA_PROCESSED / "stops_clean.parquet"
    df.to_parquet(out, index=False)
    logger.info("Clean stops saved to %s (%d rows)", out, len(df))
    return df


def clean_gtfs_routes(gtfs_dir: Path | None = None) -> pd.DataFrame:
    """Load and clean GTFS routes.txt."""
    if gtfs_dir is None:
        gtfs_dir = DATA_RAW / "gtfs"

    routes_file = gtfs_dir / "routes.txt"
    logger.info("Loading GTFS routes from %s", routes_file)
    df = pd.read_csv(routes_file, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]

    out = DATA_PROCESSED / "routes_clean.parquet"
    df.to_parquet(out, index=False)
    logger.info("Clean routes saved to %s (%d rows)", out, len(df))
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_all() -> dict[str, pd.DataFrame]:
    """Run all cleaning pipelines and return DataFrames."""
    return {
        "ridership": clean_ridership(),
        "stops": clean_gtfs_stops(),
        "routes": clean_gtfs_routes(),
    }


if __name__ == "__main__":
    results = clean_all()
    for name, df in results.items():
        print(f"{name}: {df.shape}")
