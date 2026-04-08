"""
Build a normalised DuckDB schema from cleaned Parquet files.

Schema:
    dim_lines       – one row per metro line
    dim_stations    – one row per station (FK → dim_lines)
    dim_date        – date spine with time attributes
    fact_ridership  – ridership counts (FK → dim_stations, dim_date)

Usage:
    python schema.py          # build / refresh the whole schema
"""

import logging
from pathlib import Path

import duckdb
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import DATA_PROCESSED, DB_PATH

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def get_connection() -> duckdb.DuckDBPyConnection:
    """Return a persistent DuckDB connection."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(DB_PATH))


def build_schema(con: duckdb.DuckDBPyConnection | None = None) -> duckdb.DuckDBPyConnection:
    """
    Create or replace all schema objects.

    Returns the open connection so callers can run further queries.
    """
    if con is None:
        con = get_connection()

    ridership_parquet = str(DATA_PROCESSED / "ridership_clean.parquet")
    stops_parquet = str(DATA_PROCESSED / "stops_clean.parquet")
    routes_parquet = str(DATA_PROCESSED / "routes_clean.parquet")

    logger.info("Building DuckDB schema in %s", DB_PATH)

    # ------------------------------------------------------------------
    # dim_lines
    # ------------------------------------------------------------------
    con.execute("""
        CREATE OR REPLACE TABLE dim_lines AS
        SELECT DISTINCT
            line_id,
            line_name,
        FROM read_parquet($1)
        ORDER BY line_id
    """, [ridership_parquet])
    logger.info("  dim_lines: %d rows", con.execute("SELECT COUNT(*) FROM dim_lines").fetchone()[0])

    # ------------------------------------------------------------------
    # dim_stations
    # ------------------------------------------------------------------
    con.execute("""
        CREATE OR REPLACE TABLE dim_stations AS
        SELECT DISTINCT
            r.station_id,
            r.station_name,
            r.line_id,
            s.stop_lat,
            s.stop_lon,
        FROM read_parquet($1) AS r
        LEFT JOIN read_parquet($2) AS s
            ON r.station_id = s.stop_id
        ORDER BY r.line_id, r.station_id
    """, [ridership_parquet, stops_parquet])
    logger.info("  dim_stations: %d rows",
                con.execute("SELECT COUNT(*) FROM dim_stations").fetchone()[0])

    # ------------------------------------------------------------------
    # dim_date  (date spine)
    # ------------------------------------------------------------------
    con.execute("""
        CREATE OR REPLACE TABLE dim_date AS
        SELECT DISTINCT
            date,
            year,
            month,
            week,
            day_of_week,
            is_weekend,
            season,
            CASE
                WHEN date BETWEEN '2020-03-13' AND '2021-12-31' THEN 'COVID'
                WHEN date < '2020-03-13' THEN 'Pre-COVID'
                ELSE 'Post-COVID'
            END AS covid_era,
        FROM read_parquet($1)
        ORDER BY date
    """, [ridership_parquet])
    logger.info("  dim_date: %d rows",
                con.execute("SELECT COUNT(*) FROM dim_date").fetchone()[0])

    # ------------------------------------------------------------------
    # fact_ridership
    # ------------------------------------------------------------------
    con.execute("""
        CREATE OR REPLACE TABLE fact_ridership AS
        SELECT
            r.date,
            r.station_id,
            r.line_id,
            r.period,
            r.is_peak,
            r.ridership,
        FROM read_parquet($1) AS r
        ORDER BY r.date, r.line_id, r.station_id
    """, [ridership_parquet])
    logger.info("  fact_ridership: %d rows",
                con.execute("SELECT COUNT(*) FROM fact_ridership").fetchone()[0])

    return con


def export_dim_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Export dimension and fact tables to CSV for dashboard consumption."""
    from config.settings import DATA_OUTPUT
    DATA_OUTPUT.mkdir(parents=True, exist_ok=True)

    tables = ["dim_lines", "dim_stations", "dim_date", "fact_ridership"]
    for tbl in tables:
        out = DATA_OUTPUT / f"{tbl}.csv"
        con.execute(f"COPY {tbl} TO '{out}' (HEADER, DELIMITER ',')")
        count = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        logger.info("  Exported %s → %s (%d rows)", tbl, out, count)


if __name__ == "__main__":
    con = build_schema()
    export_dim_tables(con)
    con.close()
    print("Schema build complete.")
