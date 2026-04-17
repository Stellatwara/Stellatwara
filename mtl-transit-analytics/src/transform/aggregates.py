"""
Pre-compute aggregate tables for analysis and the dashboard.

Aggregates produced:
    agg_daily_line       – daily ridership per line
    agg_monthly_line     – monthly ridership per line
    agg_peak_summary     – peak vs off-peak ridership per line per month
    agg_seasonal         – seasonal ridership averages per line
    agg_station_rank     – station-level ridership totals (sorted)

All tables are written both into DuckDB and as CSVs in data/output/.
"""

import logging
from pathlib import Path

import duckdb

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import DATA_OUTPUT, DB_PATH

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


_AGGREGATES = {
    "agg_daily_line": """
        SELECT
            f.date,
            d.year,
            d.month,
            d.day_of_week,
            d.is_weekend,
            d.season,
            d.covid_era,
            f.line_id,
            l.line_name,
            SUM(f.ridership) AS total_ridership,
        FROM fact_ridership f
        JOIN dim_date d USING (date)
        JOIN dim_lines l USING (line_id)
        GROUP BY ALL
        ORDER BY f.date, f.line_id
    """,
    "agg_monthly_line": """
        SELECT
            d.year,
            d.month,
            d.covid_era,
            f.line_id,
            l.line_name,
            SUM(f.ridership)                   AS total_ridership,
            AVG(SUM(f.ridership)) OVER (
                PARTITION BY f.line_id
                ORDER BY d.year, d.month
                ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
            )                                  AS ridership_3m_avg,
        FROM fact_ridership f
        JOIN dim_date d USING (date)
        JOIN dim_lines l USING (line_id)
        GROUP BY d.year, d.month, d.covid_era, f.line_id, l.line_name
        ORDER BY d.year, d.month, f.line_id
    """,
    "agg_peak_summary": """
        SELECT
            d.year,
            d.month,
            f.line_id,
            l.line_name,
            f.is_peak,
            SUM(f.ridership) AS ridership,
        FROM fact_ridership f
        JOIN dim_date d USING (date)
        JOIN dim_lines l USING (line_id)
        GROUP BY d.year, d.month, f.line_id, l.line_name, f.is_peak
        ORDER BY d.year, d.month, f.line_id, f.is_peak
    """,
    "agg_seasonal": """
        SELECT
            d.year,
            d.season,
            f.line_id,
            l.line_name,
            AVG(f.ridership)   AS avg_daily_ridership,
            SUM(f.ridership)   AS total_ridership,
            COUNT(*)           AS observation_count,
        FROM fact_ridership f
        JOIN dim_date d USING (date)
        JOIN dim_lines l USING (line_id)
        GROUP BY d.year, d.season, f.line_id, l.line_name
        ORDER BY d.year, d.season, f.line_id
    """,
    "agg_station_rank": """
        SELECT
            s.station_id,
            s.station_name,
            s.line_id,
            l.line_name,
            s.stop_lat,
            s.stop_lon,
            SUM(f.ridership)   AS total_ridership,
            AVG(f.ridership)   AS avg_daily_ridership,
        FROM fact_ridership f
        JOIN dim_stations s USING (station_id)
        JOIN dim_lines l USING (line_id)
        GROUP BY s.station_id, s.station_name, s.line_id, l.line_name, s.stop_lat, s.stop_lon
        ORDER BY total_ridership DESC
    """,
}


def build_aggregates(con: duckdb.DuckDBPyConnection | None = None) -> duckdb.DuckDBPyConnection:
    """
    Build all aggregate tables in DuckDB and export to CSV.

    Parameters
    ----------
    con : open DuckDB connection (optional).  If None a new one is created.

    Returns
    -------
    The open DuckDB connection.
    """
    if con is None:
        con = duckdb.connect(str(DB_PATH))

    DATA_OUTPUT.mkdir(parents=True, exist_ok=True)

    for tbl, sql in _AGGREGATES.items():
        con.execute(f"CREATE OR REPLACE TABLE {tbl} AS {sql}")
        count = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        logger.info("  Built %-25s  %d rows", tbl, count)

        out = (DATA_OUTPUT / f"{tbl}.csv").as_posix()
        con.execute(f"COPY {tbl} TO '{out}' (HEADER, DELIMITER ',')")
        logger.info("    → exported to %s", out)

    return con


if __name__ == "__main__":
    con = build_aggregates()
    con.close()
    print("All aggregates built and exported.")
