"""
Compute and print summary statistics for the transit dataset.
"""

import logging
from pathlib import Path

import duckdb
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import DB_PATH, DATA_OUTPUT

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def overall_summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """High-level ridership summary across all lines and years."""
    return con.execute("""
        SELECT
            year,
            SUM(total_ridership)   AS annual_ridership,
            AVG(total_ridership)   AS avg_monthly_ridership,
            MIN(total_ridership)   AS min_monthly_ridership,
            MAX(total_ridership)   AS max_monthly_ridership,
        FROM agg_monthly_line
        GROUP BY year
        ORDER BY year
    """).df()


def line_summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Per-line ridership totals and peak share."""
    return con.execute("""
        SELECT
            f.line_id,
            l.line_name,
            SUM(f.ridership)                                          AS total_ridership,
            SUM(CASE WHEN f.is_peak THEN f.ridership ELSE 0 END) * 1.0 /
                NULLIF(SUM(f.ridership), 0)                           AS peak_share,
        FROM fact_ridership f
        JOIN dim_lines l USING (line_id)
        GROUP BY f.line_id, l.line_name
        ORDER BY total_ridership DESC
    """).df()


def weekday_profile(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Average ridership by day of week."""
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df = con.execute("""
        SELECT
            d.day_of_week,
            AVG(f.ridership) AS avg_ridership,
        FROM fact_ridership f
        JOIN dim_date d USING (date)
        GROUP BY d.day_of_week
        ORDER BY d.day_of_week
    """).df()
    df["day_name"] = df["day_of_week"].map(dict(enumerate(days)))
    return df


def export_summaries(con: duckdb.DuckDBPyConnection) -> None:
    """Export summary tables to CSV."""
    DATA_OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, fn in [
        ("summary_overall", overall_summary),
        ("summary_by_line", line_summary),
        ("summary_weekday", weekday_profile),
    ]:
        df = fn(con)
        path = DATA_OUTPUT / f"{name}.csv"
        df.to_csv(path, index=False)
        logger.info("Saved %s", path)


if __name__ == "__main__":
    con = duckdb.connect(str(DB_PATH), read_only=True)
    print("\n=== Overall Summary ===")
    print(overall_summary(con).to_string(index=False))
    print("\n=== By Line ===")
    print(line_summary(con).to_string(index=False))
    print("\n=== Weekday Profile ===")
    print(weekday_profile(con).to_string(index=False))
    export_summaries(con)
    con.close()
