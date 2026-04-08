"""
pytest tests for the transform layer (src/transform/).

Tests validate:
    - Schema creation (dim_lines, dim_stations, dim_date, fact_ridership)
    - Referential integrity between tables
    - Aggregate computation correctness
    - Data quality invariants (no negative ridership, date spine completeness)
    - DuckDB SQL logic (window functions, GROUP BY, etc.)

All tests use an in-memory DuckDB database seeded with a small synthetic
dataset so they run offline without the full pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import date, timedelta

import duckdb
import pandas as pd
import pytest

# Allow importing project modules
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def raw_ridership_df() -> pd.DataFrame:
    """Minimal synthetic ridership DataFrame that mirrors the real schema."""
    records = []
    lines = [("1", "Verte"), ("2", "Orange"), ("4", "Jaune"), ("5", "Bleue")]
    stations = {
        "1": [("STA", "StationA"), ("STB", "StationB")],
        "2": [("STC", "StationC")],
        "4": [("STD", "StationD")],
        "5": [("STE", "StationE")],
    }
    periods = ["Heure de pointe AM", "Heure de pointe PM", "Hors pointe", "Nuit"]
    period_weights = [0.30, 0.35, 0.30, 0.05]

    base_date = date(2019, 1, 1)
    for delta in range(60):       # 60 days of data
        d = base_date + timedelta(days=delta)
        for line_id, line_name in lines:
            for st_id, st_name in stations[line_id]:
                for period, weight in zip(periods, period_weights):
                    records.append({
                        "date": d.isoformat(),
                        "line_id": line_id,
                        "line_name": line_name,
                        "station_id": st_id,
                        "station_name": st_name,
                        "period": period,
                        "ridership": int(1000 * weight),
                        "year": d.year,
                        "month": d.month,
                        "week": d.isocalendar()[1],
                        "day_of_week": d.weekday(),
                        "is_weekend": d.weekday() >= 5,
                        "is_peak": "pointe" in period,
                        "season": "Winter" if d.month in (12, 1, 2) else "Spring",
                    })
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


@pytest.fixture(scope="module")
def raw_stops_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"stop_id": "STA", "stop_name": "StationA", "stop_lat": 45.50, "stop_lon": -73.57, "route_id": "1"},
        {"stop_id": "STB", "stop_name": "StationB", "stop_lat": 45.51, "stop_lon": -73.58, "route_id": "1"},
        {"stop_id": "STC", "stop_name": "StationC", "stop_lat": 45.52, "stop_lon": -73.59, "route_id": "2"},
        {"stop_id": "STD", "stop_name": "StationD", "stop_lat": 45.53, "stop_lon": -73.60, "route_id": "4"},
        {"stop_id": "STE", "stop_name": "StationE", "stop_lat": 45.54, "stop_lon": -73.61, "route_id": "5"},
    ])


@pytest.fixture(scope="module")
def con(raw_ridership_df, raw_stops_df) -> duckdb.DuckDBPyConnection:
    """
    In-memory DuckDB connection with the full schema built from synthetic data.
    """
    c = duckdb.connect()   # in-memory

    # Register source DataFrames as views
    c.register("ridership_src", raw_ridership_df)
    c.register("stops_src", raw_stops_df)

    # dim_lines
    c.execute("""
        CREATE TABLE dim_lines AS
        SELECT DISTINCT line_id, line_name FROM ridership_src ORDER BY line_id
    """)

    # dim_stations (with geo from stops)
    c.execute("""
        CREATE TABLE dim_stations AS
        SELECT DISTINCT r.station_id, r.station_name, r.line_id,
                        s.stop_lat, s.stop_lon
        FROM ridership_src r
        LEFT JOIN stops_src s ON r.station_id = s.stop_id
        ORDER BY r.line_id, r.station_id
    """)

    # dim_date
    c.execute("""
        CREATE TABLE dim_date AS
        SELECT DISTINCT date, year, month, week, day_of_week,
                        is_weekend, season,
                        CASE
                            WHEN date BETWEEN '2020-03-13' AND '2021-12-31' THEN 'COVID'
                            WHEN date < '2020-03-13' THEN 'Pre-COVID'
                            ELSE 'Post-COVID'
                        END AS covid_era
        FROM ridership_src
        ORDER BY date
    """)

    # fact_ridership
    c.execute("""
        CREATE TABLE fact_ridership AS
        SELECT date, station_id, line_id, period, is_peak, ridership
        FROM ridership_src
        ORDER BY date, line_id, station_id
    """)

    # Aggregates used by tests
    c.execute("""
        CREATE TABLE agg_monthly_line AS
        SELECT d.year, d.month, d.covid_era, f.line_id, l.line_name,
               SUM(f.ridership) AS total_ridership
        FROM fact_ridership f
        JOIN dim_date d USING (date)
        JOIN dim_lines l USING (line_id)
        GROUP BY d.year, d.month, d.covid_era, f.line_id, l.line_name
        ORDER BY d.year, d.month, f.line_id
    """)

    c.execute("""
        CREATE TABLE agg_station_rank AS
        SELECT s.station_id, s.station_name, s.line_id, l.line_name,
               s.stop_lat, s.stop_lon,
               SUM(f.ridership) AS total_ridership,
               AVG(f.ridership) AS avg_daily_ridership
        FROM fact_ridership f
        JOIN dim_stations s USING (station_id)
        JOIN dim_lines l USING (line_id)
        GROUP BY s.station_id, s.station_name, s.line_id, l.line_name,
                 s.stop_lat, s.stop_lon
        ORDER BY total_ridership DESC
    """)

    yield c
    c.close()


# ---------------------------------------------------------------------------
# dim_lines tests
# ---------------------------------------------------------------------------

class TestDimLines:
    def test_row_count(self, con):
        """Should have exactly 4 lines."""
        n = con.execute("SELECT COUNT(*) FROM dim_lines").fetchone()[0]
        assert n == 4

    def test_expected_line_ids(self, con):
        ids = {r[0] for r in con.execute("SELECT line_id FROM dim_lines").fetchall()}
        assert ids == {"1", "2", "4", "5"}

    def test_no_null_line_names(self, con):
        nulls = con.execute("SELECT COUNT(*) FROM dim_lines WHERE line_name IS NULL").fetchone()[0]
        assert nulls == 0


# ---------------------------------------------------------------------------
# dim_stations tests
# ---------------------------------------------------------------------------

class TestDimStations:
    def test_row_count(self, con):
        n = con.execute("SELECT COUNT(*) FROM dim_stations").fetchone()[0]
        assert n == 5   # STA, STB, STC, STD, STE

    def test_all_stations_have_line(self, con):
        orphans = con.execute("""
            SELECT COUNT(*) FROM dim_stations s
            LEFT JOIN dim_lines l USING (line_id)
            WHERE l.line_id IS NULL
        """).fetchone()[0]
        assert orphans == 0, "All stations must reference a valid line"

    def test_geo_coordinates_reasonable(self, con):
        bad = con.execute("""
            SELECT COUNT(*) FROM dim_stations
            WHERE stop_lat NOT BETWEEN 45.4 AND 45.7
               OR stop_lon NOT BETWEEN -73.7 AND -73.5
        """).fetchone()[0]
        assert bad == 0, "All lat/lon values should be within Montreal bounding box"

    def test_no_duplicate_station_ids(self, con):
        dupes = con.execute("""
            SELECT COUNT(*) FROM (
                SELECT station_id, COUNT(*) AS c FROM dim_stations
                GROUP BY station_id HAVING c > 1
            )
        """).fetchone()[0]
        assert dupes == 0


# ---------------------------------------------------------------------------
# dim_date tests
# ---------------------------------------------------------------------------

class TestDimDate:
    def test_date_count_matches_expected(self, con):
        """60 days were generated."""
        n = con.execute("SELECT COUNT(*) FROM dim_date").fetchone()[0]
        assert n == 60

    def test_no_null_dates(self, con):
        nulls = con.execute("SELECT COUNT(*) FROM dim_date WHERE date IS NULL").fetchone()[0]
        assert nulls == 0

    def test_covid_era_labels(self, con):
        eras = {r[0] for r in con.execute("SELECT DISTINCT covid_era FROM dim_date").fetchall()}
        assert eras <= {"Pre-COVID", "COVID", "Post-COVID"}

    def test_weekend_flag_correctness(self, con):
        """is_weekend should be True iff day_of_week >= 5."""
        mismatches = con.execute("""
            SELECT COUNT(*) FROM dim_date
            WHERE (day_of_week >= 5 AND is_weekend = FALSE)
               OR (day_of_week < 5 AND is_weekend = TRUE)
        """).fetchone()[0]
        assert mismatches == 0

    def test_year_range(self, con):
        min_yr, max_yr = con.execute("SELECT MIN(year), MAX(year) FROM dim_date").fetchone()
        assert min_yr == 2019
        assert max_yr == 2019


# ---------------------------------------------------------------------------
# fact_ridership tests
# ---------------------------------------------------------------------------

class TestFactRidership:
    def test_no_negative_ridership(self, con):
        neg = con.execute("SELECT COUNT(*) FROM fact_ridership WHERE ridership < 0").fetchone()[0]
        assert neg == 0

    def test_all_dates_have_ridership(self, con):
        """Every date in dim_date must appear in fact_ridership."""
        orphan_dates = con.execute("""
            SELECT COUNT(*) FROM dim_date d
            WHERE d.date NOT IN (SELECT DISTINCT date FROM fact_ridership)
        """).fetchone()[0]
        assert orphan_dates == 0

    def test_station_fk_integrity(self, con):
        bad = con.execute("""
            SELECT COUNT(*) FROM fact_ridership f
            LEFT JOIN dim_stations s USING (station_id)
            WHERE s.station_id IS NULL
        """).fetchone()[0]
        assert bad == 0, "All fact rows must reference a valid station"

    def test_line_fk_integrity(self, con):
        bad = con.execute("""
            SELECT COUNT(*) FROM fact_ridership f
            LEFT JOIN dim_lines l USING (line_id)
            WHERE l.line_id IS NULL
        """).fetchone()[0]
        assert bad == 0, "All fact rows must reference a valid line"

    def test_period_values(self, con):
        valid_periods = {
            "Heure de pointe AM", "Heure de pointe PM",
            "Hors pointe", "Nuit",
        }
        actual = {r[0] for r in con.execute("SELECT DISTINCT period FROM fact_ridership").fetchall()}
        assert actual <= valid_periods

    def test_ridership_positive_for_peak_periods(self, con):
        """Peak periods should have non-zero ridership on weekdays."""
        zero_peak = con.execute("""
            SELECT COUNT(*) FROM fact_ridership f
            JOIN dim_date d USING (date)
            WHERE f.is_peak = TRUE
              AND d.is_weekend = FALSE
              AND f.ridership = 0
        """).fetchone()[0]
        assert zero_peak == 0


# ---------------------------------------------------------------------------
# agg_monthly_line tests
# ---------------------------------------------------------------------------

class TestAggMonthlyLine:
    def test_total_matches_fact(self, con):
        """Monthly aggregate totals should equal the sum in fact_ridership."""
        agg_total = con.execute("SELECT SUM(total_ridership) FROM agg_monthly_line").fetchone()[0]
        fact_total = con.execute("SELECT SUM(ridership) FROM fact_ridership").fetchone()[0]
        assert agg_total == fact_total

    def test_each_line_has_record(self, con):
        lines_in_agg = {r[0] for r in con.execute("SELECT DISTINCT line_id FROM agg_monthly_line").fetchall()}
        assert lines_in_agg == {"1", "2", "4", "5"}

    def test_no_null_ridership(self, con):
        nulls = con.execute("SELECT COUNT(*) FROM agg_monthly_line WHERE total_ridership IS NULL").fetchone()[0]
        assert nulls == 0

    def test_monthly_ridership_positive(self, con):
        neg = con.execute("SELECT COUNT(*) FROM agg_monthly_line WHERE total_ridership <= 0").fetchone()[0]
        assert neg == 0


# ---------------------------------------------------------------------------
# agg_station_rank tests
# ---------------------------------------------------------------------------

class TestAggStationRank:
    def test_station_count(self, con):
        n = con.execute("SELECT COUNT(*) FROM agg_station_rank").fetchone()[0]
        assert n == 5

    def test_total_ridership_sorted_desc(self, con):
        totals = [r[0] for r in con.execute(
            "SELECT total_ridership FROM agg_station_rank ORDER BY total_ridership DESC"
        ).fetchall()]
        assert totals == sorted(totals, reverse=True)

    def test_avg_leq_total(self, con):
        """avg_daily_ridership must be ≤ total_ridership for every station."""
        violations = con.execute("""
            SELECT COUNT(*) FROM agg_station_rank
            WHERE avg_daily_ridership > total_ridership
        """).fetchone()[0]
        assert violations == 0


# ---------------------------------------------------------------------------
# SQL logic tests (window functions, GROUP BY)
# ---------------------------------------------------------------------------

class TestSQLLogic:
    def test_window_function_rolling_avg(self, con):
        """Verify a 3-month rolling average window function runs without error."""
        result = con.execute("""
            SELECT year, month, line_id,
                   AVG(total_ridership) OVER (
                       PARTITION BY line_id
                       ORDER BY year, month
                       ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
                   ) AS rolling_avg_3m
            FROM agg_monthly_line
            ORDER BY line_id, year, month
        """).df()
        assert len(result) > 0
        assert "rolling_avg_3m" in result.columns
        assert result["rolling_avg_3m"].isna().sum() == 0

    def test_peak_share_within_bounds(self, con):
        """Peak share (0-1) for every line."""
        result = con.execute("""
            SELECT line_id,
                   SUM(CASE WHEN is_peak THEN ridership ELSE 0 END) * 1.0 /
                   NULLIF(SUM(ridership), 0) AS peak_share
            FROM fact_ridership
            GROUP BY line_id
        """).df()
        assert ((result["peak_share"] >= 0) & (result["peak_share"] <= 1)).all()

    def test_group_by_season_returns_expected_seasons(self, con):
        seasons = {r[0] for r in con.execute(
            "SELECT DISTINCT season FROM dim_date"
        ).fetchall()}
        assert seasons <= {"Spring", "Summer", "Fall", "Winter"}

    def test_covid_era_partition_exhaustive(self, con):
        """Every date must belong to exactly one COVID era."""
        unlabelled = con.execute("""
            SELECT COUNT(*) FROM dim_date
            WHERE covid_era NOT IN ('Pre-COVID', 'COVID', 'Post-COVID')
        """).fetchone()[0]
        assert unlabelled == 0
