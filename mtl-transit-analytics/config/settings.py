"""Central configuration for mtl-transit-analytics."""

from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
DATA_OUTPUT = ROOT_DIR / "data" / "output"

# DuckDB database path
DB_PATH = DATA_PROCESSED / "transit.duckdb"

# Montreal Open Data portal base URL
OPEN_DATA_BASE_URL = "https://donnees.montreal.ca/api/3/action"

# STM dataset IDs on the Montreal open data portal
STM_DATASETS = {
    # Achalandage (ridership) by station / time period
    "ridership_metro": "5b6f3225-cd50-4d56-8768-ef9d52348a0a",
    # STM GTFS static feed (stops, routes, trips, stop_times)
    "gtfs_stm": "ed9c48d9-b7a9-44f5-9a9c-cf1b1e6e7b90",
}

# Period labels used in STM data
PEAK_HOURS = {
    "AM_PEAK": ("06:30", "09:00"),
    "PM_PEAK": ("15:30", "18:30"),
}

# COVID reference dates for hypothesis testing
COVID_START = "2020-03-13"   # First Quebec lockdown
COVID_END = "2021-12-31"     # End of major restrictions

# Lines of interest
METRO_LINES = {
    "1": "Verte (Green)",
    "2": "Orange",
    "4": "Jaune (Yellow)",
    "5": "Bleue (Blue)",
}

# Ensure directories exist when module is loaded
for _d in (DATA_RAW, DATA_PROCESSED, DATA_OUTPUT):
    _d.mkdir(parents=True, exist_ok=True)
