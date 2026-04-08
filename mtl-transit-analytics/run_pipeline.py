"""
Master pipeline runner for mtl-transit-analytics.

Steps:
    1. Ingest  – fetch raw data, clean, save Parquet
    2. Transform – build DuckDB schema + aggregate tables, export CSVs
    3. Analyse  – EDA plots + summary statistics
    4. Stats    – hypothesis tests + predictive model

Usage:
    python run_pipeline.py [--steps ingest transform analyse stats]
    python run_pipeline.py                     # run all steps
    python run_pipeline.py --steps ingest      # run only ingestion
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def step_ingest(args):
    logger.info("━━━ STEP 1: INGEST ━━━")
    from src.ingestion.fetch_data import fetch_all
    from src.ingestion.clean_data import clean_all
    fetch_all(force=args.force)
    frames = clean_all()
    for name, df in frames.items():
        logger.info("  %s: %s", name, df.shape)


def step_transform(args):
    logger.info("━━━ STEP 2: TRANSFORM ━━━")
    from src.transform.schema import build_schema, export_dim_tables
    from src.transform.aggregates import build_aggregates
    con = build_schema()
    export_dim_tables(con)
    build_aggregates(con)
    con.close()


def step_analyse(args):
    logger.info("━━━ STEP 3: ANALYSE ━━━")
    import duckdb
    from config.settings import DB_PATH
    from src.analysis.eda import run_all as run_eda
    from src.analysis.summary_stats import export_summaries

    run_eda(save=True)
    con = duckdb.connect(str(DB_PATH), read_only=True)
    export_summaries(con)
    con.close()


def step_stats(args):
    logger.info("━━━ STEP 4: STATS ━━━")
    from src.stats.hypothesis_test import run_all_tests
    from src.stats.predictive_model import run as run_model

    run_all_tests()
    run_model()


STEPS = {
    "ingest": step_ingest,
    "transform": step_transform,
    "analyse": step_analyse,
    "stats": step_stats,
}


def main():
    parser = argparse.ArgumentParser(description="Run the mtl-transit-analytics pipeline")
    parser.add_argument(
        "--steps", nargs="+", choices=list(STEPS.keys()),
        default=list(STEPS.keys()),
        help="Which pipeline steps to run (default: all)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-download of raw data even if cached",
    )
    args = parser.parse_args()

    for step_name in args.steps:
        STEPS[step_name](args)

    logger.info("Pipeline finished. Check data/output/ for results.")


if __name__ == "__main__":
    main()
