"""
Transform pipeline: build schema + aggregates in one command.
"""

import logging
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from schema import build_schema, export_dim_tables
from aggregates import build_aggregates

logger = logging.getLogger(__name__)


def run():
    logger.info("=== Transform pipeline START ===")
    con = build_schema()
    export_dim_tables(con)
    build_aggregates(con)
    con.close()
    logger.info("=== Transform pipeline DONE ===")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
