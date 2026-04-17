"""
End-to-end ingestion pipeline: fetch + clean in one call.
"""

import logging
from pathlib import Path

from src.ingestion.fetch_data import fetch_all
from src.ingestion.clean_data import clean_all

logger = logging.getLogger(__name__)


def run():
    logger.info("=== Ingestion pipeline START ===")
    paths = fetch_all()
    logger.info("Raw data available: %s", list(paths.keys()))
    frames = clean_all()
    for name, df in frames.items():
        logger.info("  cleaned %-15s  shape=%s", name, df.shape)
    logger.info("=== Ingestion pipeline DONE ===")
    return frames


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
