#!/usr/bin/env python
"""Generate and load the synthetic universe into DuckDB (Phase 12/13, step 1).

    uv run python scripts/generate_sample_data.py
"""

from __future__ import annotations

from core.config import settings
from core.logging import configure_logging, get_logger
from data.market_data import load_all_synthetic_data
from database.db import get_connection

logger = get_logger(__name__)


def main() -> None:
    configure_logging()
    con = get_connection()
    counts = load_all_synthetic_data(
        con,
        seed=settings.synthetic_seed,
        start_date=settings.synthetic_start_date,
        end_date=settings.synthetic_end_date,
        out_dir=settings.data_store_dir / "raw" / "synthetic",
    )
    logger.info("synthetic_data_loaded", **counts)


if __name__ == "__main__":
    main()
