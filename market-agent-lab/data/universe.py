"""Point-in-time investable universe (Phase 4/5).

Version 0.2 does NOT have a licensed historical index-constituent-change
feed (that data is generally only available from paid vendors). The
membership mechanism below is fully point-in-time-*capable* -- every
membership row has a ``start_date``/``end_date`` interval and every
cross-sectional feature/universe query goes through
``get_point_in_time_universe(as_of)`` rather than "today's symbol list"
-- but for Version 0.2 the membership intervals are seeded starting from
the configured backtest start date rather than backfilled with real
historical S&P/index inclusion and exclusion events.

**This means the shipped default universe is NOT free of survivorship
bias**, and callers must not present it as historically representative of
"the S&P 100 in 2015" or similar. See ``docs/point_in_time_data.md`` for
the full explanation. The mechanism is correct; the seed data is a known,
documented limitation, not a silent gap.
"""

from __future__ import annotations

from datetime import datetime

import duckdb

from core.schemas_v2 import UniverseMembership
from data.real_prices import REAL_BENCHMARK_SYMBOL
from database import repository_v2 as repo_v2

SURVIVORSHIP_BIAS_WARNING = (
    "Universe membership is seeded from a CURRENT symbol list with no licensed "
    "historical index-constituent feed. This is a survivorship-biased universe: "
    "it does not reflect which symbols would actually have been investable "
    "members of any given index at each historical date. Do not present "
    "backtest results on this universe as free of survivorship bias."
)

# A liquid, well-known 20-symbol US large-cap universe for the default
# real-demo run (kept deliberately small and manageable -- see brief
# section 55). Configurable at the CLI layer; this is just the default.
DEFAULT_REAL_UNIVERSE: list[str] = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM", "V", "UNH",
    "HD", "PG", "MA", "JNJ", "XOM", "COST", "MRK", "ABBV", "KO", "PEP",
]


def seed_universe_membership(
    con: duckdb.DuckDBPyConnection,
    universe_name: str,
    symbols: list[str],
    start_date: datetime,
    source: str = "manual_configuration_current_list",
) -> int:
    memberships = [
        UniverseMembership(
            universe_name=universe_name, symbol=symbol, start_date=start_date, end_date=None,
            source=source, notes=SURVIVORSHIP_BIAS_WARNING,
        )
        for symbol in symbols
    ]
    return repo_v2.insert_universe_membership(con, memberships)


def get_point_in_time_universe(con: duckdb.DuckDBPyConnection, universe_name: str, as_of: datetime) -> list[str]:
    return repo_v2.get_point_in_time_universe(con, universe_name, as_of)


def universe_with_benchmark(symbols: list[str]) -> list[str]:
    return list(symbols) + ([REAL_BENCHMARK_SYMBOL] if REAL_BENCHMARK_SYMBOL not in symbols else [])
