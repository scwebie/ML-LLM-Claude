# Point-in-time data discipline (Version 0.2)

The single most important correctness property of V0.2: **every feature
value used to train or predict at timestamp `t` must only ever depend on
information that was actually knowable at `t`.** This document enumerates
where that discipline is implemented and tested, by data type.

## The core rule: publication timestamp, never descriptive date

A record's *descriptive* date (a fundamental's `period_end`, a macro
series' reference month, an event's `resolution_date`) is never the join
key against the trading calendar. The *publication* timestamp -- when the
information actually became public -- is:

| Data type | Descriptive date (never joined on) | Publication timestamp (joined on) |
|---|---|---|
| Fundamentals | `period_end` | `publication_timestamp` (the SEC filing date) |
| Macro | reference month/quarter | each series' documented publication lag |
| News | (n/a) | `published_at`, strictly `<=` the decision timestamp |
| Corporate actions | ex-date | applied only for bars on/after the ex-date |
| Event probabilities | `resolution_date` | `observed_timestamp` (when the market-implied probability was itself observed) |
| Universe membership | (n/a) | interval-based `[start_date, end_date)`, queried via `get_point_in_time_universe(as_of)` |

## Prices and corporate actions

Two independent sources (Yahoo Finance, StockAnalysis.com) are fetched
and reconciled per bar (`data/providers/prices/reconciliation.py`); a bar
with a major, unreconciled discrepancy is excluded from training
(`filter_trainable_bars`) rather than silently trusted. Corporate actions
(splits, dividends) are parsed and applied without double-adjustment --
Yahoo's `adjusted_close` is already split/dividend-adjusted, so the
adjustment is applied exactly once, tracked via `CorporateAction` records
rather than blindly trusting either provider's own adjustment.

## Fundamentals

SEC XBRL facts carry both `period_end` and `publication_timestamp`
(`core/schemas_v2.py::FundamentalFact`). The real feature matrix joins a
feature row's `timestamp` against fundamentals via `pd.merge_asof(...,
direction="backward")` on `publication_timestamp`
(`data/real_features.py`) -- a filing is only visible to rows dated on or
after the day it was actually made public, never on or after its
accounting period end (which would leak the filing's contents backward
into the weeks/months before it existed).

## Macro

Each macro series' typical publication lag is documented next to its
fetch function (`data/providers/macro/*.py`) and respected when building
the point-in-time macro history a feature row can see
(`data.macro.get_macro_history_asof`, reused unchanged from V0.1's
mechanism, since it was already publication-timestamp-based).

## News

`NewsArticle.published_at <= decision_timestamp` is enforced everywhere
news is read for feature computation (`data/real_features.py`,
`features/news_features.py`) -- see `docs/news_pipeline.md` for the full
pipeline and the `timestamp_uncertain` flag for sources without a
reliable publication timestamp.

## Read-only event-probability signal

`EventProbabilityObservation.observed_timestamp` is the moment a market's
implied probability was actually observed; `_lookup_asof` in
`data/real_features.py` performs a positional as-of lookup (bisect on a
sorted per-event observation history) that returns `None` -- not the
latest known value -- when no observation existed yet at the requested
timestamp. `tests/test_real_features.py` includes an explicit test where
one event's only observation is dated *after* the query timestamp and
asserts it contributes nothing to that row's feature.

## Point-in-time universe membership

`data/universe.py` stores membership as `[start_date, end_date)`
intervals per `(universe_name, symbol)`, queried via
`get_point_in_time_universe(con, universe_name, as_of)` -- every
cross-sectional computation (percentile ranks, market breadth, the
default 20-symbol universe) goes through this rather than "today's symbol
list."

**Known, documented limitation:** V0.2 does not have a licensed
historical index-constituent-change feed. `DEFAULT_REAL_UNIVERSE`'s
membership intervals are seeded starting from the configured backtest
start date using a *current* symbol list, not backfilled with real
historical inclusion/exclusion events. `SURVIVORSHIP_BIAS_WARNING`
(`data/universe.py`) is attached to every seeded membership row and
repeated in every report this system produces: **the shipped default
universe is survivorship-biased** -- it reflects which 20 companies are
large, liquid, and still-listed *today*, not which companies would
actually have been investable members of any historical universe at each
past date. The point-in-time *mechanism* is correct and fully
interval-based; the *seed data* behind it is a known gap, not a silent
one.

## Leakage guards enforced at runtime, not just by convention

Several point-in-time properties are checked with hard runtime
assertions, not just documented as intent:

* `backtesting/purged_walk_forward.py::run_purged_walk_forward` asserts,
  independent of fold-boundary configuration, that no training row's
  20-day target-realization window reaches the earliest validation
  timestamp actually used in that fold.
* `backtesting/holdout.py::split_development_and_holdout` applies the
  same purge/embargo machinery at the holdout boundary, verified with an
  adversarial test (`tests/test_holdout.py`) that constructs a row whose
  target window reaches into the holdout and asserts it is excluded from
  development.
* `tests/test_purged_walk_forward.py` includes explicit
  "adversarial future-data-injection" tests: a row is deliberately placed
  where a naive `timestamp < validation_start` split would include it,
  and the test asserts purge/embargo excludes it.

See `docs/evaluation_v02.md` for the full purged+embargoed+nested
walk-forward and holdout methodology.
