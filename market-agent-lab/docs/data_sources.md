# Data sources (Version 0.2)

Every external data source used by V0.2's real-data pipeline, registered
in `data/providers/registry.py::PROVIDER_CATALOG` -- the single place
that knows which providers exist, whether each is currently enabled, and
why. Nothing in this project silently substitutes synthetic values for
missing real data: an unavailable provider reports `UNAVAILABLE` with an
explicit reason (in `data_ingestion_runs` and the `/v2/providers/health`
API endpoint), never a fabricated number.

## Prices

| Source | Role | Key required | Status |
|---|---|---|---|
| Yahoo Finance (chart API) | Primary | No (needs a browser-like User-Agent) | Enabled |
| StockAnalysis.com API | Secondary / cross-check | No | Enabled |
| Stooq | (would-be tertiary) | No | **Disabled** -- returns a JavaScript proof-of-work bot-verification challenge instead of data; not solvable by a plain HTTP client |

Every trading day's bar is fetched from *both* Yahoo and StockAnalysis
and reconciled (`data/providers/prices/reconciliation.py`): if the two
sources' close prices differ by more than a configurable tolerance, the
bar is flagged (`PriceReconciliation.status`) rather than silently
averaged or trusted from one source. A bar is trainable
(`filter_trainable_bars`) only once reconciled cleanly or within
tolerance -- see `docs/point_in_time_data.md`.

## Fundamentals

| Source | Role | Key required | Status |
|---|---|---|---|
| SEC EDGAR XBRL company facts | Sole source | No (requires a descriptive User-Agent per SEC's fair-access policy) | Enabled |

SEC EDGAR is authoritative for US-listed companies and free. Facts are
tagged with both `period_end` (the accounting period) and
`publication_timestamp` (when the filing was actually made public) --
every downstream join uses `publication_timestamp`, never `period_end`,
to avoid look-ahead bias (see `docs/point_in_time_data.md`).

XBRL data is noisier than it looks: the same tag (e.g. `Revenues`) can
carry a discrete quarter's figure, a cumulative year-to-date figure, or a
prior-year comparative figure, all under the same concept in the same
filing. `data/real_fundamentals.py::_pick_by_duration` disambiguates by
each candidate fact's own `period_start`/`period_end` duration (closest
to 91 days for quarterly, 365 for annual, within a tolerance) before
falling back to most-recent `period_end` -- a naive "just take the latest
period_end" approach was tried first and produced implausible growth
figures (verified against live AAPL data) before this fix.

## Macro

| Source | Role | Key required | Status |
|---|---|---|---|
| FRED (`fredgraph.csv`) | Rates, yields, VIX | No | Enabled |
| Bureau of Labor Statistics API v2 | CPI, unemployment, payrolls | No (an API key raises the daily request cap) | Enabled |
| US Treasury Fiscal Data API | Treasury issuance | No | Enabled |
| Bureau of Economic Analysis API | GDP and related series | **Yes** (`BEA_API_KEY`) | **Disabled by default** -- dataset-list metadata is reachable unauthenticated, but real series retrieval requires a registered key |

Each series' typical publication lag is documented alongside its fetch
function (`data/providers/macro/*.py`) and respected by the point-in-time
join (a macro observation is only visible to a feature row once its
actual publication date has passed).

## News

| Source | Role | Tier | Key required | Status |
|---|---|---|---|---|
| SEC 8-K filings | Official corporate event disclosures, item-classified | Tier 1 (official) | No | Enabled |
| GDELT Doc API | Broad financial-media coverage | Tier 2 | No | **Disabled** -- `api.gdeltproject.org` is unreachable from this network (connection reset/timeout on every attempt) |
| Company investor-relations RSS | Official company announcements | Tier 1 | No | **Disabled** -- no universal free feed registry exists; per-company RSS URLs would need to be configured explicitly, out of scope for the default universe |
| NewsAPI.org | General financial media | Tier 2 | **Yes** (`NEWS_API_KEY`) | **Disabled by default** |

See `docs/news_pipeline.md` for the full news pipeline, tier weighting,
and point-in-time enforcement.

## Read-only prediction-market research signal

| Source | Role | Key required | Status |
|---|---|---|---|
| Polymarket (`gamma-api`, read-only) | Public market-implied event probabilities | No | Enabled |

This is a **read-only research signal only**. `PredictionMarketReadOnlyProvider`
(`data/providers/events/prediction_market_readonly.py`) exposes exactly
two public methods (`source_id`, `get_active_events`) -- enforced
structurally by `tests/test_prediction_market_readonly.py`, which asserts
the class's complete public API against an explicit allow-list and scans
for any execution-shaped method name (order, buy, sell, wager, wallet,
...) anywhere on the class, public or private. There is no order
placement, wallet, authentication, or wagering capability anywhere in
this codebase, full stop.

## Caching and rate limiting

Every provider goes through `data/providers/base.py`:

* **On-disk JSON response cache** (`data_store/cache/`, content-hash
  keyed) so repeated runs against the same parameters don't re-fetch.
* **`RateLimitedClient`** -- exponential backoff on 429/5xx, a configured
  minimum delay between requests per provider.
* **`ProviderHealthTracker`** (in-memory) plus a persisted
  `data_ingestion_runs` row for every ingestion attempt (success or
  failure), which is what `/v2/providers/health` and the dashboard's
  Provider Health tab actually read -- a health view that survives
  process restarts, unlike the in-memory tracker alone.

See `docs/provider_setup.md` for how to configure API keys and check
provider status.
