# Provider setup (Version 0.2)

How to configure the real-data providers, check what's enabled, and add
API keys for the ones that require them.

## Nothing extra is required for most providers

Yahoo Finance, StockAnalysis.com, SEC EDGAR, FRED, BLS (unregistered tier),
US Treasury Fiscal Data, SEC 8-K, and the read-only Polymarket signal all
work with **no API key** -- `uv run python main.py real-demo` works
out of the box against the default 20-symbol universe.

## Optional API keys

Copy `.env.example` to `.env` and fill in whichever of these you have:

```bash
# --- Optional keys for providers that are disabled without them ------------
BEA_API_KEY=          # Bureau of Economic Analysis (GDP series) -- register free at https://apps.bea.gov/API/signup/
BLS_API_KEY=          # Bureau of Labor Statistics -- raises the daily request cap; works unregistered at a lower cap
NEWS_API_KEY=         # NewsAPI.org -- broader financial-media news coverage
```

Without `BEA_API_KEY`, the `bea` provider reports `UNAVAILABLE` (with
this exact reason) rather than being silently skipped or faked --
`GET /v2/providers/health` and the dashboard's Provider Health tab show
this explicitly. The same is true for `news_api` without `NEWS_API_KEY`.

`BLS_API_KEY` is different: BLS's public API v2 already works
unregistered at a lower daily request cap, so `bls` is enabled either
way; the key only raises the cap for heavier use.

## Checking provider status

```bash
# Static catalog: every registered provider, enabled or not, and why
curl http://localhost:8000/v2/providers/catalog

# Live-ish health: catalog + the most recent ingestion attempt per source
curl http://localhost:8000/v2/providers/health

# Full ingestion-run history for one source
curl "http://localhost:8000/v2/providers/ingestion-runs?source_id=sec_edgar"
```

Or in the dashboard (`uv run python main.py serve-dashboard`): the
**Provider Health** tab.

## Providers that are disabled regardless of configuration

Two sources are hard-disabled in `data/providers/registry.py::KNOWN_UNAVAILABLE`
because they were found, during development, to be technically unusable
from a plain automated HTTP client -- not a policy choice, a verified
limitation:

* **Stooq** -- returns a JavaScript proof-of-work bot-verification
  challenge instead of data.
* **GDELT** -- `api.gdeltproject.org` is unreachable from this network
  (connection reset/timeout on every attempt).

If either becomes usable in a different network environment (or via a
headless-browser bridge), remove the corresponding entry from
`KNOWN_UNAVAILABLE` and it re-enables normally -- no other code changes
required.

**Company investor-relations RSS** is disabled for a different reason:
there is no universal free feed registry, so per-company RSS URLs would
need to be configured explicitly per symbol -- out of scope for the
default configurable universe, but the provider module
(`data/providers/news/company_ir.py`) exists and can be wired up manually
for a specific symbol list.

## Rate limits and caching

Every provider goes through `data/providers/base.py::RateLimitedClient`
(exponential backoff on 429/5xx) and an on-disk JSON response cache
(`data_store/cache/`, gitignored, content-hash keyed) -- re-running
ingestion over an already-fetched date range is fast and does not
re-hit the network. Delete `data_store/cache/` to force a clean refetch.

## User-Agent requirements

Both Yahoo Finance and SEC EDGAR require a descriptive User-Agent (SEC's
fair-access policy explicitly requests a contact identifier). This is set
automatically by the provider clients; no configuration needed, but if
you see 403s from SEC EDGAR outside this codebase's own clients, that is
almost always why.
