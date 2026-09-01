# News pipeline (Version 0.2)

## Sources

Only **SEC 8-K filings** are enabled as a real news source by default --
see `docs/data_sources.md` for why GDELT (unreachable), company IR feeds
(no universal registry), and NewsAPI (requires a key) are wired but
disabled. This is a real, meaningful limitation: 8-K filings capture
officially-disclosed material events, not broad market sentiment or
analyst commentary. The system reports this honestly rather than
supplementing with a fabricated sentiment feed.

## SEC 8-K as a news source (`data/providers/news/sec_events.py`)

Every 8-K filing declares one or more numbered "Item" codes describing
what triggered it (e.g. Item 2.02 = results of operations, Item 5.02 =
officer/director changes). `SecEventsProvider` maps each item code to one
of nine deterministic categories via `ITEM_CATEGORY_MAP`:

| Item(s) | Category |
|---|---|
| 1.01, 1.02, 2.01, 5.01 | `M&A` |
| 1.03 | `litigation` |
| 2.02 | `earnings` |
| 2.03, 2.04 | `credit` |
| 2.05, 7.01 | `guidance` |
| 3.01 | `regulation` |
| 5.02 | `management` |
| 6.01 | `product` |
| everything else (2.06, 3.02, 3.03, 4.01, 4.02, 5.03, 5.07, 8.01, 9.01) | `other` |

A filing with multiple item codes maps to multiple categories; the
alphabetically-first is used as the article's `primary_category`. The
headline is generated deterministically (`"{symbol} 8-K filing (items
{items})"`) -- there is no LLM involved in classification anywhere in
this pipeline by default.

## News tiers and source-quality weighting

`core/schemas_v2.py::NewsTier` ranks sources by reliability:

| Tier | Weight | Example |
|---|---|---|
| `TIER_1_OFFICIAL` | 1.0 | SEC 8-K filings, company IR |
| `TIER_1_MAJOR_NEWS` | 0.85 | (not currently populated) |
| `TIER_2_FINANCIAL_MEDIA` | 0.6 | GDELT, NewsAPI (disabled) |
| `TIER_3_OTHER` | 0.3 | (not currently populated) |
| `UNKNOWN` | 0.1 | fallback |

`features/news_features.py::compute_deterministic_news_counts` uses these
weights for `source_weighted_count_7d` -- a simple article count would
treat an 8-K filing and an unverified blog post identically; the weight
reflects that they are not equally trustworthy signals.

## Deterministic classification (`classify_article`, `heuristic_v1`)

Every article is scored on four axes by a fixed lookup table keyed on its
category -- **no LLM call, ever, by default**:

* **Sentiment** (`_CATEGORY_SENTIMENT`): e.g. `earnings` +0.10,
  `litigation` -0.30, `M&A` 0.0 (ambiguous by nature).
* **Impact** (`_CATEGORY_IMPACT`): 0.15-0.6, how material the category
  typically is.
* **Horizon** (`_CATEGORY_HORIZON`): the typical timescale the event's
  effect plays out over (`intraday`, `1-5d`, `5-20d`, `long-term`).
* **Uncertainty**: 0.2 for Tier-1-official sources, 0.4 otherwise, +0.2
  if `timestamp_uncertain` is set.

Every `NewsAgentFeatures` record produced this way is stamped
`llm_model="heuristic_v1"` (`agents/event_relevance.py` uses the same
pattern for event-probability relevance) -- an explicit provenance label
so nothing produced by a fixed rule table is ever mistaken for, or later
silently swapped for, actual model/LLM-generated content.

An optional `use_llm=True` path exists (matching V0.1's agent design) for
a natural-language *narrative* summary only; every numeric score above
remains deterministic regardless.

## Point-in-time enforcement

`published_at <= decision_timestamp` is enforced at read time
(`data/real_features.py::_news_row_for_agent`, `features/news_features.py`).
An article whose exact publication time is unknown or unreliable is
marked `timestamp_uncertain=True` rather than assigned a fabricated
timestamp -- this raises its `uncertainty` score (above) instead of being
silently treated as precisely dated.

## Deduplication

`database.repository_v2::is_duplicate_article` checks a dedupe key before
insert (`data/real_news.py::ingest_news`) so re-running ingestion over an
overlapping date range does not double-count the same filing.
