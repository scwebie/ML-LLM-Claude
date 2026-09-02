# market-agent-lab

**Version 0.1 + Version 0.2 -- PAPER TRADING / SIMULATION ONLY.**

V0.1 (below) proved the architecture end to end on synthetic data. V0.2
extends it (does not replace it -- `main.py demo` still runs the original
synthetic pipeline unchanged) with real market data, real fundamentals,
real macro data, real news, an optional read-only prediction-market
signal, and a materially stricter evaluation methodology. See
[**Version 0.2**](#version-02----real-data-research-platform) below for
the V0.2-specific quickstart, or jump straight to the experiment report:
[`docs/EXPERIMENT_REPORT.md`](docs/EXPERIMENT_REPORT.md).

A multi-agent quantitative market-research and machine-learning system.
This version proves that the architecture -- research agents, feature
engineering, an ML alpha model, walk-forward backtesting, a deterministic
risk engine, simulated execution, and controlled model retraining -- works
correctly end to end, entirely on synthetic data.

**This system does not, and will never in this version:**
connect to a real-money brokerage; execute a real trade; integrate a
prediction market, betting platform, or gambling service. See
[`docs/architecture.md`](docs/architecture.md) for the full list of
enforced safety boundaries.

## What it does

For every trading day in the backtest period, the pipeline runs:

```
Research Agents -> Structured Features -> Feature Store
  -> ML Prediction Model -> Portfolio Decision Engine
  -> Deterministic Risk Engine -> Paper Execution Engine
  -> Performance Database -> Retraining Pipeline
```

Five research agents (Technical, Fundamental, Market Overview, Historical
Research, News/Event Intelligence) each produce a bounded, validated
Pydantic output from already-computed features -- none of them do their
own indicator math, and none of them can place an order. A LightGBM model
trained with expanding-window walk-forward validation predicts four
targets (5d/20d excess return, 5d/20d probability-positive). A
deterministic Portfolio Decision Engine turns predictions into target
weights; a deterministic Risk Engine approves or rejects every resulting
order with an explicit reason code; a simulated Paper Execution Engine
fills approved orders with spreads, slippage, commissions, and partial
fills. Everything is recorded, and a champion/challenger promotion gate
controls when a newly retrained model is ever allowed to replace the one
currently trading.

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the full component
map and a Mermaid data-flow diagram. See
[`docs/model_design.md`](docs/model_design.md) for feature/target/label
definitions, retraining, validation, and promotion-criteria details.

## Installation / local setup

Requires Python 3.12+ and [`uv`](https://github.com/astral-sh/uv).

```bash
git clone <this repo>
cd market-agent-lab
uv sync --extra dev
cp .env.example .env   # optional -- see below
```

Nothing in `.env` is required to run the offline synthetic demo.

## Environment variables

See [`.env.example`](.env.example) for the full list. Notable ones:

| Variable | Purpose | Required? |
|---|---|---|
| `OPENAI_API_KEY` | Optional LLM narrative enhancement for research agents | No -- agents run fully deterministic without it |
| `DATA_STORE_DIR`, `DUCKDB_PATH` | Local storage locations | No -- sane defaults |
| `SYNTHETIC_SEED`, `SYNTHETIC_START_DATE`, `SYNTHETIC_END_DATE` | Synthetic dataset generation | No -- sane defaults |
| `API_HOST`, `API_PORT` | FastAPI bind address | No |

## How to run the demo

```bash
uv run python main.py demo
```

This runs the full pipeline: generates the synthetic universe, builds and
stores features, trains + walk-forward-validates the LightGBM model,
runs champion/challenger promotion, backtests the champion through the
Portfolio Decision Engine -> Risk Engine -> Paper Execution Engine chain,
compares against three benchmarks, labels realised outcomes, and prints a
performance report (also saved under `data_store/results/`).

## How to run tests

```bash
uv run pytest
```

## How to start the API

```bash
uv run python main.py serve-api
# or directly:
uv run uvicorn api.main:app --reload
```

Read-only endpoints for predictions, risk decisions, fills, model
registry/promotions, and portfolio snapshots. See `api/main.py`.

## How to start the dashboard

```bash
uv run python main.py serve-dashboard
# or directly:
uv run streamlit run dashboard/app.py
```

Run `main.py demo` at least once first so there is data to show.

## Data leakage: what it is and how this codebase prevents it

Look-ahead bias creeps into a backtest whenever a feature or target at
time `t` is (even indirectly) computed using information not actually
available at `t`. Three concrete guards live in this codebase:

1. **Publication-time joins.** `FundamentalObservation` and
   `MacroObservation` both carry a `publication_timestamp` distinct from
   the period they describe. Every join against them
   (`database/repository.py::get_fundamentals_asof`,
   `get_macro_asof`) filters strictly on
   `publication_timestamp <= as_of` -- never on the report's own period
   end date. Real companies report Q4 results 6-8 weeks after quarter
   end; joining on period-end would silently hand the model next
   quarter's numbers early. Tested in `tests/test_temporal_joins.py`.
2. **Structural forward-return truncation.** The historical-analogue
   similarity engine (`features/historical.py`) computes forward returns
   on a feature history already truncated to `timestamp <= as_of`, so
   the rows that couldn't yet have a realised outcome are `NaN` "for
   free" and are excluded from the candidate pool -- there is no
   separately-tunable cutoff to get wrong.
3. **Walk-forward assertions.** `backtesting/walk_forward.py`'s fold
   dataclass rejects a fold whose training window overlaps its
   validation window at construction time, and `run_walk_forward` adds a
   second, independent runtime assertion that the actual training
   timestamps used are all strictly before the actual validation
   timestamps used.

## Walk-forward validation

The project brief is explicit: **never use a random train/test split for
the primary financial evaluation.** `backtesting/walk_forward.py`
implements expanding-window folds only, with fully configurable dates:

```
fold 0: train 2015-2019, validate 2020
fold 1: train 2015-2020, validate 2021
fold 2: train 2015-2021, validate 2022
```

Each fold trains a fresh model on all data up to that fold's cutoff and
validates strictly on the following, non-overlapping period -- so
validation performance always reflects genuinely out-of-sample,
forward-in-time behaviour.

## Champion/challenger retraining

Once a prediction's horizon has elapsed, `learning/outcomes.py` labels
the realised outcome from actual market data. `learning/retrain.py`
trains a new `CHALLENGER` model on the extended dataset.
`learning/champion_challenger.py::decide_promotion` then decides whether
to promote it -- requiring a real edge (information coefficient), no
material calibration regression, no material drawdown regression, and no
material Sharpe regression relative to the current `CHAMPION`. A
challenger with a higher raw return but a worse drawdown is rejected.
Every decision is logged to `promotion_log` with its full rationale.

## Limitations (Version 0.1)

* All market/fundamental/macro/news data is synthetic
  (`data/synthetic.py`) -- there is no real market-data integration in
  this version.
* The Portfolio Decision Engine's sizing rule is a deliberately simple,
  documented heuristic (see `docs/model_design.md`), not a
  variance-covariance portfolio optimiser.
* `predicted_volatility` is a naive persistence forecast, not a trained
  output (only four targets are specified in the brief, and volatility
  isn't one of them).
* Drift detection (`learning/drift.py`) provides the statistics (PSI,
  IC-drop) but is not yet wired into an automatic retraining scheduler --
  retraining in v0.1 is triggered manually / by the demo pipeline.
  Real automated triggering is worth adding, but on infra-scale data it
  is a separate reliability-engineering project and out of scope here.
* Order-level risk checks (position size) are always caught immediately;
  a limit breached only by the *sum* of several same-day orders may not
  be caught until the following rebalance (see `docs/model_design.md`).
* No sector/asset-class breakdown beyond the five fictional sectors in
  the synthetic universe; `max_sector_concentration` is a real,
  enforced limit, but with only 10 symbols across 5 sectors it is a
  coarse approximation of real portfolio construction.

## Recommended next development milestone (Version 0.1)

Wire `learning/drift.py`'s PSI and IC-drop detectors into a scheduled
trigger (e.g. a periodic job that calls `learning/retrain.py` +
`learning/champion_challenger.py` automatically whenever drift crosses a
threshold), and replace the synthetic data sources in `data/` with a
real, licensed, delayed (never real-time-trading-grade) market-data feed
behind the same `data/market_data.py` / `data/fundamentals.py` /
`data/macro.py` / `data/news.py` interfaces -- the rest of the pipeline
should not need to change.

*(V0.2, below, is exactly that milestone.)*

---

## Version 0.2 -- real-data research platform

**Still paper-trading / simulation only.** V0.2 replaces synthetic data
with real market data, real SEC fundamentals, real macro data, real news,
and an optional read-only prediction-market research signal -- but the
hard safety boundaries in `docs/architecture.md` are unchanged: no real
brokerage, no real trade, no execution/wagering capability anywhere, and
the LLM never controls risk or execution.

### What's new

* **Real data** -- see [`docs/data_sources.md`](docs/data_sources.md) for
  every provider, what's enabled/disabled and why, and
  [`docs/provider_setup.md`](docs/provider_setup.md) for API-key setup.
* **Point-in-time discipline** across prices, fundamentals, macro, news,
  events, and universe membership -- see
  [`docs/point_in_time_data.md`](docs/point_in_time_data.md).
* **Purged + embargoed + nested walk-forward evaluation, a fixed
  historical holdout period, a ten-piece robustness suite, and a
  stricter champion/challenger gate** -- see
  [`docs/evaluation_v02.md`](docs/evaluation_v02.md). (V0.3 note: that
  holdout has since been formally evaluated once -- see the V0.3 section
  below -- and is now a *used* historical test set, not an untouched one.)
* **A read-only prediction-market signal** (Polymarket) with an absolute,
  structurally-tested prohibition on any execution/wagering/wallet/auth
  capability.
* **New CLI commands**, **new `/v2` API endpoints**, and **three new
  dashboard tabs** (Provider Health, Data Quality, Robustness).

### V0.2 quickstart

```bash
# No API keys required for the default run.
uv run python main.py real-demo
```

This runs the full V0.2 pipeline end to end: ingest real prices,
fundamentals, macro data, and news for the default 20-symbol universe;
build the point-in-time real feature matrix; purged+embargoed
walk-forward evaluation with the V0.2 champion/challenger gate; and a
genuine paper-trading backtest through the same Portfolio/Risk/Execution
engine V0.1 uses. Or run each stage independently:

```bash
uv run python main.py ingest-prices
uv run python main.py ingest-fundamentals
uv run python main.py ingest-macro
uv run python main.py ingest-news
uv run python main.py build-real-features
uv run python main.py evaluate-real       # never touches the final holdout
```

`main.py demo` (V0.1's synthetic pipeline) is completely unaffected by
any of the above -- they write to separate, additively-designed tables
(`core/schemas_v2.py`) plus a distinct `feature_version` (`real_fv1`) in
V0.1's shared `feature_snapshots` table.

### Full experiment report

[`docs/EXPERIMENT_REPORT.md`](docs/EXPERIMENT_REPORT.md) documents this
project's actual `real-demo` run against real market data (2019-2025,
default 20-symbol universe): engineering results, data coverage,
evaluation methodology, model performance vs. benchmarks, robustness
results, and explicit safety confirmations -- reported without
cherry-picking, per the project brief.

## Version 0.3 -- scientific validity, instability diagnosis, reproducibility

**Still paper-trading / simulation only.** The V0.2 holdout result above
has been observed once and is treated from V0.3 onward as a **used**
historical test set -- never re-tuned against, never re-evaluated
repeatedly, and never again described as "untouched." V0.3's focus is
diagnosing WHY the V0.2 model didn't generalise convincingly, using
**development data only**.

### What's new

* **Fixed a champion/challenger self-promotion bug** where a deterministic
  retrain on unchanged data could report `promoted=true` against a
  vacuous, identical-metrics "challenger" -- see
  `learning/champion_challenger_v2.py`.
* **Audited and fixed the Sharpe ratio calculation** -- the previously
  reported Sharpe (6.649) was computed from an overlapping multi-day
  target series misread as independent daily returns. A genuinely
  chronological one-row-per-trading-day portfolio series now backs every
  Sharpe number -- see `backtesting/daily_portfolio.py`.
* **Development-only diagnostics**: IC by year, market regime, and
  symbol; IC decay; signal breadth (`backtesting/development_diagnostics.py`).
* **Feature-family ablation** with per-fold IC, bootstrap CIs, and
  genuine daily Sharpe (`backtesting/ablation_v3.py`).
* **Feature-importance stability** across folds (native + permutation
  importance, sign-flip and one-period-only detection;
  `backtesting/feature_stability.py`).
* **Five automated negative controls** (shuffled target, time-shifted
  target, random feature, a deliberate future-data leak the harness must
  catch, and symbol-label permutation; `backtesting/negative_controls.py`).
* **Re-audited purge/embargo boundaries** derived directly from the same
  masks training actually uses (`backtesting/purge_audit.py`).
* **Finite-permutation-corrected p-values**, IC information ratio,
  effective sample size under target overlap, and probabilistic/deflated
  Sharpe ratios (`backtesting/statistical_significance.py`).
* **Investigated Polymarket event-probability coverage**: only
  single-snapshot data is available (no historical archive endpoint
  exists), so those feature columns are correctly absent from historical
  training rows rather than backfilled -- see
  `data/event_coverage_diagnostics.py`.
* **A standalone, on-demand `evaluate-forward-paper` command** that loads
  the already-frozen champion and scores it against the post-holdout
  period exactly once, performing no training or model selection --
  see `backtesting/forward_paper.py`.
* **Simple-model benchmarks** (ridge, logistic, momentum, mean-reversion,
  equal-weight composite) run on the identical purge/embargo splits as
  LightGBM (`backtesting/simple_benchmarks.py`).
* **Transaction-cost, execution-delay, and rebalance-cadence stress
  tests** on a fixed, non-tuned grid (`backtesting/cost_delay_stress.py`).
* **Model registry reproducibility provenance** -- git commit,
  target-definition hash, random seed, data fingerprint, and artifact
  hash on every registered model, plus a `verify-reproducibility` CLI
  command (`models/reproducibility.py`).
* **Split CLI commands** so development, historical-holdout, and
  forward-paper evaluation are never conflated: `evaluate-development`
  (alias of `evaluate-real`, never touches either test period),
  `evaluate-historical-holdout` (standalone, on-demand, audit-logged,
  no retraining), `evaluate-forward-paper` (same discipline, post-holdout
  period).

### V0.3 research report

[`docs/V03_RESEARCH_REPORT.md`](docs/V03_RESEARCH_REPORT.md) is generated
from code (`scripts/generate_v03_report.py`) against this project's real,
already-ingested data. It separates **DEVELOPMENT RESULTS** (safe to
re-run any number of times), **USED HISTORICAL HOLDOUT RESULTS** (read
from the existing access log, never re-evaluated to produce the report),
and **FUTURE FORWARD-PAPER RESULTS** (reserved, not yet evaluated) --
including results that are not favorable (e.g. two of five negative
controls did not cleanly pass on real data, and feature importance
proved unstable across folds), reported honestly rather than omitted.
