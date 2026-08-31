# market-agent-lab

**Version 0.1 -- PAPER TRADING / SIMULATION ONLY.**

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

## Recommended next development milestone

Wire `learning/drift.py`'s PSI and IC-drop detectors into a scheduled
trigger (e.g. a periodic job that calls `learning/retrain.py` +
`learning/champion_challenger.py` automatically whenever drift crosses a
threshold), and replace the synthetic data sources in `data/` with a
real, licensed, delayed (never real-time-trading-grade) market-data feed
behind the same `data/market_data.py` / `data/fundamentals.py` /
`data/macro.py` / `data/news.py` interfaces -- the rest of the pipeline
should not need to change.
