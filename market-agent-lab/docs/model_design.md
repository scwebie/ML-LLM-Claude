# Model Design (Version 0.1)

## Features

Every feature is deterministic and computed before any model or agent
sees it -- nothing in `features/` calls an LLM.

* **Technical** (`features/technical.py`): returns (1/5/10/20/60d), SMA
  (10/20/50/100/200) and distance-from-SMA (20/50/200), RSI-14, MACD +
  signal, ATR-14, Bollinger Band position, realised volatility
  (10/20/60d), 20d volume z-score and relative volume, 52-week
  high/low distance.
* **Fundamental** (`features/fundamental.py`): as-of ratios (growth,
  margins, ROIC, debt/cash) plus cross-sectional z-scores against the
  rest of the universe on the same date (valuation, profitability,
  growth).
* **Macro** (`features/macro.py`): trailing z-score and level per macro
  series, using only observations published at or before the query date.
* **Historical similarity** (`features/historical.py`): standardised
  Euclidean distance nearest-neighbour search
  (`D_t = sqrt(sum(w_i * (x_i,t - x_i,current)^2))`) against the
  symbol's own history, returning the realised forward-return
  distribution of the k nearest analogues.
* **Agent-derived** (`agents/*.py` via `agents/orchestrator.py`): each
  research agent's bounded scores (documented ranges in its module),
  plus `agent_disagreement` (population stddev of five agents'
  directional scores, Phase 11) and a disagreement-penalised composite
  confidence.

All feature rows are versioned (`feature_version`, default `fv1`) and
stored immutably in `feature_snapshots`; changing feature-engineering
logic requires bumping the version rather than rewriting history.

## Targets

Four targets, computed in `models/train.py::compute_excess_return_targets`,
against the synthetic benchmark series (`SYN_BENCH`) -- never against the
symbol's own price history, so the model learns relative (alpha-seeking)
behaviour:

1. `excess_return_5d` (regression)
2. `excess_return_20d` (regression)
3. `positive_5d` = `excess_return_5d > 0` (binary classification)
4. `positive_20d` = `excess_return_20d > 0` (binary classification)

The model never predicts BUY/SELL directly -- sizing and direction are
entirely the Portfolio Decision Engine's job (`portfolio/allocation.py`),
downstream of the model.

## Labels and temporal assumptions

* A target at row `t` is only realised `horizon` trading days later; rows
  in the trailing `horizon` days of any loaded dataset are `NaN` and are
  dropped before training/evaluation -- never imputed.
* Fundamentals and macro data are joined using `publication_timestamp`
  (fundamentals) / `publication_timestamp` (macro), never
  `reporting_period_end` -- this is the specific mechanism that prevents
  look-ahead bias from "the market already knew Q4 numbers before they
  were published." See `tests/test_temporal_joins.py`.
* The historical-similarity engine's leakage guard is structural, not a
  manually tuned cutoff: forward returns are computed on a feature
  history already truncated to `timestamp <= as_of`, so the trailing
  `horizon` rows get `NaN` forward returns "for free" and are excluded
  from the nearest-neighbour candidate pool.

## Retraining and validation

* **Walk-forward** (`backtesting/walk_forward.py`): expanding-window
  folds only -- e.g. train 2015-2019 / validate 2020, train 2015-2020 /
  validate 2021, etc. `run_walk_forward` asserts at runtime that no
  training-row timestamp is >= any validation-row timestamp in the same
  fold; this is not just a convention, it is a hard assertion.
* **Retraining** (`learning/retrain.py`): once a prediction's horizon has
  elapsed, `learning/outcomes.py` labels the realised outcome; a
  challenger model is trained on the extended dataset and registered with
  role `CHALLENGER`.
* **Drift detection** (`learning/drift.py`): population stability index
  (PSI) for feature-distribution drift, and a relative information-
  coefficient drop check for performance drift. In v0.1 these are
  diagnostic utilities available to trigger a retraining cycle; they are
  not yet wired into an automatic scheduler.

## Promotion criteria (`learning/champion_challenger.py`)

A challenger is **never** promoted merely because its raw return is
higher. `decide_promotion` requires, in order:

1. `information_coefficient >= min_information_coefficient` (a real edge
   must exist at all, positive or explicitly configured).
2. Calibration (Brier score) not worse than the champion's by more than
   `max_brier_regression_tolerance`.
3. Drawdown not worse than the champion's by more than
   `max_drawdown_regression_tolerance`.
4. Sharpe not worse than the champion's by more than
   `min_sharpe_improvement` (a small negative tolerance is allowed, but a
   large regression is not).
5. Information coefficient not meaningfully worse than the champion's.

If there is no existing champion, the first viable challenger (passing
criterion 1) is auto-promoted. Every decision -- promoted or rejected,
with the full rationale string and both metric dictionaries -- is written
to `promotion_log`, so the promotion history is fully auditable
(`dashboard/app.py`'s Model tab renders it directly).

## Documented simplifications (v0.1)

* `predicted_volatility` on `ModelPrediction` is a naive persistence
  forecast (the already-computed trailing 20-day realised volatility),
  not a learned output -- the brief specifies exactly four ML targets and
  volatility is not one of them.
* `fundamental_guidance_score` is a proxy built from EPS-growth/revenue-
  growth consistency; v0.1's synthetic dataset has no analyst-guidance or
  estimate-revision series.
* `execution/orders.py::evaluate_and_route` checks exposure limits
  against the portfolio snapshot at the *start* of a rebalance batch, not
  incrementally order-by-order within the same batch -- so a limit that
  is only breached by the *sum* of several same-day orders may not be
  caught until the next rebalance. Per-order limits (position size) are
  always caught immediately.
* Per-fold walk-forward promotion decisions (`main.py demo`) use a
  placeholder `max_drawdown=0.0` in the criteria comparison, since
  fold-level walk-forward validation doesn't run a full backtest with
  equity curve; the final held-out backtest (Phase 6-9) does compute a
  real max drawdown for the deployed champion.
* Portfolio sizing (`portfolio/allocation.py`) risk-adjusts a 5-day
  expected-return signal against a horizon-scaled (not full-annualised)
  volatility estimate, then clips to `max_position_weight` -- a
  deliberately simple, documented Kelly-adjacent heuristic, not a
  variance-covariance optimiser.
