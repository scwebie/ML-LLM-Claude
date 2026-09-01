# Evaluation methodology (Version 0.2)

V0.1 used a plain expanding-window walk-forward split
(`backtesting/walk_forward.py`, still used unchanged by `main.py demo`).
That is not safe for real-data targets, which look up to 20 trading days
into the future: a training row timestamped strictly before a
validation window's start can still have its *target* window reach into
that validation period. V0.2 adds a purged + embargoed + nested
walk-forward evaluator, a final untouched holdout, and a ten-piece
robustness suite, all built specifically for this.

## Purge and embargo (`backtesting/purged_walk_forward.py`)

For every evaluation window `[eval_start, eval_end]`:

* **PURGE** -- drop any candidate training row whose target-realization
  window `[timestamp, timestamp + horizon_days]` (in *trading days*, via
  a calendar built from the data itself, not fixed calendar-day offsets)
  overlaps the evaluation window at all. `horizon_days` defaults to 20
  (the longest target, `excess_return_20d`/`positive_20d`).
* **EMBARGO** -- additionally drop rows within `embargo_days` (default 5,
  matching the shorter 5-day target) trading days strictly after
  `eval_end`, guarding against residual serial-correlation leakage beyond
  the purge window.

A hard runtime assertion, independent of the mask construction, re-checks
after every fold that no training row's target-realization window reaches
the earliest validation timestamp actually used. This is verified with
adversarial tests (`tests/test_purged_walk_forward.py`) that deliberately
construct a row a naive split would leak and assert purge/embargo
excludes it.

Both **expanding** and **rolling** window modes are supported
(`generate_purged_folds(..., window_mode=...)`) -- rolling tests whether
the model needs unboundedly old data at all and is generally more robust
to regime change than an ever-growing training window.

## Nested inner cross-validation (`run_nested_purged_walk_forward`)

Hyperparameter selection is itself a place look-ahead bias can creep in:
picking the model that happens to score best on a validation fold is a
subtle form of leakage if that same fold's score is also what's reported.
`run_nested_purged_walk_forward` selects hyperparameters using ONLY an
inner purged+embargoed split carved out of each outer fold's training
window; the outer validation window is never touched until the final,
single evaluation of the selected model.

## Final untouched holdout (`backtesting/holdout.py`)

A configurable date range (`HOLDOUT_START_DATE`/`HOLDOUT_END_DATE`, fixed
once in advance -- default 2024-07-01 to 2025-06-30) that development,
walk-forward, ablation, and robustness work never touches:

* `split_development_and_holdout` purges/embargoes development rows
  against the holdout boundary using the same machinery as an outer
  walk-forward fold.
* `evaluate_on_holdout` is the **only** function in the project permitted
  to score a model on holdout rows. It trains nothing -- the model must
  already be fully fit and selected on development data alone -- and
  every single call is unconditionally logged to the `holdout_access_log`
  table (`core/schemas_v2.py::HoldoutAccessLog`, exposed at
  `/v2/holdout/access-log`), giving an auditable trail of exactly how
  many times, and for what purpose, the holdout was accessed.
* `assert_no_fold_touches_holdout` is a defensive check wired into
  `evaluate_real_step` (`real_pipeline.py`) so a configuration mistake
  that widens development into the holdout can never pass silently.

## Robustness / ablation / regime suite (`backtesting/robustness.py`)

Ten independently callable diagnostics, all operating on genuine
out-of-sample predictions (concatenated across purged walk-forward
folds):

1. **Block / stationary bootstrap confidence intervals**
   (`block_bootstrap_ci`) for a performance statistic on a time-ordered
   series -- resamples contiguous blocks (fixed-length, or
   Politis-Romano stationary with random block lengths) to preserve
   short-range serial dependence a plain iid bootstrap would erase.
2. **Feature-family ablation** (`run_feature_ablation`,
   `classify_feature_family`) -- retrains with each named family of
   columns removed (technical, fundamental, macro raw/regime,
   cross-sectional, news, agent, market breadth, and `eventprob` -- the
   read-only prediction-market signal ablates through the exact same
   mechanism as every other family) and reports the mean IC delta.
3. **Permutation test** (`permutation_test_ic`) -- shuffles the realised
   target and recomputes rank IC many times to build a null distribution;
   the p-value is the fraction of shuffles whose |IC| meets or exceeds
   the observed |IC|.
4. **Negative-control random feature** (`add_negative_control_feature`,
   `negative_control_report`) -- injects a column of pure iid noise and
   flags whether the model suspiciously ranks it as important.
5. **Regime-specific / year-by-year performance**
   (`evaluate_by_group`, `metrics_by_year`).
6. **Rank IC / calibration** (`rank_ic_report`, `calibration_report`,
   the latter wrapping `models.evaluate.calibration_curve` for the
   classification targets).
7. **Transaction-cost stress testing** (`build_quantile_portfolio_returns`,
   `cost_stress_test`) -- a simple quantile long/short portfolio built
   from the predicted signal, swept across a round-trip cost grid to show
   at what cost level the edge disappears.
8. **Execution-delay stress testing** (`execution_delay_stress_test`) --
   shifts the prediction used at each row forward by N trading days and
   recomputes rank IC, showing how fast the signal decays if acted on
   late.
9. **Factor exposure report** (`factor_exposure_report`) -- OLS-regresses
   the predicted signal on style-factor proxies (momentum, liquidity,
   volatility percentiles, ...) so a reviewer can tell whether "alpha" is
   a repackaged known factor.
10. **Feature-importance stability** (`feature_importance_stability`) --
    mean pairwise Spearman correlation of feature importances across
    folds; a low value flags overfitting to fold-specific noise.

## Champion/challenger promotion, V0.2 (`learning/champion_challenger_v2.py`)

V0.1's gate (`learning/champion_challenger.py`, untouched) auto-promotes
the very first challenger whenever there is no existing champion, gated
only on `information_coefficient > 0` -- appropriate for V0.1's synthetic
demo, too weak for real data where a barely-positive IC is easily noise.
V0.2 adds, alongside V0.1's module rather than modifying it:

* `learning/initial_qualification.py::InitialQualificationBar` -- a
  strictly higher bar for the FIRST champion: a minimum out-of-sample
  observation count, a real information coefficient *and* a real
  backtest-level Sharpe ratio (never silently defaulted from IC), and
  statistical significance via the permutation test above. Failing any
  criterion rejects the model outright, with every failing reason
  reported.
* Once a champion already exists, `decide_promotion_v2` delegates
  unchanged to V0.1's `decide_promotion` -- the ongoing sharpe/IC/
  drawdown/calibration comparison there is already the right test.

## Drift monitoring (`learning/drift_v2.py`)

Adds Kolmogorov-Smirnov and Wasserstein-distance statistics alongside
V0.1's population stability index (`learning/drift.py`, untouched):
`detect_feature_drift_full` flags a feature as drifted if PSI exceeds its
threshold, OR the KS test rejects equal distributions, OR the Wasserstein
distance exceeds a configurable multiple of the reference period's own
standard deviation -- reported per feature, flagged or not, so a reviewer
sees the full picture.

## CLI entry point

`uv run python main.py evaluate-real` runs steps 1-2 above (purged
walk-forward on development, V0.2 promotion decision) without ever
touching the holdout. `uv run python main.py real-demo` runs the full
pipeline end to end, finishing with a genuine paper-trading backtest
through the same Portfolio/Risk/Execution engine V0.1 uses. Robustness
diagnostics and the one-time holdout evaluation are run separately and
deliberately (see the final experiment report for this project's actual
run).
