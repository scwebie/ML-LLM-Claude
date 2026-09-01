# V0.2 Real-Data Experiment Report

**PAPER TRADING / SIMULATION ONLY.** This report documents an actual run
of `uv run python main.py real-demo` against real market data (Yahoo
Finance, StockAnalysis.com, SEC EDGAR, FRED/BLS/Treasury, SEC 8-K,
Polymarket read-only), 2019-01-02 through 2025-08-15, on the default
20-symbol universe. Results are reported as they came out, including a
real bug this run itself surfaced and a rejected promotion. Per the
project brief: **profitability was not required to declare engineering
success, and results are reported without cherry-picking.**

---

## 1. Engineering results

**Build:** All 17 staged phases of the V0.2 extension were implemented
and independently verified. `uv run pytest -q` -> **297 passed**, 0
failed, 0 skipped. `uv run ruff check .` -> **all checks passed**.
`main.py demo` (V0.1's original synthetic pipeline) is untouched and
still exercised by the existing V0.1 test suite within that same 297.

**A real bug, found by running real data, fixed with a regression test.**
The single most important engineering finding of this experiment: the
first live `real-demo` run silently produced a feature matrix in which
every cross-sectional percentile-rank feature (`momentum_percentile`,
`value_percentile`, `quality_percentile`, `growth_percentile`,
`volatility_percentile`, `liquidity_percentile`,
`earnings_quality_percentile`) was **~100% missing**, even though their
source columns (`dollar_volume`, `return_60d`, ...) were fully populated.
Root cause: `features/cross_sectional.py::compute_percentile_ranks`
assigned a ranked `pandas.Series` into a freshly-indexed output
`DataFrame` without resetting the index; pandas aligns column assignment
by index *label*, not position, and a `groupby("timestamp")` split (the
only way this function is ever actually called) hands each group its
*original*, non-contiguous row indices from the larger frame -- so the
assignment silently produced `NaN` everywhere the original index didn't
happen to coincide with the output's fresh `0..n-1` index. The existing
unit test suite did not catch this because its one multi-date test
happened to assert only on the date whose group's original index started
at 0 (a coincidence of construction order, not by design).

- **Fix:** `.to_numpy()` on the ranked Series before assignment, forcing
  positional (not label) alignment.
- **Regression tests added:** a corrected/strengthened existing test
  (`test_percentile_ranks_computed_independently_per_date`, now also
  asserting the second date's group) plus a new, explicit
  `test_compute_percentile_ranks_is_positional_not_index_aligned` that
  constructs exactly the failure condition (a non-zero-starting index)
  and would fail under the old code.
- **Verified fixed against live data:** after the fix, a fresh
  `real-demo` run against the same real data showed `liquidity_percentile`
  at 0% missing (previously 99.94%) and every other percentile feature
  populated in proportion to its underlying source data's real coverage
  (1-17% missing, matching genuine fundamentals gaps -- see Data
  Coverage below), not the previous near-total failure.
- This is exactly the kind of defect the spec's staged, test-first, then
  real-data-verified process exists to catch, and it would not have been
  found without actually running the full pipeline against real data at
  real scale (the bug was invisible on my earlier 2-3-symbol synthetic
  smoke tests too, for the same reason -- their date-groups also
  happened to start with a small enough index).

**CLI:** All seven V0.2 commands (`ingest-prices`, `ingest-fundamentals`,
`ingest-macro`, `ingest-news`, `build-real-features`, `evaluate-real`,
`real-demo`) ran successfully against real data end to end.

**Risk-engine integration:** `tests/test_real_demo_risk_integration.py`
proves, with real ticker symbols and the unmodified production Portfolio/
Risk/Execution engine, that the risk engine produces **both** approvals
(fills) **and** rejections spanning multiple distinct categories
(position limit, sector concentration, gross/net exposure) in one
realistic run under tight limits. The actual `real-demo` run reported
below used V0.1's *default* (looser) `RiskLimits`, under which every one
of 690 proposed orders was approved and filled -- zero rejections is the
correct, expected outcome at that limit configuration, not evidence the
risk engine is inactive; see the dedicated integration test for the
proof it rejects when limits actually bind.

---

## 2. Data coverage

Universe: the default 20-symbol `DEFAULT_REAL_UNIVERSE` (AAPL, MSFT,
AMZN, GOOGL, META, NVDA, TSLA, JPM, V, UNH, HD, PG, MA, JNJ, XOM, COST,
MRK, ABBV, KO, PEP) plus the SPY benchmark. Date range: 2019-01-02 to
2025-08-15 (2,391 calendar days; the final holdout window, 2024-07-01 to
2025-06-30, sits entirely inside this range with a ~6-week buffer after
it for 20-day-forward target computation).

| Source | Status | Records |
|---|---|---|
| Yahoo Finance (prices) | SUCCESS, 21/21 symbols | 34,356 bars |
| StockAnalysis.com (price cross-check) | used for reconciliation | -- |
| SEC EDGAR (fundamentals) | SUCCESS, 19/20 symbols | 712 observations |
| SEC 8-K (news) | SUCCESS, 20/20 symbols | 1,239 articles |
| FRED (macro) | SUCCESS | 10,055 observations |
| US Treasury Fiscal Data | SUCCESS | 79 observations |
| BLS (macro) | **FAILED this run** -- daily unregistered-tier request cap exceeded | 0 |
| BEA (macro) | UNAVAILABLE (no `BEA_API_KEY` configured) | 0, by design |
| Polymarket (read-only event probabilities) | SUCCESS | 333 events, 6,660 symbol mappings |

**Honest gaps, not fabricated substitutes:**

- **BLS failed this specific run** with `"the daily threshold for total
  number of requests allocated to the user... has been reached"` -- the
  unregistered-tier cap documented in `docs/provider_setup.md` was
  exhausted (by an earlier run in the same session). CPI/unemployment/
  payroll series were therefore unavailable this run; VIX, Treasury
  yields, and Treasury issuance data (from FRED/Treasury, unaffected)
  were still ingested. No value was substituted -- the affected macro
  features are genuinely absent from this run's feature matrix, not
  faked. A `BLS_API_KEY` (free to register) removes this cap.
- **One symbol, V (Visa), has zero fundamental observations** -- the SEC
  EDGAR ticker-to-CIK lookup did not resolve for this symbol in this run;
  the other 19/20 symbols resolved correctly. `V`'s fundamentals-derived
  features are `NaN` for that symbol, correctly excluded from training
  rows rather than imputed.
- **The five `eventprob_*` (read-only prediction-market) feature columns
  were >99% missing and automatically dropped before training.** This is
  an inherent, structural limitation of the data source, not a bug:
  Polymarket's `get_active_events` endpoint returns only *currently
  active* markets, so every observation's timestamp is close to the
  actual ingestion date (2026) -- there is no historical archive of past
  resolved markets available through this endpoint. For a backtest over
  2019-2024/2025 history, the point-in-time as-of lookup
  (`data/real_features.py::_lookup_asof`) correctly returns "unknown" for
  every one of those historical rows, since the observation genuinely
  didn't exist yet at that point in time. This signal is real and
  correctly wired (see `tests/test_prediction_market_readonly.py` and
  `tests/test_real_features.py`), but is only ever informative for
  research conducted close to real time, not for historical backtesting
  -- a limitation of the source, honestly surfaced rather than hidden.
- **Price reconciliation:** of 34,986 reconciled bars, 34,335 (98.1%)
  validated cleanly between Yahoo and StockAnalysis.com, 609 (1.7%) were
  flagged `MAJOR_DIFFERENCE`, and 21 (0.06%) had `PRIMARY_MISSING`. Flagged
  bars are excluded from training via `filter_trainable_bars`, not
  silently trusted from either source.
- **MRK (Merck)'s price history starts 2021-06-03**, not 2019-01-02 --
  the earliest history either source actually returned for that symbol.

Feature matrix: **32,691 rows x 137 columns** stored under
`feature_version=real_fv1`. 5 columns (the `eventprob_*` family above)
were automatically excluded before training as effectively missing;
**126 features were used**, covering technical, fundamental,
macro (raw values + regime codes), cross-sectional percentile ranks,
market breadth, and deterministic news features.

---

## 3. Evaluation methodology

Per `docs/evaluation_v02.md`. Concretely, for this run:

- **Development set:** 27,191 rows, everything before the holdout with
  purge (20-trading-day target horizon) and embargo (5 trading days)
  applied at the holdout boundary.
- **Holdout set:** 5,000 rows, the fixed 2024-07-01 to 2025-06-30 window,
  never touched during model selection.
- **Purged+embargoed walk-forward folds** (expanding window, 60%
  initial-train / 15% validation fractions of the development calendar):

  | Fold | Train start | Validation window |
  |---|---|---|
  | 0 | 2019-01-02 | 2022-04-25 -> 2023-02-21 |
  | 1 | 2019-01-02 | 2023-02-22 -> 2023-12-18 |

  `assert_no_fold_touches_holdout` confirmed neither fold's validation
  window overlaps the holdout period.
- **Initial-champion qualification bar** (`learning/champion_challenger_v2.py`):
  evaluated on the final fold's model. **Result: REJECTED.**
  `sharpe_ratio=-1.0813 below minimum 0.3` -- the model's own
  out-of-sample quantile-portfolio Sharpe on its validation fold was
  negative, so the stricter V0.2 gate correctly declined to promote it
  to champion rather than auto-promoting the first candidate. This is
  the gate working exactly as designed (see Stage 14): a positive but
  modest information coefficient alone is not sufficient.
- **Holdout evaluation:** exactly **one** deliberate, logged access
  (`holdout_access_log`: purpose "V0.2 final experiment report --
  one-time out-of-sample confirmation", 5,000 rows), performed only
  after development-side model selection was complete. This is the only
  access to the holdout period anywhere in this experiment.

---

## 4. Model performance vs. benchmarks

**Alpha-model out-of-sample metrics** (target: `excess_return_20d`
unless noted):

| | Development (pooled OOS, n=1,204) | Holdout (n=815-5,000 depending on target) |
|---|---|---|
| Rank IC (`excess_return_20d`) | **0.0949** | **0.0665** |
| Permutation test p-value | **0.0015** | **0.051** |
| `positive_5d` AUC | 0.539 | 0.499 |
| `positive_20d` AUC | 0.582 | 0.526 |
| `positive_20d` Brier score | 0.243 | 0.250 |

**Paper-trading backtest vs. buy-and-hold** (last walk-forward fold's
validation window, 2023-02-23 to 2023-12-18, real prices, real Portfolio/
Risk/Execution engine, default `RiskLimits`/`AllocationConfig`,
$1,000,000 starting cash):

| | ML Strategy | Equal-weight Buy & Hold (same 20 symbols) |
|---|---|---|
| Total return | **+1.45%** | **+26.37%** |
| CAGR | +1.78% | +33.15% |
| Max drawdown | -1.62% | -8.94% |
| Sharpe ratio | 0.92 | 2.31 |
| Orders approved / rejected | 690 / 0 | -- |

**The strategy substantially underperformed a simple buy-and-hold of the
same universe over this specific ~10-month window** -- 2023 was a strong
bull market for this large-cap-tech-heavy universe, and the strategy's
conservative net exposure (bounded by `RiskLimits`) captured only a
fraction of that move, on both an absolute and risk-adjusted (Sharpe)
basis. This is reported plainly: **profitability was not required to
declare engineering success**, and it was not achieved here.

---

## 5. Robustness results

All computed on genuine out-of-sample predictions (`backtesting/robustness.py`,
Stage 13), on the corrected data:

- **Bootstrap CI (block bootstrap, daily rank IC series, n=218 trading
  days):** point estimate **0.027**, 95% CI **[-0.092, 0.130]**. The
  interval straddles zero -- day-to-day predictive skill is not
  statistically distinguishable from noise, even though the *pooled*
  correlation across all 1,204 (symbol, date) pairs is significant. This
  is the clearest single piece of evidence that the signal, while real in
  aggregate, is not a reliably exploitable day-to-day edge.
- **Permutation test:** development p=0.0015 (real signal, 2,000
  shuffles), holdout p=0.051 (borderline, just above the conventional
  0.05 threshold).
- **Year-by-year:** IC 0.076 (2022) -> 0.146 (2023) -- present in both
  years covered by walk-forward validation, stronger in 2023.
- **Factor exposure:** predicted signal regressed on momentum/liquidity/
  volatility percentiles gives R²=0.055 -- only ~5.5% of the prediction's
  variance is explained by these three known factors, with a negative
  momentum loading (-0.19) and positive liquidity/volatility loadings
  (+0.23/+0.14). The signal is not simply a repackaged momentum factor,
  but the loadings are non-trivial and worth further ablation in future
  work.
- **Transaction-cost stress test:** the quantile long/short portfolio's
  **gross Sharpe was already negative (-0.96) before any costs**, and
  monotonically worsens with cost (-3.04 at 100bps round-trip). This is
  consistent with the development-period backtest underperformance above
  and is the most direct evidence against a currently tradeable edge on
  the development side.
- **Execution-delay stress test:** rank IC decays from 0.095 (no delay)
  to 0.013 (5-day delay) to **-0.024 (10-day delay)** -- whatever signal
  exists is short-lived and requires prompt execution; it does not
  survive acting on it a week and a half late.
- **Feature-importance stability:** mean pairwise Spearman correlation
  0.644 between the two folds' feature-importance rankings -- moderate,
  not high, stability.
- **Calibration** (`positive_20d`): predicted probabilities track
  realised positive rates reasonably across bins (e.g. predicted 0.40 ->
  realised 0.41; predicted 0.57 -> realised 0.59), with some bin-to-bin
  noise expected at these sample sizes.
- **Feature-family ablation and the negative-control check** are
  implemented and unit-tested (35 dedicated tests in
  `tests/test_robustness.py`, all passing) but were **not re-run against
  this full 20-symbol/6-year dataset** in this pass, due to the added
  compute cost of retraining per ablated family at this scale --
  disclosed here rather than presented as done.
- **The holdout-period quantile portfolio's gross Sharpe was +1.95** --
  *positive*, in direct contrast to the negative development-period
  Sharpe above. This inconsistency across periods, combined with the wide
  development-period bootstrap CI, is reported as evidence of
  **instability**, not as a second confirmation of a real edge: a metric
  that swings from strongly negative to strongly positive across two
  different market periods is not behaving like a stable, reliable
  signal.

---

## 6. Safety confirmations

Explicitly re-verified for this experiment, on top of the structural
tests that enforce them continuously:

- **No real brokerage, no real trade, anywhere.** The `real-demo` run
  used `execution/paper.py`'s in-process simulated broker exclusively;
  no HTTP client or credential to any brokerage exists in this codebase.
- **No prediction-market execution capability.** The Polymarket
  integration used in this run is read-only, verified structurally: its
  complete public API is exactly `{source_id, get_active_events}`
  (`tests/test_prediction_market_readonly.py::test_public_api_is_exactly_the_declared_allow_list`),
  and a second independent test scans for any execution-shaped method
  name (order/buy/sell/wager/wallet/...) anywhere on the class, public or
  private -- both passed as part of the 297-test suite run for this
  experiment.
- **The LLM did not control risk or execution.** This run used
  `use_llm=False` (the CLI default); even with it enabled, every research
  agent only ever contributes a bounded numeric score or an optional
  narrative string, never an order (`agents/*.py` import neither
  `execution/` nor `portfolio/risk.py`).
- **The risk engine is deterministic and was the only path to a fill.**
  Every one of the 690 approved orders in this run's backtest passed
  through `portfolio.risk.RiskEngine.evaluate_order`; see
  `tests/test_real_demo_risk_integration.py` for proof it also rejects,
  across multiple reason categories, when limits actually bind.
- **The final holdout was accessed exactly once**, logged, for this
  report -- `holdout_access_log` contains exactly one row.
- **No real data was fabricated.** Every gap in this report (BLS,
  BEA, V's fundamentals, eventprob_* history) is a genuine provider/data
  limitation, surfaced via `UNAVAILABLE`/dropped-column reporting, never
  a substituted synthetic value.

---

## Verdicts

**V0.2 ENGINEERING: PASS**

All 17 staged phases implemented; 297/297 tests passing; ruff clean; all
seven CLI commands verified against live real data; a real, previously-
undetected engineering defect (the cross-sectional ranking index-
alignment bug) was found by this real-data run itself, root-caused, fixed,
and covered by a regression test that fails under the old code; the risk
engine is proven, by a dedicated integration test, to both approve and
reject orders in a realistic multi-sector run; the final holdout was
accessed exactly once, logged, and only after development-side model
selection was complete.

**EVIDENCE OF OUT-OF-SAMPLE SIGNAL: WEAK**

There is a statistically detectable, same-signed relationship between
the model's `excess_return_20d` predictions and realised forward returns
in both the development set (pooled rank IC 0.095, permutation p=0.0015)
and the untouched holdout (rank IC 0.067, p=0.051) -- this is unlikely to
be pure chance. But the signal is not stable day to day (bootstrap CI
straddles zero), decays to negative within 10 trading days of execution
delay, produced a negative gross portfolio Sharpe before costs on the
development side (and a sharply inconsistent positive one on holdout),
and did not clear even a modest Sharpe-based promotion bar. This is real
but weak and not yet a reliable, tradeable edge.

**READY FOR CONTINUED PAPER TRADING: YES**

The engineering, safety, and evaluation infrastructure are sound enough
to keep running and iterating in a zero-financial-risk paper-trading
context -- champion/challenger, drift monitoring, and the robustness
suite exist precisely to keep testing whether a promotable signal
emerges as more data, feature refinement, and evaluation cycles
accumulate. **Do NOT recommend real-money deployment based solely on
this experiment** -- no model here cleared the promotion bar, and the
out-of-sample evidence found is explicitly weak and unstable, not a
basis for risking real capital.
