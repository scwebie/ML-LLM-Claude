# V0.2 Real-Data Experiment Report

**PAPER TRADING / SIMULATION ONLY.**

> **CORRECTION NOTICE (2026-09-01).** The version of this report previously
> delivered was computed with a buggy `split_development_and_holdout()`
> that let observations **after** the holdout window leak back into
> `development_df`. That run's leakage window was small (ingestion ended
> 2025-08-15, ~6 weeks past the 2025-06-30 holdout end), but it was leakage
> nonetheless, and the same bug is catastrophic at the CLI's real
> default (`end=today`), where it raised `ValueError: fold 1: validation
> window [2024-03-04, 2026-02-05] overlaps the holdout period [2024-07-01,
> 2025-06-30]` and made `real-demo` unusable with its own default end date.
> The bug, its fix (a hard three-way `TemporalPartition` split — pre-holdout
> development / fixed holdout / post-holdout forward data, with the
> pre-holdout cutoff enforced *before* purge/embargo rather than relying on
> embargo alone), and 8 new regression tests are described in Section 0
> below. **Every number in this document past Section 0 is from a fresh
> `real-demo` run against the corrected code**, and several conclusions —
> most importantly the out-of-sample signal verdict — changed materially
> as a result. Per the project brief: **profitability was not required to
> declare engineering success, and results are reported without
> cherry-picking**, including where the corrected numbers are worse than
> what was previously reported.

---

## 0. The post-holdout leakage bug: report and fix

**Symptom.** Running `uv run python main.py real-demo` with its real
default end date (today) raised `assert_no_fold_touches_holdout`'s guard:
a walk-forward fold's validation window extended almost to the present,
crossing into the fixed 2024-07-01/2025-06-30 holdout. The guard did
exactly its job and correctly refused to proceed.

**Root cause.** `split_development_and_holdout()` computed
`development_mask = (~holdout_mask) & eligible`, which excludes rows
*inside* the holdout interval but not rows *after* it. `eligible` (purge +
embargo) only removes a narrow window immediately following the holdout
for label-overlap reasons; it was never designed to exclude "everything
from over a year in the future." Real ingestion in `real-demo` runs
through today by design, so essentially all post-holdout data (14+ months
of it, at the default end date) silently entered `development_df`, and
`generate_purged_folds` — which builds its calendar from
`development_df`'s own timestamps — produced folds reaching into the
present, overlapping the holdout.

**Fix (`backtesting/holdout.py`).** Added `split_temporal_partitions()`,
which returns an explicit `TemporalPartition(development_df, holdout_df,
post_holdout_df)` and enforces a **hard pre-holdout cutoff**
(`timestamp < holdout.start_date`) as a first-class filter, combined with
(not superseded by) the existing purge/embargo mask. `post_holdout_df` is
preserved, never silently discarded, and is asserted to contain only rows
strictly after `holdout.end_date`. Internal invariants
(`development_df.max() < holdout.start`, holdout bounds, post-holdout
lower bound) are asserted inside the function itself, not left to callers.
`split_development_and_holdout()` is now a two-value compatibility
wrapper over the same corrected logic — no divergent code path exists.
`assert_no_fold_touches_holdout` was strengthened to also reject a fold
that doesn't *end* strictly before the holdout start (previously it only
checked literal date-range overlap, which is not the same as "generated
entirely from legitimate pre-holdout data"). **This guard was not
weakened at any point** — the fix is entirely in what data reaches fold
generation in the first place.

**`real_pipeline.py`.** `run_real_demo()` gained an explicit final stage:
after model selection (walk-forward + champion qualification) is
completely finished on `development_df` and the selected model is frozen,
`evaluate_on_holdout()` is called **exactly once** on the untouched
`holdout_df`, using the same trained model object (never retrained). The
pipeline is now: pre-holdout development -> purged/nested walk-forward ->
champion qualification -> freeze model -> **one formal holdout
evaluation** -> holdout access log. The previous development-fold paper
backtest still runs, but is now explicitly labelled a **diagnostic**, not
the final out-of-sample result.

**Regression tests added (8 new, all passing):**

| Test | Proves |
|---|---|
| `test_post_holdout_exclusion_three_way_partition_boundaries` | The three regions partition correctly at their exact boundaries. |
| `test_split_development_and_holdout_wrapper_matches_three_way_partition` | The compat wrapper never diverges from the new function. |
| `test_no_fold_may_cross_the_holdout_even_when_source_data_extends_past_it` | Direct reproduction of the reported production failure (data to 2026, holdout 2024-07-01/2025-06-30) — no fold overlaps. |
| `test_assert_no_fold_touches_holdout_rejects_a_fold_entirely_after_holdout` | A fold entirely past the holdout (no literal overlap) is still rejected. |
| `test_post_holdout_mutation_does_not_affect_development_partition_or_model_selection` | Rewriting all post-holdout values byte-for-byte does not change the trained model's predictions (anti-leakage). |
| `test_holdout_mutation_does_not_affect_development_partition_or_model_selection` | Rewriting holdout target values does not change pre-holdout selection/champion qualification. |
| `test_model_selection_causes_zero_holdout_access_log_entries_and_formal_evaluation_adds_exactly_one` | Selection touches the holdout log 0 times; formal evaluation adds exactly 1. |
| `test_evaluate_real_step_never_crosses_holdout_when_data_extends_far_past_it` | Same regression at the `real_pipeline.py` orchestration layer, ~300 real trading days of post-holdout data. |

**Verification, in the order requested:**

| Command | Result |
|---|---|
| `uv run pytest -q` | **305 passed**, 0 failed (297 previously + 8 new) |
| `uv run pytest tests/test_holdout.py -v` | **19 passed** (12 previously + 7 new) |
| `uv run pytest -k "purge or embargo or temporal or leakage or holdout" -v` | **47 passed**, 258 deselected |
| `uv run ruff check .` | **All checks passed** |
| `uv run python main.py real-demo` (default `end=today`) | **Exit 0.** Previously failed with the `ValueError` above; now completes and produces the results in Sections 1-6. |

---

## 1. Engineering results

**Build:** All 17 staged phases of the V0.2 extension were implemented
and independently verified, plus the Section 0 correction above.
`uv run pytest -q` -> **305 passed**, 0 failed, 0 skipped. `uv run ruff
check .` -> **all checks passed**. `main.py demo` (V0.1's original
synthetic pipeline) is untouched and still exercised within that same 305.

**Two real bugs were found by running real data, both fixed with
regression tests** — this is the second. The first (Stage 17, prior to
this correction): `features/cross_sectional.py::compute_percentile_ranks`
assigned a ranked `pandas.Series` into a freshly-indexed output
`DataFrame` without resetting the index, causing every cross-sectional
percentile feature to be ~100% missing under `groupby("timestamp")`
non-contiguous indices; fixed with `.to_numpy()` positional assignment.
The second is the post-holdout leakage bug described in full in Section 0.
Neither would have been found on synthetic smoke tests or by inspection
alone — both required running the full pipeline at real scale, in the
second case specifically with `end=today` rather than a fixed historical
end date.

**CLI:** All seven V0.2 commands (`ingest-prices`, `ingest-fundamentals`,
`ingest-macro`, `ingest-news`, `build-real-features`, `evaluate-real`,
`real-demo`) ran successfully against real data end to end, `real-demo`
now including the corrected three-way split and the formal holdout stage.

**Risk-engine integration:** `tests/test_real_demo_risk_integration.py`
proves, with real ticker symbols and the unmodified production Portfolio/
Risk/Execution engine, that the risk engine produces both approvals and
rejections spanning multiple categories under tight limits. The
diagnostic backtest below used V0.1's default (looser) `RiskLimits`,
under which 583/584 proposed orders were approved (1 rejected,
`RISK_POSITION_LIMIT`) — direct evidence the risk engine is active even
under loose default limits in this run.

---

## 2. Data coverage

Universe: the default 20-symbol `DEFAULT_REAL_UNIVERSE` (AAPL, MSFT,
AMZN, GOOGL, META, NVDA, TSLA, JPM, V, UNH, HD, PG, MA, JNJ, XOM, COST,
MRK, ABBV, KO, PEP) plus the SPY benchmark. **Date range for this
corrected run: 2020-01-02 to 2026-08-31** (`real-demo`'s real defaults —
start 2020-01-01, end today — not the earlier report's fixed
2019-01-02/2025-08-15 window; this is precisely the "ingest through today"
behavior the fix was required to support without truncating it).

| Table | Rows |
|---|---|
| `market_observations` | 34,797 (21 symbols x ~1,674 bars each; MRK starts 2021-06-03 with 1,317) |
| `price_reconciliation` | 35,154 (34,784 `VALIDATED`, 357 `MAJOR_DIFFERENCE`, 13 `PRIMARY_MISSING`) |
| `fundamental_observations` | 712, 19/20 symbols (V unresolved — same known SEC EDGAR ticker-to-CIK gap as before) |
| `macro_observations` | 10,195 |
| `news_articles` | 1,341 (SEC 8-K) |
| `event_probability_observations` | 333 (Polymarket, read-only) |
| `event_symbol_mappings` | 6,660 |

**Honest gaps, carried forward and re-confirmed on this run:**

- **V (Visa) has zero fundamental observations** — SEC EDGAR ticker-to-CIK
  lookup still does not resolve for this symbol; excluded from
  fundamentals-derived features rather than imputed.
- **The five `eventprob_*` prediction-market feature columns are
  effectively 100% missing and were dropped before training**
  (`eventprob_monetary_policy_probability`,
  `eventprob_economic_outcomes_probability`,
  `eventprob_elections_policy_probability`,
  `eventprob_geopolitical_probability`,
  `eventprob_regulatory_probability`) — the same structural limitation as
  before: Polymarket's read-only endpoint only returns currently active
  markets, so it has no historical archive for point-in-time as-of lookups
  over 2020-2026 history.
- **Per-provider ingestion status (BLS/BEA/etc.) for this specific
  extended 2020-2026 run was not separately re-captured** in this
  correction pass — the underlying provider integrations are unchanged
  from Stage 3-9 and their own test suites remain green; this document
  does not repeat the earlier report's per-provider pass/fail table since
  it was measured on a different (shorter) ingestion run and re-asserting
  it here without re-observing it would not be honest.

**Feature matrix:** 21,843 (development) + 5,000 (holdout) + 5,880
(post-holdout) = **32,723 total rows** stored under
`feature_version=real_fv1`. 5 columns (`eventprob_*` above) were
automatically dropped as effectively missing; **126 features were used**.

---

## 3. Evaluation methodology (corrected three-way split)

Per `docs/evaluation_v02.md` and the Section 0 fix. Concretely, for this
run, with `HoldoutConfig(start_date=2024-07-01, end_date=2025-06-30)`
(unchanged fixed dates — **not altered to make anything pass**):

- **Pre-holdout development set:** **21,843 rows**, 2020-01-02 ->
  2024-05-30. `development_df["timestamp"].max() < holdout.start_date`
  asserted and holds (2024-05-30 < 2024-07-01).
- **Final holdout set:** **5,000 rows**, exactly 2024-07-01 -> 2025-06-30.
  `holdout_df["timestamp"].min() >= holdout.start_date` and
  `holdout_df["timestamp"].max() <= holdout.end_date` asserted and hold.
- **Post-holdout forward data:** **5,880 rows**, 2025-07-01 -> 2026-08-31.
  Preserved (not discarded), `post_holdout_df["timestamp"].min() >
  holdout.end_date` asserted and holds. **Not used in any model-selection,
  feature-selection, hyperparameter, or champion-qualification decision**
  in this run — reserved for future genuine forward-paper analysis only.
- **Purged+embargoed walk-forward folds** (expanding window, generated
  only from the pre-holdout development calendar):

  | Fold | Train start | Validation window |
  |---|---|---|
  | 0 | 2020-01-02 | 2022-08-24 -> 2023-04-21 |
  | 1 | 2020-01-02 | 2023-04-24 -> 2023-12-18 |

  **Confirmed: no fold touches the holdout.** `assert_no_fold_touches_holdout`
  passed for both, and both additionally satisfy the stricter check
  `fold.validation_end < holdout.start_date` explicitly re-verified
  (2023-04-21 and 2023-12-18, both well before 2024-07-01).
- **Champion selection:** challenger `lgbm_v0001`, evaluated against the
  initial-qualification bar (`learning/initial_qualification.py`).
  **Decision: REJECTED** (no existing champion to compare against, and the
  challenger itself failed the bar). **Rationale (verbatim from
  `promotion_log`):** *"no existing champion, and the challenger failed
  the initial-qualification bar: permutation test p-value=0.2680 exceeds
  0.1 -- observed IC is not distinguishable from noise."* Recorded at
  `2026-09-01 07:23:01 UTC`.
- **Holdout access log:** **0 accesses before or during model
  selection**; **exactly 1 formal access**, recorded at
  `2026-09-01 07:23:30 UTC` — **after** the promotion decision above (29
  seconds later), confirming the ordering requirement. Logged purpose:
  *"real-demo CLI: final formal out-of-sample evaluation after model
  selection was completed."* `n_rows=5000`, `model_version=lgbm_v0001`,
  `holdout_start=2024-07-01`, `holdout_end=2025-06-30`. This is the only
  row in `holdout_access_log` for this experiment.

---

## 4. Final holdout evaluation (the untouched out-of-sample result)

This is the one result that was **never available to influence model
selection** — the frozen, rejected-for-promotion `lgbm_v0001` model,
evaluated once against the 5,000-row holdout it had never seen:

| Target | RMSE / AUC | R² / Accuracy | Rank IC / Brier |
|---|---|---|---|
| `excess_return_5d` | RMSE 0.0426 | R² -0.0224 | **IC +0.0711** |
| `excess_return_20d` (primary) | RMSE 0.0834 | R² -0.0114 | **IC -0.0225** |
| `positive_5d` | AUC 0.559 | Acc 0.525 | Brier 0.249 |
| `positive_20d` | AUC **0.434** | Acc 0.502 | Brier 0.256 |

**On the primary target, `excess_return_20d`, the formal holdout IC is
slightly negative (-0.0225), and the directional target's AUC
(`positive_20d`, 0.434) is below 0.5 — worse than random.** The 5-day
horizon shows a small positive IC (+0.071), but this is one number among
several targets and was not the target the champion-qualification bar
scored on. There is no cherry-picking here: all four target/metric
combinations produced by the same single, logged, one-time evaluation are
reported above.

---

## 5. Development-side diagnostics (NOT the final OOS result)

These are re-run for transparency and to show the robustness-suite
machinery still works, but per Section 0's corrected pipeline **none of
this is the formal out-of-sample result — that is Section 4.** All
numbers below come from the two pre-holdout walk-forward folds only.

**Pooled development OOS predictions** (both folds concatenated, n=949
symbol-date pairs):

- Rank IC (`excess_return_20d`): **0.0282**
- Permutation test (2,000 shuffles): **p=0.3875** — not significant. This
  is materially weaker than the p=0.0015 previously reported, and is
  consistent with (not contradicted by) the holdout's near-zero IC in
  Section 4.
- Block-bootstrap CI on the **daily** IC series (n=135 trading days):
  point estimate 0.155, 95% CI **[0.025, 0.288]** — entirely positive,
  in tension with the pooled permutation test above. Reported honestly
  rather than resolved in either direction: a two-fold, 949-observation
  pooled test and a 135-day block-bootstrap are different statistics with
  different power, and with only 2 folds this tension itself is a sign of
  an unstable estimate, not proof of a real edge.
- By year: **2022 IC=0.180** (n=235), **2023 IC=-0.015** (n=714) — sign
  flips between the two years covered, and the larger-sample year (2023)
  is the one closer to zero.
- Factor exposure (predicted signal regressed on momentum/liquidity/
  volatility percentiles): R²=0.035, loadings momentum +0.120, liquidity
  -0.014, volatility -0.158 — not simply a repackaged momentum factor, but
  the R² is low enough that this is weak evidence either way.
- Execution-delay stress: rank IC is 0.028 at 0-day delay and **rises**
  to 0.034 (1d), 0.036 (2d/3d), 0.031 (5d), and **0.103 at 10-day delay**.
  A genuine, decaying alpha signal should weaken with execution delay;
  IC *increasing* at a 10-day delay is a red flag for noise/instability
  on this small sample, not a confirmation of signal.
- Feature-importance stability across the 2 folds: mean pairwise Spearman
  **0.187** — low.
- Calibration (`positive_20d`, 10 bins): several bins are inverted (e.g.
  predicted 0.500 -> realised 0.400; predicted 0.524 -> realised 0.338),
  consistent with a model that is not reliably calibrated at this sample
  size.
- Cost-stress test on the pooled development quantile long/short
  portfolio: **raw gross Sharpe 4.09** (no cost), staying positive
  (1.26) even at 100bps round-trip. This number is treated with
  suspicion, not presented at face value: it is computed on the same
  short, bull-market-heavy 2-fold window as the diagnostic backtest below,
  which itself underperformed buy-and-hold; a portfolio-level Sharpe this
  high on ~949 observations concentrated in a strong 2023 rally is far
  more consistent with a small, favorable sample than with a real
  tradeable edge, and is contradicted by the near-zero IC on the actual
  holdout in Section 4.

**Diagnostic paper-trading backtest** (last walk-forward fold's
validation window only, 2023-04-25 to 2023-12-18 — **this is a
development-side diagnostic of the pipeline wiring, not a claim about
holdout-period performance**, since no portfolio backtest was run over
the holdout itself in this pass):

| | ML Strategy | Equal-weight Buy & Hold (same 20 symbols) |
|---|---|---|
| Total return | +1.55% | +19.96% |
| CAGR | +2.39% | +32.27% |
| Max drawdown | -1.00% | -9.16% |
| Sharpe ratio | 1.18 | 2.28 |
| Orders approved / rejected | 583 / 1 (`RISK_POSITION_LIMIT`) | -- |

The strategy substantially underperformed buy-and-hold over this specific
bull-market window, on both absolute and risk-adjusted terms — consistent
with, not contradicting, the weak/absent signal found in Sections 4-5.

**Post-holdout forward-paper analysis:** `post_holdout_df` (5,880 rows,
2025-07-01 -> 2026-08-31) is preserved and available in the database, but
**no forward-paper simulation was run over it in this pass** — it was
deliberately left untouched, exactly as required, rather than used to
improve or re-litigate the frozen model's holdout result. It remains
available for a genuinely prospective analysis in a later, separate run.

---

## 6. Safety confirmations

Re-verified for this corrected run:

- **No real brokerage, no real trade, anywhere** — `execution/paper.py`'s
  in-process simulated broker only.
- **No prediction-market execution capability** — Polymarket integration
  remains read-only (`tests/test_prediction_market_readonly.py`).
- **The LLM did not control risk or execution** — `use_llm=False`
  (default); no research agent imports `execution/` or `portfolio/risk.py`.
- **The risk engine is deterministic and the only path to a fill** —
  every one of the 584 proposed orders in the diagnostic backtest passed
  through `RiskEngine.evaluate_order`, with 1 genuine rejection.
- **The final holdout was accessed exactly once, after and never before
  model selection** — `holdout_access_log` contains exactly 1 row,
  timestamped 29 seconds after the promotion decision; this was
  structurally verified (not just observed) by
  `test_model_selection_causes_zero_holdout_access_log_entries_and_formal_evaluation_adds_exactly_one`.
  Two additional structural anti-leakage tests
  (`test_post_holdout_mutation_...` and `test_holdout_mutation_...`)
  proved, by mutating each excluded region and re-running, that neither
  post-holdout nor holdout data can influence model selection even in
  principle, not merely that it didn't in this one run.
- **No real data was fabricated.** Every gap in this report (V's
  fundamentals, `eventprob_*` history) is a genuine provider/data
  limitation, never a substituted synthetic value.

---

## Verdicts

**V0.2 ENGINEERING: PASS**

All 17 staged phases implemented; 305/305 tests passing (including 8 new
regression tests for the post-holdout leakage bug); ruff clean; all seven
CLI commands verified against live real data, including `real-demo` with
its real default `end=today`, which previously failed outright; two real,
previously-undetected defects were found by running real data at real
scale and real default settings (not by inspection), root-caused, fixed,
and each covered by regression tests that fail under the old code; the
risk engine is proven to both approve and reject orders; the final
holdout was accessed exactly once, structurally proven (not just
observed) to be unreachable from model selection, and only after
selection was complete.

**EVIDENCE OF OUT-OF-SAMPLE SIGNAL: NONE / INCONCLUSIVE**
*(previously reported as WEAK — that verdict is superseded by this
correction and should not be relied on.)*

The pooled development-side permutation test is not significant
(p=0.3875, vs. the earlier leak-affected report's p=0.0015), by-year IC
sign-flips (2022 +0.18, 2023 -0.02), feature-importance stability across
folds is low (0.187), and the execution-delay stress test shows IC rising
rather than decaying with delay — none of this looks like a stable,
decaying real-world alpha signal. Most importantly, **the one evaluation
that was genuinely never available to the model or its selection — the
formal holdout in Section 4 — shows an IC of -0.0225 on the primary
20-day target and a directional AUC of 0.434, below random.** The
champion-qualification bar correctly rejected this model. Where the
pooled bootstrap CI on daily IC came out entirely positive [0.025, 0.288],
that is reported too, honestly, as a genuine tension in the evidence
rather than resolved in whichever direction looks better — but it does
not outweigh a negative result on the untouched holdout itself. The
overall weight of evidence does not support claiming a real, exploitable
signal at this stage.

**READY FOR CONTINUED PAPER TRADING: YES**

The engineering, safety, and evaluation infrastructure are sound —
including, as of this correction, a temporal-partitioning implementation
that is now structurally (not just procedurally) proven not to leak
future or holdout data into model selection. Champion/challenger, drift
monitoring, and the robustness suite exist precisely to keep testing
whether a promotable signal emerges as more data, feature refinement, and
evaluation cycles accumulate; the qualification bar correctly declined to
promote here. **Do NOT recommend real-money deployment based on this
experiment** — no model cleared the promotion bar, and the untouched
holdout showed no evidence of a tradeable edge.
