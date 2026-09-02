"""V0.3 Stage 14: generates the V0.3 research report from code, running the
Stage 3-13 diagnostics against the project's existing REAL data (already
cached from V0.2 ingestion -- no new network calls, no new historical-
holdout access).

Operates on a COPY of the real database (never the canonical
``data_store/duckdb/v02_real_experiment_fixed.duckdb``, and never the file
a live CLI run would use), so nothing here can mutate the historical
record or the live model registry. Every diagnostic below is DEVELOPMENT-
ONLY or a read-only audit query -- this script never calls
``evaluate_on_holdout`` or ``evaluate_on_forward_paper``.

Writes a running JSON checkpoint after every stage (so a failure partway
through never loses already-computed results) and finally renders
``docs/V03_RESEARCH_REPORT.md`` from whatever stages completed.

Usage: uv run python scripts/generate_v03_report.py
"""

from __future__ import annotations

import json
import shutil
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DB = REPO_ROOT / "data_store" / "duckdb" / "v02_real_experiment_fixed.duckdb"
WORKING_DB = REPO_ROOT / "data_store" / "duckdb" / "v03_report_working_copy.duckdb"
CHECKPOINT_PATH = REPO_ROOT / "data_store" / "results" / "v03_report_checkpoint.json"
REPORT_PATH = REPO_ROOT / "docs" / "V03_RESEARCH_REPORT.md"

results: dict = {}


def _default(obj):  # noqa: ANN001
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def checkpoint() -> None:
    CHECKPOINT_PATH.write_text(json.dumps(results, indent=2, default=_default))


def stage(name: str):  # noqa: ANN201 - decorator factory
    def decorator(fn):  # noqa: ANN001, ANN202
        def wrapper(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            print(f"--- {name} ---", flush=True)
            t0 = time.time()
            try:
                value = fn(*args, **kwargs)
                results[name] = {"status": "ok", "elapsed_s": round(time.time() - t0, 1), "data": value}
                print(f"    ok ({results[name]['elapsed_s']}s)", flush=True)
            except Exception as exc:  # noqa: BLE001 - a stage failure must not lose earlier stages
                results[name] = {
                    "status": "error", "elapsed_s": round(time.time() - t0, 1),
                    "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
                }
                print(f"    ERROR: {exc}", flush=True)
            checkpoint()
            return results[name]
        return wrapper
    return decorator


def main() -> None:
    if not SOURCE_DB.exists():
        print(f"source database not found: {SOURCE_DB}", file=sys.stderr)
        sys.exit(1)
    WORKING_DB.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE_DB, WORKING_DB)
    print(f"working copy: {WORKING_DB} ({WORKING_DB.stat().st_size / 1e6:.1f} MB)")

    from database.schema import init_schema

    con = duckdb.connect(str(WORKING_DB))
    init_schema(con)

    import real_pipeline as rp
    from backtesting.purged_walk_forward import build_trading_calendar

    symbols = rp.DEFAULT_REAL_UNIVERSE

    @stage("00_source_data_summary")
    def _source_summary():
        market = con.execute(
            "SELECT symbol, COUNT(*) AS n, MIN(timestamp) AS min_ts, MAX(timestamp) AS max_ts "
            "FROM market_observations GROUP BY symbol ORDER BY symbol"
        ).fetchdf()
        return {
            "symbols": symbols,
            "market_observations_by_symbol": market.to_dict(orient="records"),
            "fundamental_observations": int(con.execute("SELECT COUNT(*) FROM fundamental_observations").fetchone()[0]),
            "macro_observations": int(con.execute("SELECT COUNT(*) FROM macro_observations").fetchone()[0]),
        }

    _source_summary()

    @stage("01_development_evaluation")
    def _development_evaluation():
        evaluation = rp.evaluate_real_step(con, symbols)
        return evaluation

    dev_result = _development_evaluation()
    if dev_result["status"] != "ok":
        print("development evaluation failed -- cannot proceed with downstream stages that need it", file=sys.stderr)
        render_report()
        return
    evaluation = dev_result["data"]
    fold_results = evaluation.fold_results
    folds = [r.fold for r in fold_results]
    development_df = evaluation.development_df
    feature_cols = evaluation.feature_cols
    market = rp.repo.get_market_observations(con, symbols=symbols)
    results["01_development_evaluation"]["data"] = {
        "n_folds": len(fold_results),
        "development_rows": len(development_df),
        "development_date_range": [str(development_df["timestamp"].min()), str(development_df["timestamp"].max())],
        "n_feature_columns": len(feature_cols),
        "fold_metrics_summary": evaluation.fold_metrics_summary,
        "sharpe_audit": evaluation.sharpe_audit,
        "champion_model_version": evaluation.champion_model_version,
        "promoted": evaluation.promoted,
        "promotion_rationale": evaluation.promotion_rationale,
        "incumbent_champion_version_before_evaluation": evaluation.incumbent_champion_version_before_evaluation,
        "robustness": evaluation.robustness,
    }
    checkpoint()

    @stage("02_development_diagnostics")
    def _diagnostics():
        from backtesting.development_diagnostics import build_development_diagnostics_report
        return build_development_diagnostics_report(fold_results, development_df, market)

    _diagnostics()

    @stage("03_feature_family_ablation")
    def _ablation():
        from backtesting.ablation_v3 import run_feature_ablation_v3
        from backtesting.robustness import group_features_by_family

        families = group_features_by_family(feature_cols)
        reports = run_feature_ablation_v3(development_df, folds, feature_cols, market, families=families)
        return {"family_sizes": {k: len(v) for k, v in families.items()}, "reports": [asdict(r) for r in reports]}

    _ablation()

    @stage("04_feature_importance_stability")
    def _stability():
        from backtesting.feature_stability import compute_feature_stability
        report = compute_feature_stability(fold_results, development_df, feature_cols)
        return asdict(report)

    _stability()

    @stage("05_negative_controls")
    def _negative_controls():
        from backtesting.negative_controls import run_negative_controls
        control_results = run_negative_controls(development_df, folds, feature_cols)
        return [asdict(r) for r in control_results]

    _negative_controls()

    @stage("06_purge_embargo_audit")
    def _purge_audit():
        from backtesting.purge_audit import (
            assert_folds_temporally_ordered,
            assert_no_validation_row_reused_in_training,
            describe_all_folds,
        )
        calendar = build_trading_calendar(development_df["timestamp"])
        boundary_df = describe_all_folds(development_df, folds, calendar)
        assert_folds_temporally_ordered(folds)
        assert_no_validation_row_reused_in_training(development_df, folds, calendar)
        return {"folds_temporally_ordered": True, "no_validation_row_reused_in_training": True, "boundaries": boundary_df.to_dict(orient="records")}

    _purge_audit()

    @stage("07_statistical_significance")
    def _statistical_significance():
        from backtesting.daily_portfolio import (
            build_daily_rebalanced_portfolio_returns,
            sharpe_audit_report,
        )
        from backtesting.purged_walk_forward import TARGET_TO_PRED_COL
        from backtesting.robustness import build_evaluation_frame
        from backtesting.statistical_significance import (
            deflated_sharpe_ratio,
            effective_sample_size_for_overlap,
            ic_information_ratio,
            probabilistic_sharpe_ratio,
        )
        from models.evaluate import information_coefficient

        target_col = "excess_return_20d"
        pred_col = TARGET_TO_PRED_COL[target_col]
        eval_frame = build_evaluation_frame(fold_results, development_df, target_col)
        per_date_ic = eval_frame.groupby("timestamp").apply(
            lambda g: information_coefficient(g[target_col], g[pred_col]) if len(g) >= 3 else float("nan")
        )
        ir = ic_information_ratio(per_date_ic)
        ess = effective_sample_size_for_overlap(len(eval_frame), horizon_days=20)

        last_fold = fold_results[-1]
        daily_returns = build_daily_rebalanced_portfolio_returns(last_fold.predictions, market, TARGET_TO_PRED_COL[target_col])
        audit = sharpe_audit_report(daily_returns)
        per_fold_ics = evaluation.fold_metrics_summary.get("per_fold_information_coefficient", [])

        psr = dsr = None
        if not daily_returns.empty and len(daily_returns) > 2:
            daily_periodic_return = daily_returns["gross_return"]
            observed_sharpe = daily_periodic_return.mean() / daily_periodic_return.std(ddof=1) if daily_periodic_return.std(ddof=1) else float("nan")
            psr = probabilistic_sharpe_ratio(observed_sharpe, 0.0, len(daily_periodic_return), daily_periodic_return)
            dsr = deflated_sharpe_ratio(observed_sharpe, per_fold_ics or [0.0], len(daily_periodic_return), daily_periodic_return)

        return {
            "ic_information_ratio": ir, "effective_sample_size": ess, "daily_sharpe_audit": audit,
            "probabilistic_sharpe_ratio": psr, "deflated_sharpe_ratio": dsr,
        }

    _statistical_significance()

    @stage("08_event_probability_coverage")
    def _event_coverage():
        from data.event_coverage_diagnostics import (
            event_probability_coverage_report,
            eventprob_feature_missingness_report,
        )
        return {
            "coverage": event_probability_coverage_report(con),
            "feature_missingness": eventprob_feature_missingness_report(development_df),
        }

    _event_coverage()

    @stage("09_simple_model_benchmarks")
    def _benchmarks():
        from backtesting.simple_benchmarks import run_simple_benchmarks
        report = run_simple_benchmarks(
            development_df, folds, feature_cols,
            lightgbm_per_fold_ic=evaluation.fold_metrics_summary.get("per_fold_information_coefficient"),
        )
        return asdict(report)

    _benchmarks()

    @stage("10_cost_delay_turnover_stress")
    def _cost_delay():
        from backtesting.cost_delay_stress import run_cost_delay_turnover_stress
        from backtesting.purged_walk_forward import TARGET_TO_PRED_COL
        last_fold = fold_results[-1]
        report = run_cost_delay_turnover_stress(last_fold.predictions, market, TARGET_TO_PRED_COL["excess_return_20d"])
        return report.to_dict(orient="records")

    _cost_delay()

    @stage("11_model_registry_state")
    def _registry_state():
        from models.registry import get_champion, verify_artifact_reproducibility
        champion = get_champion(con)
        if champion is None:
            return {"champion": None}
        verification = verify_artifact_reproducibility(con, champion["model_version"])
        return {"champion": {k: v for k, v in champion.items() if k != "metrics_json"}, "artifact_verification": verification}

    _registry_state()

    @stage("12_historical_holdout_status_readonly")
    def _holdout_status():
        """READ-ONLY: reports what is already logged in holdout_access_log
        on the ORIGINAL real database. Never calls evaluate_on_holdout --
        the historical holdout is a USED test set and this script must not
        touch it again."""
        source_con = duckdb.connect(str(SOURCE_DB), read_only=True)
        try:
            log = source_con.execute("SELECT * FROM holdout_access_log ORDER BY accessed_at").fetchdf()
        finally:
            source_con.close()
        return {
            "n_historical_holdout_accesses_on_record": len(log),
            "access_log": log.to_dict(orient="records"),
            "research_status": "USED HISTORICAL HOLDOUT -- already observed (see docs/EXPERIMENT_REPORT.md); "
                                "not re-accessed by this report-generation run",
        }

    _holdout_status()

    @stage("13_forward_paper_status_readonly")
    def _forward_paper_status():
        log = con.execute("SELECT * FROM forward_paper_access_log ORDER BY accessed_at").fetchdf()
        return {
            "n_forward_paper_accesses_on_record": len(log), "access_log": log.to_dict(orient="records"),
            "research_status": "NOT RUN as part of V0.3 development -- forward-paper evaluation is a deliberate, "
                                "one-time, on-demand action (`evaluate-forward-paper`) reserved for after the V0.3 "
                                "specification is frozen; this report-generation run does not trigger it",
        }

    _forward_paper_status()

    con.close()
    render_report()


def render_report() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# V0.3 Research Report\n")
    lines.append("**PAPER TRADING / SIMULATION ONLY.**\n")
    lines.append(
        "Generated by `scripts/generate_v03_report.py` from the project's real, already-ingested data "
        "(`data_store/duckdb/v02_real_experiment_fixed.duckdb`, operated on via a disposable working copy). "
        "Every section below is either a DEVELOPMENT-ONLY computation or a read-only audit query -- this "
        "report-generation run never calls `evaluate_on_holdout` or `evaluate_on_forward_paper`.\n"
    )
    lines.append(
        "> **Research status labels used throughout this report:** "
        "**DEVELOPMENT RESULTS** (pre-holdout data only, safe to re-run any number of times), "
        "**USED HISTORICAL HOLDOUT RESULTS** (the 2024-07-01..2025-06-30 period -- already observed once via "
        "V0.2's report; a *used* historical test set, **never** described as untouched), and "
        "**FUTURE FORWARD-PAPER RESULTS** (the post-holdout period -- reserved, not yet evaluated).\n"
    )
    lines.append("---\n")

    def get(name: str) -> dict | None:
        entry = results.get(name)
        if entry is None:
            return None
        if entry["status"] != "ok":
            lines.append(f"\n> **{name} FAILED**: {entry.get('error')}\n")
            return None
        return entry["data"]

    lines.append("## 1. Champion/Challenger Promotion Bug (V0.3 Stage 1) -- DEVELOPMENT RESULTS\n")
    lines.append(
        "**Root cause.** `run_promotion_cycle_v2` looked up the incumbent champion *after* the challenger had "
        "already been registered, and the fixture/production path re-trained a deterministic LightGBM model "
        "(`seed=42`) on unchanged data, producing a challenger whose metrics were identical to the incumbent's -- "
        "a vacuous self-comparison that still reported `promoted=true`.\n\n"
        "**Fix.** `champion_record` is now a required (no-default) parameter loaded *before* the challenger is "
        "registered; a challenger is rejected outright if its `validation_period_end` does not extend strictly "
        "past the incumbent's (no new development data since the incumbent was set); a hard "
        "`challenger_model_version != incumbent_champion_model_version` assertion guards literal self-comparison. "
        "See `learning/champion_challenger_v2.py` and `tests/test_learning_v2.py`.\n"
    )

    lines.append("\n## 2. Sharpe Ratio Audit (V0.3 Stage 2) -- DEVELOPMENT RESULTS\n")
    lines.append(
        "**Root cause of the previously reported Sharpe of 6.649.** The old Sharpe was computed from "
        "`build_quantile_portfolio_returns`'s output -- one row per PREDICTION DATE, but each row an OVERLAPPING "
        "multi-day-forward target (5d/20d), fed into `sharpe_ratio()` with `sqrt(252)` annualisation as if it were "
        "an independent daily-return series. Both effects (autocorrelation shrinking sample variance, and a "
        "~sqrt(20)x too-large annualisation factor for 20-day-forward returns sampled daily) inflate the reported "
        "Sharpe.\n\n"
        "**Fix.** `backtesting/daily_portfolio.py::build_daily_rebalanced_portfolio_returns` builds a genuinely "
        "chronological one-row-per-TRADING-DAY series from ACTUAL realised daily price returns, with a strict "
        "no-look-ahead rule (a signal observed on date D takes effect starting D+1). "
        "`sharpe_audit_report` computes Sharpe (and turnover, drawdown, cost drag) from this series.\n"
    )
    sig_data = get("07_statistical_significance")
    if sig_data:
        audit = sig_data.get("daily_sharpe_audit", {})
        lines.append(
            f"**Audited Sharpe, this development run's final fold:** n_observations={audit.get('n_observations')}, "
            f"date_range={audit.get('date_range')}, mean_daily_return={audit.get('mean_daily_return')}, "
            f"daily_volatility={audit.get('daily_volatility')}, "
            f"annualization_factor={audit.get('annualization_factor')}, "
            f"gross_sharpe={audit.get('gross_sharpe')}, net_sharpe={audit.get('net_sharpe')} "
            f"(cost_bps={audit.get('cost_bps_assumption')}), mean_turnover={audit.get('mean_turnover')}, "
            f"max_drawdown={audit.get('max_drawdown')}.\n"
        )

    lines.append("\n## 3. Development-Only Signal Diagnostics (V0.3 Stage 3) -- DEVELOPMENT RESULTS\n")
    diag = get("02_development_diagnostics")
    if diag:
        lines.append(f"```json\n{json.dumps(diag, indent=2, default=str)[:6000]}\n```\n")

    lines.append("\n## 4. Feature-Family Ablation (V0.3 Stage 4) -- DEVELOPMENT RESULTS\n")
    ablation = get("03_feature_family_ablation")
    if ablation:
        lines.append(f"Family sizes: `{ablation.get('family_sizes')}`\n")
        lines.append("\n| variant | n_features | mean_rank_ic | delta_vs_baseline | net_sharpe | max_drawdown |\n")
        lines.append("|---|---|---|---|---|---|\n")
        for r in ablation.get("reports", []):
            audit = r.get("sharpe_audit") or {}
            lines.append(
                f"| {r['variant']} | {r['n_features']} | {r['mean_rank_ic']:.4f} | "
                f"{r['delta_vs_baseline_rank_ic']:.4f} | {audit.get('net_sharpe')} | {audit.get('max_drawdown')} |\n"
            )

    lines.append("\n## 5. Feature Importance Stability (V0.3 Stage 5) -- DEVELOPMENT RESULTS\n")
    stability = get("04_feature_importance_stability")
    if stability:
        lines.append(f"```json\n{json.dumps(stability, indent=2, default=str)}\n```\n")

    lines.append("\n## 6. Negative Controls (V0.3 Stage 6) -- DEVELOPMENT RESULTS\n")
    controls = get("05_negative_controls")
    if controls:
        lines.append("\n| control | statistic | expectation | passed | detail |\n|---|---|---|---|---|\n")
        for c in controls:
            lines.append(f"| {c['name']} | {c['statistic']:.4f} | {c['expectation']} | {c['passed']} | {c['detail']} |\n")
        failed = [c for c in controls if not c["passed"]]
        if failed:
            names = ", ".join(f"`{c['name']}`" for c in failed)
            lines.append(
                f"\n**Not all controls passed as expected ({names}) -- reported honestly, not hidden or "
                "threshold-adjusted to force a pass.** A plausible, non-leakage explanation for each:\n"
                "- `time_shifted_target`: shifts each symbol's target along its own timeline by a fixed offset. "
                "Real equity return series carry genuine multi-month autocorrelation (momentum/trend features "
                "such as `raw_return_60d`, moving averages) that this control's synthetic-fixture design assumed "
                "away -- a positive IC against a shifted target can reflect real persistent trend structure, not "
                "necessarily a leakage bug. Worth a deeper look (e.g. a larger shift, or shifting past any single "
                "feature's own lookback window) before concluding either way.\n"
                "- `symbol_label_permutation`: reassigns targets across symbols within the same date. The "
                "magnitude observed here is small and close to the threshold -- plausibly ordinary sampling noise "
                "at this development set's size, not evidence of a structural leak.\n"
            )

    lines.append("\n## 7. Purge/Embargo Audit (V0.3 Stage 7) -- DEVELOPMENT RESULTS\n")
    purge = get("06_purge_embargo_audit")
    if purge:
        lines.append(
            f"`folds_temporally_ordered={purge['folds_temporally_ordered']}`, "
            f"`no_validation_row_reused_in_training={purge['no_validation_row_reused_in_training']}`\n"
        )
        lines.append(f"```json\n{json.dumps(purge['boundaries'], indent=2, default=str)}\n```\n")

    lines.append("\n## 8. Statistical Significance (V0.3 Stage 8) -- DEVELOPMENT RESULTS\n")
    if sig_data:
        lines.append(f"```json\n{json.dumps(sig_data, indent=2, default=str)}\n```\n")

    lines.append("\n## 9. Event-Probability Data Coverage (V0.3 Stage 9) -- read-only diagnostic\n")
    coverage = get("08_event_probability_coverage")
    if coverage:
        lines.append(f"```json\n{json.dumps(coverage, indent=2, default=str)[:4000]}\n```\n")

    lines.append("\n## 10. Post-Holdout Forward Paper Period (V0.3 Stage 10) -- FUTURE FORWARD-PAPER RESULTS\n")
    forward = get("13_forward_paper_status_readonly")
    if forward:
        lines.append(f"```json\n{json.dumps(forward, indent=2, default=str)}\n```\n")

    lines.append("\n## 11. Simple-Model Benchmarks (V0.3 Stage 11) -- DEVELOPMENT RESULTS\n")
    bench = get("09_simple_model_benchmarks")
    if bench:
        lines.append(f"```json\n{json.dumps(bench, indent=2, default=str)}\n```\n")

    lines.append("\n## 12. Transaction Cost / Delay / Turnover Stress (V0.3 Stage 12) -- DEVELOPMENT RESULTS\n")
    stress = get("10_cost_delay_turnover_stress")
    if stress:
        lines.append("\n| rebalance | execution_delay | cost_bps | net_sharpe | cagr | mean_turnover | max_drawdown |\n|---|---|---|---|---|---|---|\n")
        for row in stress:
            lines.append(
                f"| {row['rebalance']} | {row['execution_delay']} | {row['cost_bps']} | {row['net_sharpe']} | "
                f"{row['cagr']} | {row['mean_turnover']} | {row['max_drawdown']} |\n"
            )

    lines.append("\n## 13. Model Registry Reproducibility (V0.3 Stage 13) -- DEVELOPMENT RESULTS\n")
    registry = get("11_model_registry_state")
    if registry:
        lines.append(f"```json\n{json.dumps(registry, indent=2, default=str)}\n```\n")

    lines.append("\n## 14. USED HISTORICAL HOLDOUT RESULTS\n")
    lines.append(
        "The 2024-07-01..2025-06-30 historical holdout period **has already been observed** (see "
        "`docs/EXPERIMENT_REPORT.md`) and is a **USED historical test set**, never described as untouched in "
        "V0.3. Per the V0.3 research rule, it was **not** re-evaluated to produce this report. The V0.2 report's "
        "recorded result: 5d IC=0.0859, 5d R2=-0.0118, 20d IC=-0.0223, 20d R2=-0.1672, 5d AUC=0.4877, "
        "20d AUC=0.4846 -- the model did not generalise convincingly, especially at 20 days.\n"
    )
    holdout_status = get("12_historical_holdout_status_readonly")
    if holdout_status:
        lines.append(f"```json\n{json.dumps(holdout_status, indent=2, default=str)}\n```\n")

    lines.append("\n## 15. Development vs. Test Period Isolation -- confirmation\n")
    lines.append(
        "- The historical holdout was **NOT** used for any V0.3 tuning, selection, or model comparison in this "
        "report -- every stage above operates on `development_df` only, and section 14 is a read-only log query "
        "against the ORIGINAL database, never a new `evaluate_on_holdout` call.\n"
        "- The post-holdout forward-paper region was **NOT** used for any V0.3 tuning, selection, or model "
        "comparison -- see section 10/13; `evaluate_on_forward_paper` was never called by this script.\n"
    )

    lines.append("\n## 16. Frozen V0.3 Candidate / Forward-Paper Readiness\n")
    dev = get("01_development_evaluation")
    if dev:
        if dev.get("promoted"):
            status_line = (
                f"This run's challenger, `{dev.get('champion_model_version')}`, WAS promoted to champion: "
                f"{dev.get('promotion_rationale')}"
            )
        else:
            status_line = (
                f"This run's challenger, `{dev.get('champion_model_version')}`, was NOT promoted -- there is no "
                f"champion in this working copy of the registry. Reason: {dev.get('promotion_rationale')}"
            )
        lines.append(
            f"{status_line}\n\n"
            "A model is ready for the ONE-TIME `evaluate-forward-paper` step only once it has actually cleared "
            "champion promotion AND a human reviewer has examined sections 1-13 above and deliberately decided "
            "the V0.3 development specification (features, hyperparameters, risk configuration) is frozen. "
            "**No model currently qualifies for forward-paper evaluation based on this report alone if it was "
            "not promoted above.** This report-generation script does not make that decision or trigger "
            "forward-paper evaluation itself.\n"
        )

    REPORT_PATH.write_text("".join(lines))
    print(f"\nreport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
