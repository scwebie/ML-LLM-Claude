"""V0.3 Stage 9: investigate and report event-probability (Polymarket
read-only) historical coverage.

**Finding.** ``PredictionMarketReadOnlyProvider.get_active_events()``
(``data/providers/events/prediction_market_readonly.py``) calls
Polymarket's public ``/events`` endpoint, which returns only CURRENTLY
ACTIVE markets -- there is no historical-archive endpoint in this
read-only API. Every observation's ``observed_timestamp`` is therefore
set to "now" at fetch time (see that provider's ``get_active_events``),
so a database that has only ever been ingested into once has exactly one
distinct observation day, clustered at whatever moment ingestion ran.

**No backfill.** ``data/real_features.py::_lookup_asof`` (unchanged --
confirmed correct by this audit, not a bug) does a strict as-of lookup:
for a historical row whose timestamp is before the earliest observation
that exists for its event, it returns ``None`` -- the feature comes out
NaN and ``eventprob_missing=1``, never a fabricated/backfilled value.

**Prospective collection already works.** Every
``EventProbabilityObservation`` gets a fresh, unique ``id`` (see
``core/schemas_v2.py``'s ``default_factory=_new_id``), and
``insert_event_probability_observations`` upserts by ``id`` -- so
repeated ingestion runs over time APPEND new, distinctly-timestamped
snapshots rather than overwriting the previous one. No code change was
needed to "support" prospective historical collection for forward-paper
research (V0.3 Stage 9, item 5): it is already the natural behaviour of
running ``ingest-prices``/``build-real-features``/``real-demo`` again on
a later date. This module only adds the coverage reporting.

Read-only throughout: this module makes zero HTTP requests, holds no
credentials, and has no execution/order/wager/wallet capability of any
kind -- it only reads already-ingested rows out of the local database.
"""

from __future__ import annotations

import duckdb
import pandas as pd


def event_probability_coverage_report(con: duckdb.DuckDBPyConnection) -> dict:
    """Earliest/latest observation timestamp, whether only a single
    snapshot exists, and coverage counts by date and category."""
    obs = con.execute(
        "SELECT event_id, category, observed_timestamp FROM event_probability_observations"
    ).fetchdf()
    if obs.empty:
        return {
            "n_observations": 0, "n_distinct_events": 0, "n_distinct_observation_days": 0,
            "earliest_observed_timestamp": None, "latest_observed_timestamp": None,
            "is_single_snapshot_only": None, "coverage_by_category": {}, "coverage_by_date": {},
            "finding": "no event-probability observations have been ingested into this database yet",
        }

    obs["observed_date"] = pd.to_datetime(obs["observed_timestamp"]).dt.date
    n_distinct_days = int(obs["observed_date"].nunique())
    is_single_snapshot = n_distinct_days <= 1
    by_category = obs.groupby("category").size().to_dict()
    by_date = {str(k): int(v) for k, v in obs.groupby("observed_date").size().items()}

    finding = (
        "Only a single current snapshot is available -- Polymarket's read-only get_active_events endpoint "
        "returns only currently-active markets, with no historical archive. Historical rows before this "
        "snapshot's timestamp correctly show eventprob_missing=1/NaN (data/real_features.py::_lookup_asof) "
        "rather than a fabricated backfilled value. Every ingestion run appends a genuinely new, freshly "
        "timestamped snapshot, so repeated runs over time prospectively accumulate a real historical archive "
        "for forward-paper research -- it is not, and cannot be, backdated retroactively."
        if is_single_snapshot else
        f"{n_distinct_days} distinct observation days are present -- a genuine (if still short) prospective "
        "history has begun accumulating from repeated ingestion runs."
    )

    return {
        "n_observations": int(len(obs)),
        "n_distinct_events": int(obs["event_id"].nunique()),
        "n_distinct_observation_days": n_distinct_days,
        "earliest_observed_timestamp": str(obs["observed_timestamp"].min()),
        "latest_observed_timestamp": str(obs["observed_timestamp"].max()),
        "is_single_snapshot_only": is_single_snapshot,
        "coverage_by_category": by_category,
        "coverage_by_date": by_date,
        "finding": finding,
    }


def eventprob_feature_missingness_report(feature_df: pd.DataFrame) -> dict:
    """For each eventprob_* feature column actually present in a built
    feature matrix (development, holdout, or post-holdout), what fraction
    of rows have a real (non-NaN) value -- the ground truth of how much
    of the matrix this family can actually inform, given the coverage
    reported above."""
    eventprob_cols = [c for c in feature_df.columns if c.startswith("eventprob_") and c != "eventprob_missing"]
    if not eventprob_cols or feature_df.empty:
        return {"n_rows": len(feature_df), "columns": {}}
    return {
        "n_rows": int(len(feature_df)),
        "columns": {col: float(feature_df[col].notna().mean()) for col in eventprob_cols},
        "overall_any_present_fraction": (
            float((1.0 - feature_df["eventprob_missing"]).mean()) if "eventprob_missing" in feature_df.columns else float("nan")
        ),
    }
