"""Real fundamentals orchestrator (Phase 5): SEC EDGAR ingestion, plus
building point-in-time :class:`~core.schemas.FundamentalObservation` rows
so the *existing* V0.1 fundamental feature pipeline
(``features/fundamental.py``, ``database.repository.get_fundamentals_asof``)
works unchanged on real data.

Every derived observation is keyed by ``publication_timestamp = filed_date``
-- never ``period_end`` -- which is exactly the leakage guard V0.1's
schema was already designed around.

Documented approximation: valuation ratios use the period's own EPS/revenue
rather than a trailing-twelve-month figure (TTM reconstruction needs
quarter-stitching that's out of scope for v0.2); see
``docs/data_sources.md``. ROIC and EV/EBITDA use flat/period-based proxies
for the same reason.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from core.schemas import FundamentalObservation
from core.schemas_v2 import IngestionStatus, ProviderCategory
from data.providers.base import finish_ingestion_run, make_ingestion_run
from data.providers.fundamentals.sec import SecEdgarFundamentalProvider
from database import repository as repo
from database import repository_v2 as repo_v2

STATUTORY_TAX_RATE = 0.21  # US federal corporate rate; a documented approximation for ROIC

# SEC XBRL commonly reports BOTH a discrete-quarter figure and a
# cumulative (e.g. 6-month, 9-month) figure under the identical tag within
# the same 10-Q -- picking "whichever has the latest period_end" silently
# grabs the cumulative figure and corrupts growth/margin math. These are
# duration ("flow") concepts where that ambiguity applies; instant
# ("stock") concepts like Assets have no period_start and no such
# ambiguity, so they're excluded from the duration-matching logic below.
DURATION_TAGS = {"revenue", "gross_profit", "operating_income", "net_income", "eps_diluted", "operating_cash_flow", "capex"}
QUARTER_TARGET_DAYS = 91
ANNUAL_TARGET_DAYS = 365


def _pick_by_duration(candidates: pd.DataFrame, target_days: int, tolerance_days: int = 20) -> pd.Series:
    """Among same-tag candidates (already narrowed to one filed_date),
    first narrow to those whose period_start->period_end span is within
    ``tolerance_days`` of ``target_days`` -- i.e. exclude a cumulative
    6-/9-month figure filed alongside a discrete quarter. A 10-K also
    typically reports 2-3 years of ~365-day COMPARATIVE figures under the
    identical tag, all of which pass the annual-duration filter together;
    among whatever survives that filter, the most recent ``period_end`` is
    unambiguously the current period, never a prior-year comparative."""
    with_duration = candidates.dropna(subset=["period_start"]).copy()
    if with_duration.empty:
        return candidates.sort_values("period_end").iloc[-1]
    with_duration["_duration_days"] = (with_duration["period_end"] - with_duration["period_start"]).dt.days
    within_tolerance = with_duration[(with_duration["_duration_days"] - target_days).abs() <= tolerance_days]
    pool = within_tolerance if not within_tolerance.empty else with_duration
    return pool.sort_values("period_end").iloc[-1]


def _latest_value_asof(asof: pd.DataFrame, tag: str, target_days: int) -> float | None:
    tag_rows = asof[asof["tag"] == tag]
    if tag_rows.empty:
        return None
    latest_filed = tag_rows["filed_date"].max()
    same_filing = tag_rows[tag_rows["filed_date"] == latest_filed]
    if tag in DURATION_TAGS and len(same_filing) > 1:
        row = _pick_by_duration(same_filing, target_days)
    else:
        row = same_filing.sort_values("period_end").iloc[-1]
    return float(row["value"])


def _find_prior_year_value(symbol_facts: pd.DataFrame, tag: str, period_end: pd.Timestamp, as_of_filed: pd.Timestamp, target_days: int) -> float | None:
    candidates = symbol_facts[
        (symbol_facts["tag"] == tag)
        & (symbol_facts["filed_date"] <= as_of_filed)
        & (symbol_facts["period_end"] >= period_end - pd.Timedelta(days=400))
        & (symbol_facts["period_end"] <= period_end - pd.Timedelta(days=330))
    ]
    if candidates.empty:
        return None
    if tag in DURATION_TAGS and len(candidates) > 1:
        row = _pick_by_duration(candidates, target_days)
        return float(row["value"])
    candidates = candidates.assign(_dist=(candidates["period_end"] - (period_end - pd.Timedelta(days=365))).abs())
    return float(candidates.sort_values("_dist").iloc[0]["value"])


def build_fundamental_observations_from_facts(symbol: str, facts_df: pd.DataFrame) -> list[FundamentalObservation]:
    if facts_df.empty:
        return []
    periodic = facts_df[facts_df["form_type"].isin({"10-K", "10-Q"})]
    filed_dates = sorted(periodic["filed_date"].unique())

    observations: list[FundamentalObservation] = []
    for fd in filed_dates:
        asof = facts_df[facts_df["filed_date"] <= fd]
        triggering_form = periodic[periodic["filed_date"] == fd]["form_type"].iloc[0]
        target_days = ANNUAL_TARGET_DAYS if triggering_form.startswith("10-K") else QUARTER_TARGET_DAYS

        def val(tag: str, _asof=asof, _target_days=target_days) -> float | None:
            return _latest_value_asof(_asof, tag, _target_days)

        revenue = val("revenue")
        eps = val("eps_diluted")
        if revenue is None or eps is None:
            continue  # required fields missing for this snapshot -- skip, never fabricate

        period_end_rows = facts_df[facts_df["filed_date"] == fd]
        period_end = period_end_rows["period_end"].max()

        gross_profit = val("gross_profit")
        operating_income = val("operating_income")
        operating_cash_flow = val("operating_cash_flow")
        capex = val("capex")
        debt = val("long_term_debt")
        cash = val("cash")
        equity = val("stockholders_equity")
        shares = val("shares_outstanding")

        prior_revenue = _find_prior_year_value(asof, "revenue", period_end, fd, target_days)
        prior_eps = _find_prior_year_value(asof, "eps_diluted", period_end, fd, target_days)

        gross_margin = gross_profit / revenue if gross_profit is not None and revenue else None
        operating_margin = operating_income / revenue if operating_income is not None and revenue else None
        fcf = (operating_cash_flow - capex) if operating_cash_flow is not None and capex is not None else None
        fcf_margin = fcf / revenue if fcf is not None and revenue else None
        roic = (
            operating_income * (1 - STATUTORY_TAX_RATE) / (debt + equity)
            if operating_income is not None and debt is not None and equity is not None and (debt + equity) > 0
            else None
        )

        observations.append(
            FundamentalObservation(
                symbol=symbol, publication_timestamp=fd, reporting_period_end=period_end,
                revenue=revenue, revenue_growth=(revenue - prior_revenue) / abs(prior_revenue) if prior_revenue else None,
                eps=eps, eps_growth=(eps - prior_eps) / abs(prior_eps) if prior_eps else None,
                gross_margin=gross_margin, operating_margin=operating_margin,
                free_cash_flow=fcf, fcf_margin=fcf_margin, roic=roic, debt=debt, cash=cash,
                # Valuation ratios need a point-in-time price join -- filled in by add_valuation_ratios().
                pe_ratio=None, ev_to_ebitda=None, price_to_book=None, price_to_sales=None,
            )
        )
        # Stash the inputs a price join needs (shares/equity/operating income)
        # without changing the frozen FundamentalObservation shape used
        # downstream by V0.1's feature pipeline.
        _VALUATION_INPUTS[(symbol, fd)] = {"shares": shares, "equity": equity, "operating_income": operating_income}
    return observations


_VALUATION_INPUTS: dict[tuple, dict] = {}


def add_valuation_ratios(observations: list[FundamentalObservation], price_lookup: dict[pd.Timestamp, float]) -> list[FundamentalObservation]:
    """``price_lookup``: nearest trading-day close on/after each
    observation's ``publication_timestamp`` (the earliest a market
    participant could react to the filing).

    EV/EBITDA here is an EBIT-based proxy (operating income, not a true
    EBITDA with D&A added back) -- documented in docs/data_sources.md.
    """
    updated = []
    for obs in observations:
        inputs = _VALUATION_INPUTS.get((obs.symbol, obs.publication_timestamp), {})
        shares, equity, operating_income = inputs.get("shares"), inputs.get("equity"), inputs.get("operating_income")
        price = price_lookup.get(obs.publication_timestamp)
        if shares is None or price is None:
            updated.append(obs)
            continue
        market_cap = price * shares
        pe = price / obs.eps if obs.eps else None
        ps = market_cap / obs.revenue if obs.revenue else None
        pb = market_cap / equity if equity else None
        enterprise_value = market_cap + (obs.debt or 0.0) - (obs.cash or 0.0)
        ev_ebitda = enterprise_value / operating_income if operating_income else None
        updated.append(obs.model_copy(update={"pe_ratio": pe, "price_to_sales": ps, "price_to_book": pb, "ev_to_ebitda": ev_ebitda}))
    return updated


def ingest_fundamentals(con: duckdb.DuckDBPyConnection, symbols: list[str]) -> dict:
    provider = SecEdgarFundamentalProvider()
    summary: dict[str, dict] = {}
    all_observations: list[FundamentalObservation] = []

    for symbol in symbols:
        run = make_ingestion_run(provider.source_id, ProviderCategory.FUNDAMENTAL)
        try:
            facts = provider.get_company_fundamentals(symbol)
            filings = provider.get_filings(symbol, form_types={"10-K", "10-Q", "10-K/A", "10-Q/A"})
        except Exception as exc:  # noqa: BLE001
            repo_v2.insert_ingestion_run(con, finish_ingestion_run(run, IngestionStatus.FAILED, error=str(exc)))
            summary[symbol] = {"status": "FAILED", "error": str(exc)}
            continue

        if not facts:
            repo_v2.insert_ingestion_run(con, finish_ingestion_run(run, IngestionStatus.UNAVAILABLE, error="CIK not found or no facts"))
            summary[symbol] = {"status": "UNAVAILABLE", "facts": 0}
            continue

        repo_v2.insert_fundamental_facts(con, facts)
        if filings:
            repo_v2.insert_sec_filings(con, filings)

        facts_df = pd.DataFrame([f.model_dump() for f in facts])
        facts_df["filed_date"] = pd.to_datetime(facts_df["filed_date"])
        facts_df["period_end"] = pd.to_datetime(facts_df["period_end"])
        observations = build_fundamental_observations_from_facts(symbol, facts_df)
        all_observations.extend(observations)

        repo_v2.insert_ingestion_run(con, finish_ingestion_run(run, IngestionStatus.SUCCESS, records=len(facts)))
        summary[symbol] = {"status": "SUCCESS", "facts": len(facts), "filings": len(filings), "observations": len(observations)}

    if all_observations:
        df = pd.DataFrame([o.model_dump() for o in all_observations])
        n = repo.insert_fundamental_observations(con, df)
        summary["_total_observations_written"] = n

    return summary
