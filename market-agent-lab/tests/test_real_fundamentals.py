"""Tests for point-in-time fundamental-observation construction from raw
SEC XBRL facts. Uses synthetic in-memory fact tables (no network) that
reproduce the exact ambiguities real SEC data contains: a 10-Q reporting
both a discrete quarter and a cumulative year-to-date figure under the
same tag, and a 10-K reporting several years of comparative figures under
the same tag.
"""

from __future__ import annotations

import pandas as pd
import pytest

from data.real_fundamentals import add_valuation_ratios, build_fundamental_observations_from_facts


def _fact(tag, value, period_start, period_end, filed_date, form_type="10-Q", unit="USD"):
    return {
        "symbol": "TEST", "cik": "0000000001", "tag": tag, "unit": unit,
        "period_start": pd.Timestamp(period_start) if period_start else pd.NaT,
        "period_end": pd.Timestamp(period_end), "value": value,
        "accession_number": "acc-1", "form_type": form_type, "fiscal_year": 2023,
        "fiscal_period": "Q1", "filed_date": pd.Timestamp(filed_date), "source": "sec_edgar",
        "retrieved_at": pd.Timestamp("2024-01-01"),
    }


def test_prefers_discrete_quarter_over_cumulative_figure_same_filing():
    """A Q2 10-Q often reports BOTH the discrete Q2 revenue (~91 days) and
    the H1 year-to-date cumulative revenue (~182 days) under the identical
    tag. The discrete quarter must be selected, not the larger cumulative
    number."""
    facts = pd.DataFrame(
        [
            _fact("revenue", 100.0, "2023-04-01", "2023-06-30", "2023-07-25"),  # discrete Q2, ~91d
            _fact("revenue", 210.0, "2023-01-01", "2023-06-30", "2023-07-25"),  # cumulative H1, ~181d
            _fact("eps_diluted", 1.0, "2023-04-01", "2023-06-30", "2023-07-25"),
        ]
    )
    obs = build_fundamental_observations_from_facts("TEST", facts)
    assert len(obs) == 1
    assert obs[0].revenue == pytest.approx(100.0)  # discrete quarter, not the 210 cumulative


def test_prefers_most_recent_year_among_10k_comparative_figures():
    """A 10-K typically reports the current AND 1-2 prior fiscal years'
    revenue under the identical tag, all ~365 days -- the most recent
    period_end must win, never an older comparative year."""
    facts = pd.DataFrame(
        [
            _fact("revenue", 1000.0, "2021-01-01", "2021-12-31", "2023-02-01", form_type="10-K"),  # FY2021 comparative
            _fact("revenue", 1100.0, "2022-01-01", "2022-12-31", "2023-02-01", form_type="10-K"),  # FY2022 comparative
            _fact("revenue", 1250.0, "2023-01-01", "2023-12-31", "2023-02-01", form_type="10-K"),  # FY2023 current
            _fact("eps_diluted", 5.0, "2023-01-01", "2023-12-31", "2023-02-01", form_type="10-K"),
        ]
    )
    obs = build_fundamental_observations_from_facts("TEST", facts)
    assert len(obs) == 1
    assert obs[0].revenue == pytest.approx(1250.0)
    assert obs[0].reporting_period_end == pd.Timestamp("2023-12-31")


def test_revenue_growth_matches_prior_year_same_period_length():
    facts = pd.DataFrame(
        [
            _fact("revenue", 100.0, "2022-01-01", "2022-03-31", "2022-04-20"),  # Q1 2022
            _fact("eps_diluted", 1.0, "2022-01-01", "2022-03-31", "2022-04-20"),
            _fact("revenue", 120.0, "2023-01-01", "2023-03-31", "2023-04-20"),  # Q1 2023
            _fact("eps_diluted", 1.2, "2023-01-01", "2023-03-31", "2023-04-20"),
        ]
    )
    obs = build_fundamental_observations_from_facts("TEST", facts)
    q1_2023 = [o for o in obs if o.reporting_period_end == pd.Timestamp("2023-03-31")][0]
    assert q1_2023.revenue_growth == pytest.approx((120.0 - 100.0) / 100.0)
    assert q1_2023.eps_growth == pytest.approx((1.2 - 1.0) / 1.0)


def test_skips_snapshot_missing_required_fields():
    facts = pd.DataFrame(
        [
            _fact("gross_profit", 50.0, "2023-01-01", "2023-03-31", "2023-04-20"),
            # no revenue, no eps_diluted -- required fields missing entirely
        ]
    )
    obs = build_fundamental_observations_from_facts("TEST", facts)
    assert obs == []


def test_publication_timestamp_is_filed_date_not_period_end():
    facts = pd.DataFrame(
        [
            _fact("revenue", 100.0, "2023-01-01", "2023-03-31", "2023-04-20"),
            _fact("eps_diluted", 1.0, "2023-01-01", "2023-03-31", "2023-04-20"),
        ]
    )
    obs = build_fundamental_observations_from_facts("TEST", facts)
    assert obs[0].publication_timestamp == pd.Timestamp("2023-04-20")
    assert obs[0].reporting_period_end == pd.Timestamp("2023-03-31")
    assert obs[0].publication_timestamp != obs[0].reporting_period_end


def test_gross_and_operating_margin_computed_correctly():
    facts = pd.DataFrame(
        [
            _fact("revenue", 200.0, "2023-01-01", "2023-03-31", "2023-04-20"),
            _fact("eps_diluted", 1.0, "2023-01-01", "2023-03-31", "2023-04-20"),
            _fact("gross_profit", 80.0, "2023-01-01", "2023-03-31", "2023-04-20"),
            _fact("operating_income", 40.0, "2023-01-01", "2023-03-31", "2023-04-20"),
        ]
    )
    obs = build_fundamental_observations_from_facts("TEST", facts)
    assert obs[0].gross_margin == pytest.approx(0.4)
    assert obs[0].operating_margin == pytest.approx(0.2)


def test_instant_concepts_use_latest_period_end_without_duration_filtering():
    """Balance-sheet ('instant') concepts have no period_start/duration
    ambiguity -- Assets etc. should just take the latest period_end."""
    facts = pd.DataFrame(
        [
            _fact("revenue", 100.0, "2023-01-01", "2023-03-31", "2023-04-20"),
            _fact("eps_diluted", 1.0, "2023-01-01", "2023-03-31", "2023-04-20"),
            _fact("cash", 500.0, None, "2022-12-31", "2023-04-20"),
            _fact("cash", 600.0, None, "2023-03-31", "2023-04-20"),
        ]
    )
    obs = build_fundamental_observations_from_facts("TEST", facts)
    assert obs[0].cash == pytest.approx(600.0)


def test_add_valuation_ratios_computes_pe_ps_pb():
    facts = pd.DataFrame(
        [
            _fact("revenue", 1000.0, "2023-01-01", "2023-03-31", "2023-04-20"),
            _fact("eps_diluted", 2.0, "2023-01-01", "2023-03-31", "2023-04-20"),
            _fact("stockholders_equity", 5000.0, None, "2023-03-31", "2023-04-20"),
            _fact("shares_outstanding", 100.0, None, "2023-03-31", "2023-04-20"),
        ]
    )
    obs = build_fundamental_observations_from_facts("TEST", facts)
    priced = add_valuation_ratios(obs, price_lookup={pd.Timestamp("2023-04-20"): 50.0})
    assert priced[0].pe_ratio == pytest.approx(25.0)  # 50 / 2.0
    assert priced[0].price_to_sales == pytest.approx(5.0)  # (50*100) / 1000
    assert priced[0].price_to_book == pytest.approx(1.0)  # (50*100) / 5000
