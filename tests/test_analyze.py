import json
from pathlib import Path

import pytest

from cashflow import analyze, evaluate_rules


DATA_DIR = Path(__file__).resolve().parents[1] / "samples"
RULES = Path(__file__).resolve().parents[1] / "rules.yaml"


def test_analyze_healthy():
    m = analyze(str(DATA_DIR / "healthy.csv"))
    assert isinstance(m, dict)
    assert m["avg_daily_balance"] > 3000
    assert m["nsf_count"] == 0
    assert m["inflow_outflow_ratio"] > 1.0
    assert any(k.startswith("2025-03") for k in m["monthly_gross_revenue"])
    assert any(k.startswith("2025-04") for k in m["monthly_gross_revenue"])


def test_analyze_risky_with_rules():
    m = analyze(str(DATA_DIR / "risky.csv"))
    assert m["nsf_count"] >= 10
    assert m["nsf_count_90d"] >= 10
    flags = evaluate_rules(m, str(RULES))
    assert any(fr.passed for fr in flags)


@pytest.mark.parametrize("csv_name", ["healthy.csv", "risky.csv"])
def test_kpis_shape(csv_name):
    m = analyze(str(DATA_DIR / csv_name))
    expected_keys = {
        "as_of",
        "generated_at",
        "period",
        "days_covered",
        "monthly_gross_revenue",
        "avg_daily_balance",
        "nsf_count",
        "nsf_count_90d",
        "inflow_outflow_ratio",
        "volatility_std",
        "rolling_90_day_trend",
        "seasonality_hint",
    }
    assert expected_keys.issubset(m.keys())
