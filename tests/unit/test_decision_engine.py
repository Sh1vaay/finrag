"""Unit tests for Quantitative Decision Models and Decision Engine node.

Covers:
1. Piotroski F-Score 9-signal fundamental momentum & accounting quality.
2. Altman Z-Score statistical insolvency & bankruptcy risk zones.
3. Portfolio trade rebalancing & mandate constraint enforcement.
4. Composite Factor Conviction (Quality, Solvency, Value).
5. Graph node execution: decision_engine_node (RBAC, formatting, state injection).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure root directory is on pythonpath
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.tools.decision_models import (
    DecisionModelError,
    calculate_altman_z_score,
    calculate_factor_exposures,
    calculate_piotroski_f_score,
    calculate_portfolio_var_cvar,
    evaluate_composite_investment_conviction,
    evaluate_portfolio_trade_decision,
    optimize_portfolio_weights,
)
from src.graph.nodes.decision_engine import decision_engine_node


def test_strong_piotroski_profile():
    inputs = {
        "net_income_current": 100.0,
        "net_income_prior": 80.0,
        "operating_cash_flow": 120.0,
        "total_assets_current": 1000.0,
        "total_assets_prior": 950.0,
        "long_term_debt_current": 200.0,
        "long_term_debt_prior": 220.0,
        "current_assets_current": 500.0,
        "current_liabilities_current": 250.0,
        "current_assets_prior": 400.0,
        "current_liabilities_prior": 250.0,
        "shares_outstanding_current": 10.0,
        "shares_outstanding_prior": 10.0,
        "gross_profit_current": 400.0,
        "revenue_current": 800.0,
        "gross_profit_prior": 300.0,
        "revenue_prior": 700.0,
    }
    res = calculate_piotroski_f_score(inputs)
    assert res["piotroski_f_score"] == 9
    assert res["financial_health"] == "STRONG"
    assert res["signals"]["positive_net_income"] is True
    assert res["signals"]["cash_flow_greater_than_net_income"] is True
    assert res["signals"]["deleveraging_or_stable_debt"] is True


def test_weak_piotroski_profile():
    inputs = {
        "net_income_current": -50.0,
        "net_income_prior": 20.0,
        "operating_cash_flow": -10.0,
        "total_assets_current": 1000.0,
        "total_assets_prior": 1000.0,
        "long_term_debt_current": 600.0,
        "long_term_debt_prior": 400.0,
        "current_assets_current": 100.0,
        "current_liabilities_current": 300.0,
        "current_assets_prior": 200.0,
        "current_liabilities_prior": 200.0,
        "shares_outstanding_current": 15.0,
        "shares_outstanding_prior": 10.0,
        "gross_profit_current": 50.0,
        "revenue_current": 500.0,
        "gross_profit_prior": 100.0,
        "revenue_prior": 500.0,
    }
    res = calculate_piotroski_f_score(inputs)
    assert res["piotroski_f_score"] <= 3
    assert res["financial_health"] == "WEAK"


def test_invalid_assets_raises_error():
    try:
        calculate_piotroski_f_score({"total_assets_current": 0.0})
        assert False, "Should have raised DecisionModelError"
    except DecisionModelError:
        pass


def test_safe_zone_corporate():
    inputs = {
        "working_capital": 35.0,
        "retained_earnings": 150.0,
        "ebit": 100.0,
        "market_equity": 3000.0,
        "total_liabilities": 200.0,
        "sales": 250.0,
        "total_assets": 500.0,
    }
    res = calculate_altman_z_score(inputs)
    assert res["altman_z_score"] > 2.99
    assert res["solvency_zone"] == "SAFE_ZONE"
    assert "Low Distress Risk" in res["solvency_rating"]


def test_distress_zone_corporate():
    inputs = {
        "working_capital": -80.0,
        "retained_earnings": -120.0,
        "ebit": -30.0,
        "market_equity": 20.0,
        "total_liabilities": 500.0,
        "sales": 100.0,
        "total_assets": 400.0,
    }
    res = calculate_altman_z_score(inputs)
    assert res["altman_z_score"] < 1.81
    assert res["solvency_zone"] == "DISTRESS_ZONE"
    assert "High Distress Risk" in res["solvency_rating"]


def test_approved_rebalancing_trade():
    holdings = {
        "AAPL": 14.5, "MSFT": 12.0, "NVDA": 9.5, "AMZN": 7.0, "GOOGL": 6.5,
        "META": 5.0, "TSLA": 4.5, "BRK.B": 4.0, "JPM": 3.5, "V": 3.0,
        "UNH": 3.0, "XOM": 2.5, "MA": 2.5, "PG": 2.5, "JNJ": 2.5,
        "HD": 2.0, "ABBV": 2.0, "MRK": 2.0, "COST": 2.0, "AVGO": 1.5,
    }
    res = evaluate_portfolio_trade_decision(
        holdings=holdings,
        trade_ticker="MSFT",
        trade_weight_delta_pct=1.5,  # 12.0 -> 13.5% (well within 15% cap)
        max_single_position_pct=15.0,
        max_top5_concentration_pct=60.0,
        max_hhi_threshold=2500.0,
    )
    assert res["decision"] == "APPROVED"
    assert len(res["violations"]) == 0
    assert res["post_trade_weight_pct"] <= 15.0


def test_rejected_single_name_cap_breach():
    holdings = {"AAPL": 14.5, "MSFT": 12.0, "NVDA": 9.5}
    res = evaluate_portfolio_trade_decision(
        holdings=holdings,
        trade_ticker="AAPL",
        trade_weight_delta_pct=10.0,  # Pushes AAPL well beyond 15%
        max_single_position_pct=15.0,
    )
    assert res["decision"] == "REJECTED_MANDATE_BREACH"
    assert any("Single-name limit breach" in v for v in res["violations"])


def test_strong_conviction():
    res = evaluate_composite_investment_conviction(
        piotroski_score=8,
        altman_zone="SAFE_ZONE",
        valuation_multiples={"fcf_yield_pct": 5.0, "pe_ratio": 22.0},
        profitability_metrics={"return_on_equity_pct": 25.0},
    )
    assert res["composite_quant_score"] >= 70.0
    assert res["rating"] in ("STRONG_OVERWEIGHT", "OVERWEIGHT")


def test_distress_override():
    res = evaluate_composite_investment_conviction(
        piotroski_score=7,
        altman_zone="DISTRESS_ZONE",
        valuation_multiples={"fcf_yield_pct": 8.0},
    )
    assert res["rating"] == "RESTRICTED / AVOID"


def test_equity_decision_evaluation():
    state = {
        "sanitized_query": "Evaluate Piotroski F-Score and Altman Z bankruptcy risk for Apple",
        "target_company": "aapl",
        "user_role": "research_analyst",
        "query_intent": "market_data",
    }
    out = decision_engine_node(state)
    qa = out.get("quantitative_analysis")
    assert qa is not None
    assert "piotroski_f_score" in qa
    assert "altman_z_score" in qa
    assert "composite_conviction" in qa
    assert "formatted_summary" in qa
    assert "Piotroski F-Score" in qa["formatted_summary"]
    assert "Altman Z-Score" in qa["formatted_summary"]


def test_rebalancing_analyst_denial():
    state = {
        "sanitized_query": "Add 2% to AAPL and rebalance portfolio",
        "user_role": "research_analyst",  # Not allowed to simulate portfolio trades
    }
    out = decision_engine_node(state)
    qa = out.get("quantitative_analysis")
    assert qa is not None
    assert qa["portfolio_trade_decision"]["decision"] == "PERMISSION_DENIED"


def test_rebalancing_pm_approval():
    state = {
        "sanitized_query": "Add 1.5% to MSFT in fund portfolio",
        "user_role": "portfolio_manager",
    }
    out = decision_engine_node(state)
    qa = out.get("quantitative_analysis")
    assert qa is not None
    assert qa["portfolio_trade_decision"]["decision"] in ("APPROVED", "REJECTED_MANDATE_BREACH")


def test_non_financial_query_noop():
    state = {
        "sanitized_query": "Hello, how are you today?",
        "target_company": None,
        "user_role": "research_analyst",
    }
    out = decision_engine_node(state)
    assert out == {"quantitative_analysis": None}


def test_portfolio_var_cvar():
    holdings = {"AAPL": 40.0, "MSFT": 30.0, "NVDA": 30.0}
    res95 = calculate_portfolio_var_cvar(holdings, confidence_level=0.95, portfolio_nav_usd=10_000_000.0)
    assert res95["confidence_level"] == "95%"
    assert res95["parametric_var_pct"] > 0
    assert res95["conditional_var_expected_shortfall_pct"] > res95["parametric_var_pct"]
    assert res95["conditional_var_expected_shortfall_usd"] > res95["parametric_var_usd"]

    res99 = calculate_portfolio_var_cvar(holdings, confidence_level=0.99, portfolio_nav_usd=10_000_000.0)
    assert res99["confidence_level"] == "99%"
    assert res99["parametric_var_pct"] > res95["parametric_var_pct"]


def test_constrained_portfolio_optimization():
    holdings = {"AAPL": 35.0, "MSFT": 25.0, "NVDA": 25.0, "JPM": 15.0}
    opt = optimize_portfolio_weights(holdings, max_single_cap=0.30, shrinkage_intensity=0.20)
    assert opt["optimized_annual_volatility_pct"] <= opt["baseline_annual_volatility_pct"]
    for ticker, weight in opt["optimized_weights_pct"].items():
        assert weight <= 30.01
    assert round(sum(opt["optimized_weights_pct"].values())) == 100


def test_factor_exposures():
    aapl_factors = calculate_factor_exposures("AAPL")
    assert aapl_factors["ticker"] == "AAPL"
    assert aapl_factors["market_beta"] > 1.0
    assert "Large Cap" in aapl_factors["style_classification"]
    assert aapl_factors["systematic_risk_pct"] > 0
    assert aapl_factors["idiosyncratic_risk_pct"] > 0


def test_decision_engine_var_query():
    state = {
        "sanitized_query": "What is the portfolio VaR and expected shortfall under 95% confidence?",
        "user_role": "portfolio_manager",
    }
    out = decision_engine_node(state)
    qa = out.get("quantitative_analysis")
    assert qa is not None
    assert "portfolio_var_cvar" in qa
    assert qa["portfolio_var_cvar"]["confidence_level"] == "95%"
    assert "Parametric VaR" in qa["formatted_summary"]


if __name__ == "__main__":
    test_strong_piotroski_profile()
    test_weak_piotroski_profile()
    test_invalid_assets_raises_error()
    test_safe_zone_corporate()
    test_distress_zone_corporate()
    test_approved_rebalancing_trade()
    test_rejected_single_name_cap_breach()
    test_strong_conviction()
    test_distress_override()
    test_equity_decision_evaluation()
    test_rebalancing_analyst_denial()
    test_rebalancing_pm_approval()
    test_non_financial_query_noop()
    test_portfolio_var_cvar()
    test_constrained_portfolio_optimization()
    test_factor_exposures()
    test_decision_engine_var_query()
    print("All quantitative decision models and decision engine node tests passed successfully!")
