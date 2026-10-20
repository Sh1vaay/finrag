"""Quantitative Decision Engine node for institutional asset management.

Integrates statistical and fundamental decision models directly into the LangGraph state machine:
1. Piotroski F-Score (0-9 Fundamental Health & Accounting Quality)
2. Altman Z-Score (Statistical Insolvency & Default Risk)
3. Portfolio Rebalancing & Mandate Constraint Enforcement (HHI & Concentration)
4. Composite Factor Conviction (Quality, Value, Solvency)

Executes between retrieval/tool execution and generator, producing a verified
quantitative decision sheet to eliminate LLM speculation on numbers.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from src.models.state import RAGState
from src.services.sec_edgar_service import get_fundamentals
from src.tools.decision_models import (
    calculate_altman_z_score,
    calculate_factor_exposures,
    calculate_piotroski_f_score,
    calculate_portfolio_var_cvar,
    evaluate_composite_investment_conviction,
    evaluate_portfolio_trade_decision,
    optimize_portfolio_weights,
)

logger = logging.getLogger(__name__)

# Path to project root — anchored to this file so it works regardless of CWD.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Canonical corporate financial fundamentals for standard institutional benchmark tickers
# (Derived from SEC 10-K audited financial filings in billions USD)
_BENCHMARK_FUNDAMENTALS: dict[str, dict[str, Any]] = {
    "aapl": {
        "company": "Apple Inc.",
        "ticker": "AAPL",
        "net_income_current": 96.99,
        "net_income_prior": 99.80,
        "operating_cash_flow": 110.54,
        "total_assets_current": 352.58,
        "total_assets_prior": 352.75,
        "long_term_debt_current": 95.28,
        "long_term_debt_prior": 98.96,
        "current_assets_current": 143.57,
        "current_liabilities_current": 145.31,
        "current_assets_prior": 135.41,
        "current_liabilities_prior": 153.98,
        "shares_outstanding_current": 15.55,
        "shares_outstanding_prior": 15.91,
        "gross_profit_current": 169.15,
        "gross_profit_prior": 170.78,
        "revenue_current": 383.29,
        "revenue_prior": 394.33,
        "working_capital": -1.74,
        "retained_earnings": -0.21,
        "ebit": 114.30,
        "market_equity": 3400.0,
        "total_liabilities": 290.44,
        "sales": 383.29,
        "pe_ratio": 34.2,
        "fcf_yield_pct": 3.2,
        "roe_pct": 156.0,
    },
    "msft": {
        "company": "Microsoft Corporation",
        "ticker": "MSFT",
        "net_income_current": 88.14,
        "net_income_prior": 72.36,
        "operating_cash_flow": 118.55,
        "total_assets_current": 512.16,
        "total_assets_prior": 411.98,
        "long_term_debt_current": 42.66,
        "long_term_debt_prior": 41.98,
        "current_assets_current": 159.72,
        "current_liabilities_current": 124.62,
        "current_assets_prior": 184.26,
        "current_liabilities_prior": 104.15,
        "shares_outstanding_current": 7.43,
        "shares_outstanding_prior": 7.43,
        "gross_profit_current": 170.82,
        "gross_profit_prior": 146.05,
        "revenue_current": 245.12,
        "revenue_prior": 211.92,
        "working_capital": 35.10,
        "retained_earnings": 152.96,
        "ebit": 109.43,
        "market_equity": 3150.0,
        "total_liabilities": 243.69,
        "sales": 245.12,
        "pe_ratio": 35.5,
        "fcf_yield_pct": 2.8,
        "roe_pct": 38.5,
    },
    "nvda": {
        "company": "NVIDIA Corporation",
        "ticker": "NVDA",
        "net_income_current": 29.76,
        "net_income_prior": 4.37,
        "operating_cash_flow": 28.09,
        "total_assets_current": 65.73,
        "total_assets_prior": 41.18,
        "long_term_debt_current": 8.46,
        "long_term_debt_prior": 9.70,
        "current_assets_current": 44.35,
        "current_liabilities_current": 10.63,
        "current_assets_prior": 23.07,
        "current_liabilities_prior": 6.56,
        "shares_outstanding_current": 24.6,
        "shares_outstanding_prior": 24.7,
        "gross_profit_current": 44.30,
        "gross_profit_prior": 15.36,
        "revenue_current": 60.92,
        "revenue_prior": 26.97,
        "working_capital": 33.72,
        "retained_earnings": 33.84,
        "ebit": 32.97,
        "market_equity": 3200.0,
        "total_liabilities": 22.75,
        "sales": 60.92,
        "pe_ratio": 48.0,
        "fcf_yield_pct": 2.1,
        "roe_pct": 91.0,
    },
}


def _load_benchmark_fundamentals() -> dict[str, dict[str, Any]]:
    """Load benchmark fundamentals from data/benchmark_fundamentals.json with fallback."""
    fund_path = _PROJECT_ROOT / "data" / "benchmark_fundamentals.json"
    if fund_path.exists():
        try:
            with open(fund_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                funds = data.get("fundamentals")
                if isinstance(funds, dict) and funds:
                    return funds
        except Exception as e:
            logger.warning(f"Failed to read benchmark_fundamentals.json: {e}")
    return _BENCHMARK_FUNDAMENTALS


def _load_sample_portfolio() -> dict[str, float]:
    """Load institutional portfolio holdings from disk with fallback.

    Uses an absolute path anchored to the project root so this works
    regardless of the current working directory at startup.
    """
    sample_path = _PROJECT_ROOT / "data" / "sample_portfolio.json"
    if sample_path.exists():
        try:
            with open(sample_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("holdings", {})
        except Exception as e:
            logger.warning(f"Failed to read sample_portfolio.json: {e}")
    # HARDCODED FALLBACK — only used when data/sample_portfolio.json is missing.
    # These are illustrative weights only; replace with real holdings for production.
    logger.warning(
        "_load_sample_portfolio: using hardcoded fallback portfolio weights. "
        "Create data/sample_portfolio.json to override."
    )
    return {"AAPL": 14.5, "MSFT": 12.0, "NVDA": 9.5, "AMZN": 7.0, "GOOGL": 6.5}


def _extract_trade_parameters(query: str) -> tuple[str | None, float | None]:
    """Extract ticker and percentage delta from trade/rebalance queries."""
    # Matches: "add 2% AAPL", "increase AAPL by 3.5%", "trim NVDA by 1.5%", "allocate 5% to MSFT"
    pattern = re.compile(
        r"(?:add|increase|allocate|trim|reduce|buy|sell)\s+(?:up\s+to\s+)?([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:to|of|in|for)?\s*([A-Za-z]{1,5})",
        re.IGNORECASE,
    )
    match = pattern.search(query)
    if match:
        pct = float(match.group(1))
        ticker = match.group(2).upper()
        if "trim" in query.lower() or "reduce" in query.lower() or "sell" in query.lower():
            pct = -pct
        return ticker, pct

    # Alternative order: "add AAPL 2%"
    alt_pattern = re.compile(
        r"(?:add|increase|trim|reduce)\s+([A-Za-z]{1,5})\s+(?:by\s+)?([0-9]+(?:\.[0-9]+)?)\s*%",
        re.IGNORECASE,
    )
    match_alt = alt_pattern.search(query)
    if match_alt:
        ticker = match_alt.group(1).upper()
        pct = float(match_alt.group(2))
        if "trim" in query.lower() or "reduce" in query.lower():
            pct = -pct
        return ticker, pct

    return None, None


def decision_engine_node(state: RAGState) -> dict:
    """Evaluate quantitative investment decision models and populate decision sheet.

    Runs Piotroski F-Score, Altman Z-Score, Composite Factor Conviction,
    or Portfolio Rebalance Risk Governance depending on query requirements.
    """
    query = state.get("sanitized_query", "").strip()
    target_company = state.get("target_company")
    user_role = state.get("user_role", "research_analyst")
    q_lower = query.lower()

    analysis_results: dict[str, Any] = {}
    summary_lines: list[str] = []

    # 1. Portfolio Trade Rebalance Decision
    trade_ticker, trade_delta = _extract_trade_parameters(query)
    if trade_ticker and trade_delta is not None:
        if user_role not in ("portfolio_manager", "admin"):
            summary_lines.append(
                f"> [!WARNING]\n"
                f"> **Permission Denied**: Role `{user_role}` cannot execute or simulate portfolio rebalancing mandates. "
                f"Requires `portfolio_manager` or `admin`."
            )
            analysis_results["portfolio_trade_decision"] = {
                "decision": "PERMISSION_DENIED",
                "required_role": "portfolio_manager",
            }
        else:
            holdings = _load_sample_portfolio()
            trade_eval = evaluate_portfolio_trade_decision(
                holdings=holdings,
                trade_ticker=trade_ticker,
                trade_weight_delta_pct=trade_delta,
            )
            analysis_results["portfolio_trade_decision"] = trade_eval
            summary_lines.append("### Portfolio Rebalancing Decision Governance")
            summary_lines.append(f"- **Simulated Trade**: `{trade_ticker}` {trade_delta:+.1f}%")
            summary_lines.append(f"- **Decision**: **{trade_eval['decision']}**")
            summary_lines.append(f"- **Post-Trade Weight**: {trade_eval['post_trade_weight_pct']}% (Cap: {trade_eval['mandate_limits']['max_single_position_pct']}%)")
            summary_lines.append(f"- **HHI Index Impact**: {trade_eval['pre_trade_hhi']} -> {trade_eval['post_trade_hhi']} (Delta: {trade_eval['delta_hhi']:+.1f})")
            if trade_eval["violations"]:
                summary_lines.append(f"- **Violations**: {'; '.join(trade_eval['violations'])}")
            else:
                summary_lines.append("- **Mandate Compliance**: All concentration ceilings (HHI < 2500, Single < 15%) respected.")

    # 2. Company Fundamental Decision Modeling (Piotroski & Altman Z)
    # Check if target company or canonical ticker is in query or state
    fundamentals = _load_benchmark_fundamentals()
    detected_slug = target_company.lower() if target_company else None
    if not detected_slug:
        for ticker_key in fundamentals:
            if ticker_key in q_lower or fundamentals[ticker_key]["ticker"].lower() in q_lower:
                detected_slug = ticker_key
                break

    # If not in local benchmark, extract candidate ticker symbol
    if not detected_slug:
        candidate_tickers = re.findall(r"\b[A-Z]{2,5}\b", query)
        for t in candidate_tickers:
            if t not in ("SEC", "GAAP", "CAGR", "EBIT", "EBITDA", "EPS", "USD", "ROE", "ROA", "FY", "HHI", "VAR", "CVAR"):
                detected_slug = t.lower()
                break

    fund_data = fundamentals.get(detected_slug or "")
    if not fund_data and detected_slug:
        fund_data = get_fundamentals(detected_slug)

    is_quant_query = any(k in q_lower for k in [
        "piotroski", "altman", "z-score", "f-score", "fundamental health", "bankruptcy",
        "distress", "solvency", "conviction", "decision", "overweight", "rating", "financial health"
    ]) or detected_slug is not None

    if fund_data and (is_quant_query or state.get("query_intent") in ("market_data", "retrieval")):
        # Calculate Piotroski F-Score
        piotroski = calculate_piotroski_f_score(fund_data)
        analysis_results["piotroski_f_score"] = piotroski

        # Calculate Altman Z-Score
        altman = calculate_altman_z_score(fund_data)
        analysis_results["altman_z_score"] = altman

        # Calculate Composite Factor Conviction
        conviction = evaluate_composite_investment_conviction(
            piotroski_score=piotroski["piotroski_f_score"],
            altman_zone=altman["solvency_zone"],
            valuation_multiples={"pe_ratio": fund_data.get("pe_ratio"), "fcf_yield_pct": fund_data.get("fcf_yield_pct")},
            profitability_metrics={"return_on_equity_pct": fund_data.get("roe_pct")},
        )
        analysis_results["composite_conviction"] = conviction

        summary_lines.append(f"### Quantitative Investment Decision Sheet: {fund_data['company']} ({fund_data['ticker']})")
        summary_lines.append(
            f"| Model | Score / Zone | Institutional Rating | Key Attribution |\n"
            f"| :--- | :--- | :--- | :--- |\n"
            f"| **Piotroski F-Score** | {piotroski['piotroski_f_score']}/9 | **{piotroski['financial_health']}** | {piotroski['conviction_guidance']} |\n"
            f"| **Altman Z-Score** | {altman['altman_z_score']} ({altman['solvency_zone']}) | **{altman['solvency_rating']}** | {altman['estimated_default_risk']} |\n"
            f"| **Factor Conviction** | {conviction['composite_quant_score']}/100 | **{conviction['rating']}** | {conviction['summary']} |"
        )

    # 3. Portfolio Value-at-Risk (VaR) & Expected Shortfall (CVaR)
    wants_var = any(k in q_lower for k in ["var", "value at risk", "cvar", "expected shortfall", "tail risk"])
    if wants_var:
        holdings = _load_sample_portfolio()
        conf_level = 0.99 if "99" in q_lower else 0.95
        var_eval = calculate_portfolio_var_cvar(holdings=holdings, confidence_level=conf_level)
        analysis_results["portfolio_var_cvar"] = var_eval
        summary_lines.append("### Institutional Portfolio Risk & Tail Exposure (VaR / CVaR)")
        summary_lines.append(
            f"- **Confidence Level**: {var_eval['confidence_level']} (Horizon: {var_eval['time_horizon_days']} trading day)\n"
            f"- **Parametric VaR**: **{var_eval['parametric_var_pct']}%** (${var_eval['parametric_var_usd']:,.0f})\n"
            f"- **Conditional VaR (Expected Shortfall)**: **{var_eval['conditional_var_expected_shortfall_pct']}%** (${var_eval['conditional_var_expected_shortfall_usd']:,.0f})\n"
            f"- **Annualized Volatility**: {var_eval['annualized_volatility_pct']}%\n"
            f"- **Assessment**: {var_eval['risk_interpretation']}"
        )

    # 4. Constrained Portfolio Optimization (Ledoit-Wolf Covariance)
    wants_optimization = any(k in q_lower for k in ["optimize", "minimum variance", "min variance", "reweight", "efficient frontier", "shrinkage"])
    if wants_optimization:
        if user_role not in ("portfolio_manager", "admin"):
            summary_lines.append(
                f"> [!WARNING]\n"
                f"> **Permission Denied**: Role `{user_role}` cannot run portfolio optimization mandates. Requires `portfolio_manager`."
            )
        else:
            holdings = _load_sample_portfolio()
            opt_eval = optimize_portfolio_weights(holdings=holdings, max_single_cap=0.15)
            analysis_results["portfolio_optimization"] = opt_eval
            summary_lines.append("### Ledoit-Wolf Constrained Portfolio Optimization")
            summary_lines.append(
                f"- **Objective**: Minimum Variance under 15% single-position mandate cap\n"
                f"- **Volatility Profile**: {opt_eval['baseline_annual_volatility_pct']}% -> **{opt_eval['optimized_annual_volatility_pct']}%** (Reduction: -{opt_eval['volatility_reduction_bps']} bps)\n"
                f"- **Concentration (HHI)**: {opt_eval['baseline_hhi']} -> **{opt_eval['optimized_hhi']}**\n"
                f"- **Top Reweighted Assets**: {', '.join([f'{k} ({v:+.1f}%)' for k, v in list(opt_eval['weight_adjustments_pct'].items())[:4]])}"
            )

    # 5. Factor Sensitivity & Systematic Risk Decomposition
    wants_factors = any(k in q_lower for k in ["factor", "beta", "fama french", "systematic risk", "idiosyncratic", "style tilt"])
    if wants_factors and (detected_slug or target_company):
        ticker_symbol = (fund_data["ticker"] if fund_data else (target_company or "AAPL")).upper()
        factor_eval = calculate_factor_exposures(ticker=ticker_symbol)
        analysis_results["factor_exposures"] = factor_eval
        summary_lines.append(f"### Factor Exposure & Risk Decomposition: `{ticker_symbol}`")
        summary_lines.append(
            f"- **Market Beta**: **{factor_eval['market_beta']}** (Benchmark: `{factor_eval['benchmark']}`)\n"
            f"- **Style Tilt**: **{factor_eval['style_classification']}** (SMB: {factor_eval['size_exposure_smb']:+.2f}, HML: {factor_eval['value_exposure_hml']:+.2f})\n"
            f"- **Risk Decomposition**: Systematic: {factor_eval['systematic_risk_pct']}% | Idiosyncratic: {factor_eval['idiosyncratic_risk_pct']}% (R²: {factor_eval['r_squared']})"
        )

    if not analysis_results:
        return {"quantitative_analysis": None}

    analysis_results["formatted_summary"] = "\n\n".join(summary_lines)

    # Emit OpenTelemetry tracing log
    try:
        from src.services.event_log import emit
        emit(
            "decision_engine",
            models_executed=list(analysis_results.keys()),
            company=target_company or (fund_data["ticker"] if fund_data else None),
        )
    except Exception:
        pass

    logger.info(f"Quantitative decision engine evaluated: {list(analysis_results.keys())}")
    return {"quantitative_analysis": analysis_results}
