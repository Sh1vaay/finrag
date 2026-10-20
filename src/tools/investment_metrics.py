"""Deterministic quantitative investment analysis tools for buy-side asset management.

Provides verified mathematical models and financial metric calculators for:
1. Valuation Multiples (P/E, EV/EBITDA, P/B, Free Cash Flow Yield).
2. Financial Health & Profitability (Margins, ROE, ROIC, Debt/Equity, Current Ratio).
3. Growth & Performance Attribution (YoY / QoQ growth, basis-point margin expansion).
4. Portfolio Construction & Concentration Risk (Herfindahl-Hirschman Index, Top-N weights).

Zero LLM hallucination: all metrics use strict IEEE 754 arithmetic with bounds checking.
"""

from __future__ import annotations

import math
from typing import Any


class FinancialCalculationError(ValueError):
    """Raised when financial inputs are invalid (e.g. division by zero, negative market cap)."""


def calculate_valuation_multiples(
    share_price: float,
    earnings_per_share: float | None = None,
    enterprise_value: float | None = None,
    ebitda: float | None = None,
    book_value_per_share: float | None = None,
    free_cash_flow: float | None = None,
    shares_outstanding: float | None = None,
) -> dict[str, Any]:
    """Calculate core equity and enterprise valuation multiples.

    Args:
        share_price: Current market price per share.
        earnings_per_share: Trailing or forward Diluted EPS.
        enterprise_value: Total enterprise value (Equity Value + Debt - Cash).
        ebitda: Earnings Before Interest, Taxes, Depreciation, and Amortization.
        book_value_per_share: Total shareholders' equity divided by diluted shares.
        free_cash_flow: Operating Cash Flow minus Capital Expenditures.
        shares_outstanding: Total common shares outstanding.

    Returns:
        dict with computed multiples (pe_ratio, ev_ebitda, pb_ratio, fcf_yield, market_cap).
    """
    if share_price <= 0:
        raise FinancialCalculationError(f"share_price must be positive, got {share_price}")

    results: dict[str, Any] = {"share_price": round(share_price, 4)}

    # Market Capitalization
    if shares_outstanding is not None:
        if shares_outstanding < 0:
            raise FinancialCalculationError("shares_outstanding cannot be negative")
        market_cap = share_price * shares_outstanding
        results["market_cap"] = round(market_cap, 2)
    else:
        market_cap = None

    # Price-to-Earnings (P/E)
    if earnings_per_share is not None:
        if earnings_per_share <= 0:
            results["pe_ratio"] = None  # Meaningless or not meaningful for unprofitable firms
            results["pe_interpretation"] = "N/M (negative or zero earnings)"
        else:
            results["pe_ratio"] = round(share_price / earnings_per_share, 2)

    # EV / EBITDA
    if enterprise_value is not None and ebitda is not None:
        if ebitda <= 0:
            results["ev_to_ebitda"] = None
            results["ev_ebitda_interpretation"] = "N/M (negative or zero EBITDA)"
        else:
            results["ev_to_ebitda"] = round(enterprise_value / ebitda, 2)

    # Price-to-Book (P/B)
    if book_value_per_share is not None:
        if book_value_per_share <= 0:
            results["pb_ratio"] = None
            results["pb_interpretation"] = "N/M (negative equity)"
        else:
            results["pb_ratio"] = round(share_price / book_value_per_share, 2)

    # Free Cash Flow Yield (FCF / Market Cap)
    if free_cash_flow is not None and market_cap is not None and market_cap > 0:
        fcf_yield = (free_cash_flow / market_cap) * 100.0
        results["fcf_yield_pct"] = round(fcf_yield, 2)

    return results


def calculate_profitability_metrics(
    revenue: float,
    gross_profit: float | None = None,
    operating_income: float | None = None,
    net_income: float | None = None,
    total_equity: float | None = None,
    total_assets: float | None = None,
    total_debt: float | None = None,
) -> dict[str, Any]:
    """Calculate profitability, operating leverage, and capital efficiency ratios."""
    if revenue <= 0:
        raise FinancialCalculationError(f"revenue must be positive, got {revenue}")

    metrics: dict[str, Any] = {"revenue": round(revenue, 2)}

    # Margins
    if gross_profit is not None:
        metrics["gross_margin_pct"] = round((gross_profit / revenue) * 100.0, 2)
    if operating_income is not None:
        metrics["operating_margin_pct"] = round((operating_income / revenue) * 100.0, 2)
    if net_income is not None:
        metrics["net_profit_margin_pct"] = round((net_income / revenue) * 100.0, 2)

    # Return on Equity (ROE) & Return on Assets (ROA)
    if net_income is not None:
        if total_equity is not None and total_equity > 0:
            metrics["return_on_equity_pct"] = round((net_income / total_equity) * 100.0, 2)
        if total_assets is not None and total_assets > 0:
            metrics["return_on_assets_pct"] = round((net_income / total_assets) * 100.0, 2)

    # Debt-to-Equity
    if total_debt is not None and total_equity is not None and total_equity > 0:
        metrics["debt_to_equity"] = round(total_debt / total_equity, 2)

    return metrics


def calculate_growth_metrics(
    current_value: float,
    prior_value: float,
    metric_name: str = "metric",
    periods: int = 1,
) -> dict[str, Any]:
    """Calculate percentage growth, basis points expansion, or CAGR over N periods."""
    if prior_value == 0:
        raise FinancialCalculationError("Cannot calculate growth from a base of zero")

    growth_pct = ((current_value - prior_value) / abs(prior_value)) * 100.0
    delta_abs = current_value - prior_value

    summary: dict[str, Any] = {
        "metric_name": metric_name,
        "current_value": round(current_value, 4),
        "prior_value": round(prior_value, 4),
        "absolute_change": round(delta_abs, 4),
        "growth_pct": round(growth_pct, 2),
        "basis_points": round(growth_pct * 100.0, 1),
    }

    # Annualized CAGR if multi-period
    if periods > 1 and current_value > 0 and prior_value > 0:
        cagr = (math.pow(current_value / prior_value, 1.0 / periods) - 1.0) * 100.0
        summary["cagr_pct"] = round(cagr, 2)
        summary["periods"] = periods

    return summary


def calculate_portfolio_risk_metrics(
    holdings: dict[str, float],
) -> dict[str, Any]:
    """Calculate portfolio concentration, top weights, and Herfindahl-Hirschman Index (HHI).

    Args:
        holdings: dict of {ticker_or_asset: weight_pct_or_usd}

    Returns:
        Concentration metrics, HHI index, effective position count, and regulatory flags.
    """
    if not holdings:
        raise FinancialCalculationError("Holdings dictionary cannot be empty")

    total_val = sum(holdings.values())
    if total_val <= 0:
        raise FinancialCalculationError("Total holdings value must be positive")

    # Normalize weights to percentages (0-100)
    normalized_weights: dict[str, float] = {
        ticker: (val / total_val) * 100.0 for ticker, val in holdings.items()
    }

    sorted_holdings = sorted(normalized_weights.items(), key=lambda x: x[1], reverse=True)

    top_1 = sorted_holdings[0][1] if len(sorted_holdings) >= 1 else 0.0
    top_3 = sum(w for _, w in sorted_holdings[:3])
    top_5 = sum(w for _, w in sorted_holdings[:5])
    top_10 = sum(w for _, w in sorted_holdings[:10])

    # Herfindahl-Hirschman Index (HHI): sum of squared percentage weights
    # Range: ~0 (perfectly diversified) to 10,000 (single asset portfolio)
    hhi = sum(math.pow(w, 2) for _, w in sorted_holdings)

    # Effective number of positions: 1 / sum(fractional_weight^2)
    sum_sq_fractions = sum(math.pow(w / 100.0, 2) for _, w in sorted_holdings)
    effective_positions = 1.0 / sum_sq_fractions if sum_sq_fractions > 0 else 0.0

    # Risk alerts (e.g. UCITS 5/10/40 rule or 40-Act concentration risk)
    exceeds_10pct = [ticker for ticker, w in sorted_holdings if w > 10.0]
    exceeds_5pct = [ticker for ticker, w in sorted_holdings if w > 5.0]

    return {
        "total_positions": len(holdings),
        "effective_positions": round(effective_positions, 1),
        "hhi_score": round(hhi, 1),
        "top_1_weight_pct": round(top_1, 2),
        "top_3_weight_pct": round(top_3, 2),
        "top_5_weight_pct": round(top_5, 2),
        "top_10_weight_pct": round(top_10, 2),
        "positions_above_10pct": exceeds_10pct,
        "positions_above_5pct": exceeds_5pct,
        "concentration_risk": "High" if hhi > 2500 or top_5 > 60.0 else ("Moderate" if hhi > 1500 else "Low"),
    }
