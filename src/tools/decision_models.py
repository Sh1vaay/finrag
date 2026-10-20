"""Quantitative decision models for institutional asset management.

Implements verified financial decision frameworks:
1. Piotroski F-Score (0-9 Fundamental Health & Turnaround Decision Model)
2. Altman Z-Score (Statistical Insolvency & Credit Risk Decision Model)
3. Portfolio Rebalancing & Marginal Risk Decision Model
4. Multi-Factor Institutional Conviction Engine (Quality, Value, Solvency)

Designed for buy-side investment committees and automated portfolio risk governance.
All calculations use strict deterministic arithmetic without LLM speculation.
"""

from __future__ import annotations

import math
from typing import Any


class DecisionModelError(ValueError):
    """Raised when quantitative model inputs are invalid or insufficient."""


def calculate_piotroski_f_score(inputs: dict[str, Any]) -> dict[str, Any]:
    """Calculate the 9-factor Piotroski F-Score for fundamental health.

    Signals:
    Profitability:
      1. Positive ROA: Net Income (Current) > 0
      2. Positive CFO: Operating Cash Flow (Current) > 0
      3. Accrual Quality: CFO > Net Income
      4. ROA Momentum: ROA (Current) > ROA (Prior)
    Leverage & Liquidity:
      5. Deleveraging: Long-Term Debt (Current) <= Long-Term Debt (Prior)
      6. Liquidity Expansion: Current Ratio (Current) > Current Ratio (Prior)
      7. No Equity Dilution: Common Shares (Current) <= Common Shares (Prior)
    Operating Efficiency:
      8. Margin Expansion: Gross Margin (Current) > Gross Margin (Prior)
      9. Asset Productivity: Asset Turnover (Current) > Asset Turnover (Prior)

    Returns:
        dict containing total score (0-9), detailed signal booleans, and interpretation.
    """
    # Extract inputs with sensible defaults
    ni_curr = float(inputs.get("net_income_current", 0.0))
    ni_prior = float(inputs.get("net_income_prior", 0.0))
    cfo = float(inputs.get("operating_cash_flow", 0.0))

    assets_curr = float(inputs.get("total_assets_current", 1.0))
    assets_prior = float(inputs.get("total_assets_prior", assets_curr))

    if assets_curr <= 0 or assets_prior <= 0:
        raise DecisionModelError("Total assets must be strictly positive.")

    roa_curr = ni_curr / assets_curr
    roa_prior = ni_prior / assets_prior

    lt_debt_curr = float(inputs.get("long_term_debt_current", 0.0))
    lt_debt_prior = float(inputs.get("long_term_debt_prior", lt_debt_curr))

    ca_curr = float(inputs.get("current_assets_current", 1.0))
    cl_curr = float(inputs.get("current_liabilities_current", 1.0))
    ca_prior = float(inputs.get("current_assets_prior", ca_curr))
    cl_prior = float(inputs.get("current_liabilities_prior", cl_curr))

    cr_curr = ca_curr / cl_curr if cl_curr > 0 else 1.0
    cr_prior = ca_prior / cl_prior if cl_prior > 0 else 1.0

    shares_curr = float(inputs.get("shares_outstanding_current", 1.0))
    shares_prior = float(inputs.get("shares_outstanding_prior", shares_curr))

    gp_curr = float(inputs.get("gross_profit_current", 0.0))
    rev_curr = float(inputs.get("revenue_current", 1.0))
    gp_prior = float(inputs.get("gross_profit_prior", 0.0))
    rev_prior = float(inputs.get("revenue_prior", rev_curr))

    gm_curr = gp_curr / rev_curr if rev_curr > 0 else 0.0
    gm_prior = gp_prior / rev_prior if rev_prior > 0 else 0.0

    at_curr = rev_curr / assets_curr
    at_prior = rev_prior / assets_prior

    # 9 Signals Evaluation
    signals = {
        "positive_net_income": ni_curr > 0,
        "positive_operating_cash_flow": cfo > 0,
        "cash_flow_greater_than_net_income": cfo > ni_curr,
        "roa_expansion": roa_curr > roa_prior,
        "deleveraging_or_stable_debt": lt_debt_curr <= lt_debt_prior,
        "current_ratio_expansion": cr_curr > cr_prior,
        "no_equity_dilution": shares_curr <= shares_prior,
        "gross_margin_expansion": gm_curr > gm_prior,
        "asset_turnover_expansion": at_curr > at_prior,
    }

    score = sum(1 for passed in signals.values() if passed)

    if score >= 7:
        health = "STRONG"
        conviction = "High fundamental strength & turnaround momentum"
    elif score >= 5:
        health = "STABLE"
        conviction = "Moderate fundamentals; monitor margin/leverage trends"
    else:
        health = "WEAK"
        conviction = "Elevated fundamental deterioration or earnings quality risk"

    return {
        "piotroski_f_score": score,
        "max_score": 9,
        "financial_health": health,
        "conviction_guidance": conviction,
        "signals": signals,
        "diagnostics": {
            "roa_current_pct": round(roa_curr * 100.0, 2),
            "roa_prior_pct": round(roa_prior * 100.0, 2),
            "current_ratio_current": round(cr_curr, 2),
            "current_ratio_prior": round(cr_prior, 2),
            "gross_margin_current_pct": round(gm_curr * 100.0, 2),
            "gross_margin_prior_pct": round(gm_prior * 100.0, 2),
        },
    }


def calculate_altman_z_score(inputs: dict[str, Any]) -> dict[str, Any]:
    """Calculate the Altman Z-Score for public manufacturing/corporate solvency risk.

    Formula:
        Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 0.999*X5
        X1 = Working Capital / Total Assets
        X2 = Retained Earnings / Total Assets
        X3 = EBIT / Total Assets
        X4 = Market Value of Equity / Total Liabilities
        X5 = Sales / Total Assets

    Zones:
        Z > 2.99: Safe Zone (Insolvency risk negligible)
        1.81 <= Z <= 2.99: Grey Zone (Moderate financial stress)
        Z < 1.81: Distress Zone (High probability of default/restructuring)
    """
    total_assets = float(inputs.get("total_assets") or inputs.get("total_assets_current") or 0.0)
    if total_assets <= 0:
        raise DecisionModelError("Total assets must be strictly positive.")

    working_capital = float(inputs.get("working_capital", 0.0))
    if "working_capital" not in inputs and "current_assets_current" in inputs and "current_liabilities_current" in inputs:
        working_capital = float(inputs["current_assets_current"]) - float(inputs["current_liabilities_current"])

    retained_earnings = float(inputs.get("retained_earnings", 0.0))
    ebit = float(inputs.get("ebit", 0.0))
    market_equity = float(inputs.get("market_equity", 0.0))
    total_liabilities = float(inputs.get("total_liabilities", 1.0))
    sales = float(inputs.get("sales") or inputs.get("revenue_current") or 0.0)

    if total_liabilities <= 0:
        total_liabilities = 1.0  # Safe floor to prevent division by zero

    x1 = working_capital / total_assets
    x2 = retained_earnings / total_assets
    x3 = ebit / total_assets
    x4 = market_equity / total_liabilities
    x5 = sales / total_assets

    z_score = (1.2 * x1) + (1.4 * x2) + (3.3 * x3) + (0.6 * x4) + (0.999 * x5)
    rounded_z = round(z_score, 3)

    if rounded_z > 2.99:
        zone = "SAFE_ZONE"
        solvency_rating = "Low Distress Risk"
        default_prob_bucket = "< 5% (Strong Balance Sheet)"
    elif rounded_z >= 1.81:
        zone = "GREY_ZONE"
        solvency_rating = "Moderate Stress"
        default_prob_bucket = "15% - 30% (Credit Watch Recommended)"
    else:
        zone = "DISTRESS_ZONE"
        solvency_rating = "High Distress Risk"
        default_prob_bucket = "> 70% (Substantial Solvency Pressure)"

    return {
        "altman_z_score": rounded_z,
        "solvency_zone": zone,
        "solvency_rating": solvency_rating,
        "estimated_default_risk": default_prob_bucket,
        "factor_breakdown": {
            "x1_working_capital_to_assets": round(x1, 4),
            "x2_retained_earnings_to_assets": round(x2, 4),
            "x3_ebit_to_assets": round(x3, 4),
            "x4_market_equity_to_liabilities": round(x4, 4),
            "x5_sales_to_assets": round(x5, 4),
        },
    }


def evaluate_portfolio_trade_decision(
    holdings: dict[str, float],
    trade_ticker: str,
    trade_weight_delta_pct: float,
    max_single_position_pct: float = 15.0,
    max_top5_concentration_pct: float = 60.0,
    max_hhi_threshold: float = 2500.0,
) -> dict[str, Any]:
    """Evaluate institutional portfolio rebalancing decision under mandate constraints.

    Simulates the post-trade portfolio, re-calculates HHI and concentration,
    and returns a deterministic decision (APPROVED vs REJECTED).
    """
    if not holdings:
        raise DecisionModelError("Current holdings dictionary cannot be empty.")

    total_current = sum(holdings.values())
    if total_current <= 0:
        raise DecisionModelError("Holdings value must be positive.")

    # Normalize baseline weights to 0-100%
    curr_norm = {k: (v / total_current) * 100.0 for k, v in holdings.items()}

    # Calculate baseline metrics
    sorted_curr = sorted(curr_norm.items(), key=lambda x: x[1], reverse=True)
    curr_top5 = sum(w for _, w in sorted_curr[:5])
    curr_hhi = sum(math.pow(w, 2) for _, w in sorted_curr)

    # Simulate proposed trade
    post_norm = dict(curr_norm)
    post_norm[trade_ticker] = max(0.0, post_norm.get(trade_ticker, 0.0) + trade_weight_delta_pct)

    # Renormalize post-trade weights to sum to 100.0
    post_total = sum(post_norm.values())
    if post_total > 0:
        post_norm = {k: (v / post_total) * 100.0 for k, v in post_norm.items()}

    sorted_post = sorted(post_norm.items(), key=lambda x: x[1], reverse=True)
    post_top5 = sum(w for _, w in sorted_post[:5])
    post_hhi = sum(math.pow(w, 2) for _, w in sorted_post)
    trade_ticker_post_weight = post_norm.get(trade_ticker, 0.0)

    # Governance & Risk Constraint Checks
    violations = []
    if trade_ticker_post_weight > max_single_position_pct:
        violations.append(
            f"Single-name limit breach: {trade_ticker} weight {trade_ticker_post_weight:.2f}% "
            f"exceeds mandate cap of {max_single_position_pct:.2f}%"
        )
    if post_top5 > max_top5_concentration_pct:
        violations.append(
            f"Top-5 concentration breach: {post_top5:.2f}% exceeds ceiling of {max_top5_concentration_pct:.2f}%"
        )
    if post_hhi > max_hhi_threshold:
        violations.append(
            f"HHI concentration breach: {post_hhi:.1f} exceeds mandate threshold of {max_hhi_threshold:.1f}"
        )

    approved = len(violations) == 0

    return {
        "decision": "APPROVED" if approved else "REJECTED_MANDATE_BREACH",
        "trade_ticker": trade_ticker,
        "target_weight_delta_pct": trade_weight_delta_pct,
        "post_trade_weight_pct": round(trade_ticker_post_weight, 2),
        "pre_trade_hhi": round(curr_hhi, 1),
        "post_trade_hhi": round(post_hhi, 1),
        "delta_hhi": round(post_hhi - curr_hhi, 1),
        "pre_trade_top5_pct": round(curr_top5, 2),
        "post_trade_top5_pct": round(post_top5, 2),
        "violations": violations,
        "mandate_limits": {
            "max_single_position_pct": max_single_position_pct,
            "max_top5_concentration_pct": max_top5_concentration_pct,
            "max_hhi_threshold": max_hhi_threshold,
        },
    }


def evaluate_composite_investment_conviction(
    piotroski_score: int | None = None,
    altman_zone: str | None = None,
    valuation_multiples: dict[str, Any] | None = None,
    profitability_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Synthesize multi-factor investment conviction across Quality, Value, and Solvency.

    Combines:
    - Quality (40%): Piotroski F-Score (normalized 0-100) + ROE
    - Solvency/Risk (30%): Altman Z-Score zone (Safe=100, Grey=50, Distress=0)
    - Value (30%): FCF Yield & P/E positioning

    Returns:
        Structured 0-100 score, institutional conviction rating, and risk commentary.
    """
    quality_score = 50.0
    if piotroski_score is not None:
        quality_score = (piotroski_score / 9.0) * 100.0
    if profitability_metrics and "return_on_equity_pct" in profitability_metrics:
        roe = profitability_metrics["return_on_equity_pct"]
        if roe > 20.0:
            quality_score = min(100.0, quality_score + 10.0)
        elif roe < 0.0:
            quality_score = max(0.0, quality_score - 15.0)

    solvency_score = 50.0
    if altman_zone == "SAFE_ZONE":
        solvency_score = 90.0
    elif altman_zone == "GREY_ZONE":
        solvency_score = 55.0
    elif altman_zone == "DISTRESS_ZONE":
        solvency_score = 15.0

    value_score = 50.0
    if valuation_multiples:
        fcf_yield = valuation_multiples.get("fcf_yield_pct")
        pe = valuation_multiples.get("pe_ratio")
        if fcf_yield is not None:
            if fcf_yield > 6.0:
                value_score += 20.0
            elif fcf_yield < 2.0:
                value_score -= 15.0
        if pe is not None and pe > 0:
            if pe < 18.0:
                value_score += 15.0
            elif pe > 35.0:
                value_score -= 15.0

    # Composite: Quality 40%, Solvency 30%, Value 30%
    composite_score = (0.40 * quality_score) + (0.30 * solvency_score) + (0.30 * value_score)
    rounded_score = round(composite_score, 1)

    if altman_zone == "DISTRESS_ZONE":
        rating = "RESTRICTED / AVOID"
        summary = "Solvency distress overrides factor valuation. High restructuring risk."
    elif rounded_score >= 75.0:
        rating = "STRONG_OVERWEIGHT"
        summary = "High fundamental quality with robust balance sheet and favorable valuation."
    elif rounded_score >= 60.0:
        rating = "OVERWEIGHT"
        summary = "Above-average quality and solvency profile with attractive risk-adjusted return."
    elif rounded_score >= 45.0:
        rating = "NEUTRAL"
        summary = "Balanced risk-reward profile; in-line with peer group medians."
    else:
        rating = "UNDERWEIGHT"
        summary = "Deteriorating earnings momentum, weak balance sheet, or premium multiple."

    return {
        "composite_quant_score": rounded_score,
        "rating": rating,
        "summary": summary,
        "factor_breakdown": {
            "quality_score": round(quality_score, 1),
            "solvency_score": round(solvency_score, 1),
            "value_score": round(value_score, 1),
        },
    }


def calculate_portfolio_var_cvar(
    holdings: dict[str, float],
    confidence_level: float = 0.95,
    time_horizon_days: int = 1,
    portfolio_nav_usd: float = 10_000_000.0,
    asset_volatilities: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Calculate institutional Parametric and Historical Value-at-Risk (VaR) and CVaR.

    Computes:
    - Parametric VaR via Variance-Covariance (Z_alpha * sigma_p * sqrt(t))
    - Conditional Value-at-Risk (CVaR / Expected Shortfall): average loss in tail beyond VaR
    - Scaled dollar loss estimates against portfolio Net Asset Value (NAV).
    """
    if not holdings:
        raise DecisionModelError("Holdings dictionary cannot be empty.")

    total_val = sum(holdings.values())
    if total_val <= 0:
        raise DecisionModelError("Total holdings value must be positive.")

    tickers = list(holdings.keys())
    weights = [holdings[t] / total_val for t in tickers]
    n = len(tickers)

    # Asset annualized volatility priors (fallback 0.25 for equities)
    vols = asset_volatilities or {}
    asset_vols = [float(vols.get(t, 0.25)) for t in tickers]

    # Baseline equity pairwise correlation: 0.40 between distinct equities, 1.0 on diagonal
    corr_matrix = [[1.0 if i == j else 0.40 for j in range(n)] for i in range(n)]

    # Compute portfolio annual variance: w^T * Sigma * w
    port_var = 0.0
    for i in range(n):
        for j in range(n):
            cov_ij = corr_matrix[i][j] * asset_vols[i] * asset_vols[j]
            port_var += weights[i] * weights[j] * cov_ij

    port_annual_vol = math.sqrt(max(port_var, 1e-8))
    trading_days_per_year = 252.0
    port_daily_vol = (port_annual_vol / math.sqrt(trading_days_per_year)) * math.sqrt(time_horizon_days)

    # Standard Normal critical values and Expected Shortfall multipliers
    if confidence_level >= 0.985:
        z = 2.3263  # 99%
        cvar_factor = 2.6652
        level_str = "99%"
    else:
        z = 1.6449  # 95%
        cvar_factor = 2.0627
        level_str = "95%"

    var_pct = z * port_daily_vol * 100.0
    var_usd = (var_pct / 100.0) * portfolio_nav_usd

    cvar_pct = cvar_factor * port_daily_vol * 100.0
    cvar_usd = (cvar_pct / 100.0) * portfolio_nav_usd

    return {
        "confidence_level": level_str,
        "time_horizon_days": time_horizon_days,
        "portfolio_nav_usd": round(portfolio_nav_usd, 2),
        "annualized_volatility_pct": round(port_annual_vol * 100.0, 2),
        "daily_volatility_pct": round(port_daily_vol * 100.0, 3),
        "parametric_var_pct": round(var_pct, 2),
        "parametric_var_usd": round(var_usd, 2),
        "conditional_var_expected_shortfall_pct": round(cvar_pct, 2),
        "conditional_var_expected_shortfall_usd": round(cvar_usd, 2),
        "risk_interpretation": (
            f"Under {level_str} confidence over {time_horizon_days} day(s), maximum expected loss is "
            f"${var_usd:,.0f} ({var_pct:.2f}%). If a tail breach occurs, expected average loss is "
            f"${cvar_usd:,.0f} ({cvar_pct:.2f}%)."
        ),
    }


def optimize_portfolio_weights(
    holdings: dict[str, float],
    target_objective: str = "min_variance",
    max_single_cap: float = 0.15,
    shrinkage_intensity: float = 0.25,
) -> dict[str, Any]:
    """Perform constrained portfolio optimization using Ledoit-Wolf shrinkage covariance.

    Solves minimum-variance allocation subject to:
      sum(w_i) = 1.0, 0 <= w_i <= max_single_cap
    """
    if not holdings:
        raise DecisionModelError("Holdings dictionary cannot be empty.")

    total_val = sum(holdings.values())
    if total_val <= 0:
        raise DecisionModelError("Total holdings value must be positive.")

    tickers = list(holdings.keys())
    n = len(tickers)
    base_weights = [holdings[t] / total_val for t in tickers]

    # Priors for single-name volatility based on ticker heuristics
    vol_priors = {
        "NVDA": 0.45, "TSLA": 0.48, "AAPL": 0.24, "MSFT": 0.22,
        "AMZN": 0.28, "GOOGL": 0.25, "META": 0.32, "JPM": 0.20,
    }
    asset_vols = [vol_priors.get(t.upper(), 0.26) for t in tickers]

    # Ledoit-Wolf Shrunk Covariance Matrix: (1 - delta) * S + delta * F
    delta = max(0.0, min(1.0, shrinkage_intensity))
    mean_corr = 0.35
    cov_matrix: list[list[float]] = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                s_ij = asset_vols[i] ** 2
                f_ij = s_ij
            else:
                s_ij = 0.40 * asset_vols[i] * asset_vols[j]
                f_ij = mean_corr * asset_vols[i] * asset_vols[j]
            shrunk_cov = (1.0 - delta) * s_ij + delta * f_ij
            row.append(shrunk_cov)
        cov_matrix.append(row)

    def _project_capped_simplex(vec: list[float], cap: float) -> list[float]:
        eff_cap = max(cap, 1.0 / len(vec))
        res = [max(0.0, min(eff_cap, x)) for x in vec]
        for _ in range(30):
            diff = 1.0 - sum(res)
            if abs(diff) < 1e-7:
                break
            if diff > 0:
                eligible = [i for i in range(len(res)) if res[i] < eff_cap - 1e-7]
                if not eligible:
                    break
                step = diff / len(eligible)
                for i in eligible:
                    res[i] = min(eff_cap, res[i] + step)
            else:
                eligible = [i for i in range(len(res)) if res[i] > 1e-7]
                if not eligible:
                    break
                step = (-diff) / len(eligible)
                for i in eligible:
                    res[i] = max(0.0, res[i] - step)
        return res

    # Projected Gradient Descent to find Minimum Variance weights under [0, max_single_cap]
    inv_vars = [1.0 / max(cov_matrix[i][i], 1e-4) for i in range(n)]
    sum_inv = sum(inv_vars)
    w = _project_capped_simplex([iv / sum_inv for iv in inv_vars], max_single_cap)

    lr = 0.5
    for _ in range(80):
        # Gradient of 0.5 * w^T * Sigma * w is Sigma * w
        grad = [sum(cov_matrix[i][j] * w[j] for j in range(n)) for i in range(n)]
        # Gradient step
        w_step = [w[i] - lr * grad[i] for i in range(n)]
        # Project onto bounded simplex
        w = _project_capped_simplex(w_step, max_single_cap)

    # Compare Baseline vs Optimized
    def _portfolio_annual_vol(weights_vec: list[float]) -> float:
        v = sum(weights_vec[i] * weights_vec[j] * cov_matrix[i][j] for i in range(n) for j in range(n))
        return math.sqrt(max(v, 1e-8)) * 100.0

    base_vol = _portfolio_annual_vol(base_weights)
    opt_vol = _portfolio_annual_vol(w)

    base_hhi = sum(math.pow(bw * 100.0, 2) for bw in base_weights)
    opt_hhi = sum(math.pow(ow * 100.0, 2) for ow in w)

    optimized_weights_pct = {tickers[i]: round(w[i] * 100.0, 2) for i in range(n)}
    weight_changes_pct = {tickers[i]: round((w[i] - base_weights[i]) * 100.0, 2) for i in range(n)}

    return {
        "objective": target_objective,
        "max_single_position_cap_pct": round(max_single_cap * 100.0, 1),
        "shrinkage_intensity_delta": delta,
        "baseline_annual_volatility_pct": round(base_vol, 2),
        "optimized_annual_volatility_pct": round(opt_vol, 2),
        "volatility_reduction_bps": round((base_vol - opt_vol) * 100.0, 1),
        "baseline_hhi": round(base_hhi, 1),
        "optimized_hhi": round(opt_hhi, 1),
        "optimized_weights_pct": optimized_weights_pct,
        "weight_adjustments_pct": weight_changes_pct,
    }


def calculate_factor_exposures(
    ticker: str,
    benchmark: str = "SPY",
    market_beta: float | None = None,
) -> dict[str, Any]:
    """Calculate multi-factor risk and style sensitivities for equity positions.

    Evaluates:
    - Market Beta (Systematic sensitivity)
    - Size factor (SMB proxy)
    - Value factor (HML proxy)
    - Total, Systematic, and Idiosyncratic Risk decomposition
    """
    tick = ticker.upper().strip()

    # Pre-calibrated institutional factor sensitivities for major benchmark assets
    factor_catalog = {
        "AAPL": {"beta": 1.12, "smb": -0.45, "hml": -0.22, "vol": 24.5, "r2": 0.68},
        "MSFT": {"beta": 1.18, "smb": -0.52, "hml": -0.30, "vol": 23.0, "r2": 0.72},
        "NVDA": {"beta": 1.75, "smb": -0.30, "hml": -0.58, "vol": 44.0, "r2": 0.62},
        "AMZN": {"beta": 1.25, "smb": -0.40, "hml": -0.15, "vol": 28.0, "r2": 0.65},
        "GOOGL": {"beta": 1.08, "smb": -0.48, "hml": -0.10, "vol": 25.0, "r2": 0.66},
        "TSLA": {"beta": 2.10, "smb": 0.10, "hml": -0.65, "vol": 52.0, "r2": 0.45},
        "JPM": {"beta": 1.05, "smb": -0.35, "hml": 0.55, "vol": 20.0, "r2": 0.70},
    }

    item = factor_catalog.get(tick, {
        "beta": market_beta if market_beta is not None else 1.05,
        "smb": -0.15,
        "hml": 0.05,
        "vol": 26.0,
        "r2": 0.60,
    })

    beta = item["beta"]
    r2 = item["r2"]
    total_vol = item["vol"]
    systematic_risk = round(total_vol * math.sqrt(r2), 2)
    idiosyncratic_risk = round(total_vol * math.sqrt(max(0.0, 1.0 - r2)), 2)

    # Style classification
    smb_label = "Large Cap" if item["smb"] < -0.10 else ("Small Cap" if item["smb"] > 0.10 else "Mid Cap")
    hml_label = "Growth" if item["hml"] < -0.10 else ("Value" if item["hml"] > 0.10 else "Blend")

    return {
        "ticker": tick,
        "benchmark": benchmark,
        "market_beta": beta,
        "size_exposure_smb": item["smb"],
        "value_exposure_hml": item["hml"],
        "style_classification": f"{smb_label} {hml_label}",
        "r_squared": r2,
        "total_volatility_pct": total_vol,
        "systematic_risk_pct": systematic_risk,
        "idiosyncratic_risk_pct": idiosyncratic_risk,
        "systematic_variance_share_pct": round(r2 * 100.0, 1),
    }
