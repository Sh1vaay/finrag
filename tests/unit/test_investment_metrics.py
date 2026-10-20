"""Unit tests for quantitative investment analysis tools."""

try:
    import pytest
except ImportError:
    import re

    class _MockPytest:
        @staticmethod
        def raises(exc_type, match=None):
            class _Context:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type_val, exc_val, exc_tb):
                    if exc_type_val is None:
                        raise AssertionError(f"Expected {exc_type} but no exception was raised")
                    if not issubclass(exc_type_val, exc_type):
                        return False
                    if match and not re.search(match, str(exc_val)):
                        raise AssertionError(f"Exception message '{exc_val}' did not match pattern '{match}'")
                    return True

            return _Context()

    pytest = _MockPytest()

from src.tools.investment_metrics import (
    FinancialCalculationError,
    calculate_growth_metrics,
    calculate_portfolio_risk_metrics,
    calculate_profitability_metrics,
    calculate_valuation_multiples,
)


def test_calculate_valuation_multiples_basic():
    # Apple-like figures: share price $180, EPS $6.00, P/E = 30.0
    res = calculate_valuation_multiples(
        share_price=180.0,
        earnings_per_share=6.0,
        enterprise_value=2800000.0,
        ebitda=125000.0,
        book_value_per_share=4.0,
        free_cash_flow=100000.0,
        shares_outstanding=15500.0,
    )
    assert res["share_price"] == 180.0
    assert res["pe_ratio"] == 30.0
    assert res["ev_to_ebitda"] == 22.4
    assert res["pb_ratio"] == 45.0
    assert res["market_cap"] == 2790000.0
    assert res["fcf_yield_pct"] == 3.58


def test_calculate_valuation_negative_earnings():
    res = calculate_valuation_multiples(
        share_price=50.0,
        earnings_per_share=-2.5,
    )
    assert res["pe_ratio"] is None
    assert "N/M" in res["pe_interpretation"]


def test_calculate_valuation_invalid_price():
    with pytest.raises(FinancialCalculationError, match="must be positive"):
        calculate_valuation_multiples(share_price=-10.0)


def test_calculate_profitability_metrics():
    # Microsoft-like figures: Rev $211,915M, OpInc $88,523M, NetInc $72,361M
    metrics = calculate_profitability_metrics(
        revenue=211915.0,
        gross_profit=146052.0,
        operating_income=88523.0,
        net_income=72361.0,
        total_equity=206223.0,
        total_assets=411976.0,
        total_debt=59965.0,
    )
    assert metrics["revenue"] == 211915.0
    assert metrics["gross_margin_pct"] == 68.92
    assert metrics["operating_margin_pct"] == 41.77
    assert metrics["net_profit_margin_pct"] == 34.15
    assert metrics["return_on_equity_pct"] == 35.09
    assert metrics["return_on_assets_pct"] == 17.56
    assert metrics["debt_to_equity"] == 0.29


def test_calculate_profitability_invalid_revenue():
    with pytest.raises(FinancialCalculationError, match="must be positive"):
        calculate_profitability_metrics(revenue=0.0)


def test_calculate_growth_metrics():
    # Apple net sales: FY22 $394,328M to FY23 $383,285M (-2.8%)
    res = calculate_growth_metrics(
        current_value=383285.0,
        prior_value=394328.0,
        metric_name="Total Net Sales",
    )
    assert res["metric_name"] == "Total Net Sales"
    assert res["growth_pct"] == -2.80
    assert res["basis_points"] == -280.0
    assert res["absolute_change"] == -11043.0


def test_calculate_cagr():
    # Revenue growing from 100 to 144 over 2 years (20% CAGR)
    res = calculate_growth_metrics(
        current_value=144.0,
        prior_value=100.0,
        periods=2,
    )
    assert res["growth_pct"] == 44.0
    assert res["cagr_pct"] == 20.0


def test_calculate_portfolio_risk_metrics():
    holdings = {
        "AAPL": 25.0,
        "MSFT": 20.0,
        "NVDA": 15.0,
        "AMZN": 10.0,
        "GOOGL": 10.0,
        "META": 8.0,
        "TSLA": 5.0,
        "JPM": 4.0,
        "V": 2.0,
        "UNH": 1.0,
    }
    risk = calculate_portfolio_risk_metrics(holdings)
    assert risk["total_positions"] == 10
    assert risk["top_1_weight_pct"] == 25.0
    assert risk["top_3_weight_pct"] == 60.0
    assert risk["top_5_weight_pct"] == 80.0
    assert risk["hhi_score"] > 1400
    assert "AAPL" in risk["positions_above_10pct"]
    assert "MSFT" in risk["positions_above_10pct"]
    assert len(risk["positions_above_10pct"]) == 3  # AAPL (25%), MSFT (20%), NVDA (15%)


def test_calculate_portfolio_empty():
    with pytest.raises(FinancialCalculationError, match="cannot be empty"):
        calculate_portfolio_risk_metrics({})


if __name__ == "__main__":
    test_calculate_valuation_multiples_basic()
    test_calculate_valuation_negative_earnings()
    test_calculate_valuation_invalid_price()
    test_calculate_profitability_metrics()
    test_calculate_profitability_invalid_revenue()
    test_calculate_growth_metrics()
    test_calculate_cagr()
    test_calculate_portfolio_risk_metrics()
    test_calculate_portfolio_empty()
    print("All quantitative investment metrics unit tests passed successfully!")
