"""Unit tests for SEC EDGAR filing & fundamentals service."""

import sys
from pathlib import Path

# Ensure root directory is on pythonpath
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.services.sec_edgar_service import (
    fetch_recent_filings,
    fetch_sec_company_facts,
    get_cik,
    get_fundamentals,
)


def test_cik_resolution():
    assert get_cik("AAPL") == "0000320193"
    assert get_cik("MSFT") == "0000789019"
    assert get_cik("NVDA") == "0001045810"
    assert get_cik("JPM") == "0000019617"
    assert get_cik("NONEXISTENT_XYZ") is None


def test_get_fundamentals_fallback_and_structure():
    funds = get_fundamentals("AAPL")
    assert funds["ticker"] == "AAPL"
    assert "revenue_current" in funds
    assert "net_income_current" in funds
    assert "operating_cash_flow" in funds
    assert "total_assets_current" in funds
    assert "pe_ratio" in funds
    assert "roe_pct" in funds


def test_unknown_ticker_synthetic_baseline():
    funds = get_fundamentals("XYZCORP")
    assert funds["ticker"] == "XYZCORP"
    assert funds["revenue_current"] > 0
    assert funds["total_assets_current"] > 0
    assert "working_capital" in funds


def test_recent_filings_structure():
    # If network/SEC available or offline, must return list of dicts without crashing
    filings = fetch_recent_filings("AAPL", form_type="10-K")
    assert isinstance(filings, list)
    if filings:
        first = filings[0]
        assert first["ticker"] == "AAPL"
        assert first["form"] == "10-K"
        assert "document_url" in first


if __name__ == "__main__":
    test_cik_resolution()
    test_get_fundamentals_fallback_and_structure()
    test_unknown_ticker_synthetic_baseline()
    test_recent_filings_structure()
    print("All SEC EDGAR unit tests passed successfully!")
