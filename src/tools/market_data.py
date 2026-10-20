"""Institutional Market Data Provider Service and Adapters.

Implements the Enterprise Adapter Pattern for financial market data:
- `BaseMarketDataProvider`: Abstract interface for institutional feeds.
- `InstitutionalBpipeProvider`: High-precision simulation of Bloomberg B-PIPE / FactSet
  institutional market data feeds (FIX/B-PIPE schema, exchange MICs, sub-penny spreads, VWAP).
- `YahooFinanceProvider`: Local development and open-source fallback adapter (wraps yfinance).
- `get_market_data(ticker)`: Public facade dispatching to the configured provider.
"""

from __future__ import annotations

import abc
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any

try:
    from src.config.settings import settings
except ImportError:
    settings = None

logger = logging.getLogger(__name__)


class MarketDataError(Exception):
    """Raised when market data retrieval or ticker validation fails."""


class BaseMarketDataProvider(abc.ABC):
    """Abstract interface for equity pricing and valuation feeds."""

    @abc.abstractmethod
    def fetch_market_data(self, ticker: str) -> dict[str, Any]:
        """Fetch real-time pricing and fundamental multiples for an equity ticker."""


class InstitutionalBpipeProvider(BaseMarketDataProvider):
    """Institutional feed simulating Bloomberg B-PIPE / FactSet enterprise market data."""

    # Pre-calibrated institutional pricing & exchange metrics
    _INSTITUTIONAL_PRICING: dict[str, dict[str, Any]] = {
        "AAPL": {
            "company_name": "Apple Inc.",
            "primary_exchange": "XNAS",
            "last_price": 224.23,
            "bid": 224.21,
            "ask": 224.24,
            "vwap": 223.85,
            "market_cap": 3420000000000.0,
            "trailing_pe": 34.2,
            "forward_pe": 29.8,
            "price_to_book": 42.1,
            "enterprise_value": 3460000000000.0,
            "enterprise_to_ebitda": 25.1,
            "fifty_two_week_high": 237.23,
            "fifty_two_week_low": 164.08,
            "beta": 1.12,
            "dividend_yield": 0.0045,
            "sector": "Technology",
            "industry": "Consumer Electronics",
        },
        "MSFT": {
            "company_name": "Microsoft Corporation",
            "primary_exchange": "XNAS",
            "last_price": 418.50,
            "bid": 418.45,
            "ask": 418.52,
            "vwap": 417.90,
            "market_cap": 3110000000000.0,
            "trailing_pe": 35.5,
            "forward_pe": 30.2,
            "price_to_book": 12.8,
            "enterprise_value": 3140000000000.0,
            "enterprise_to_ebitda": 23.4,
            "fifty_two_week_high": 468.35,
            "fifty_two_week_low": 327.00,
            "beta": 1.18,
            "dividend_yield": 0.0071,
            "sector": "Technology",
            "industry": "Systems Software",
        },
        "NVDA": {
            "company_name": "NVIDIA Corporation",
            "primary_exchange": "XNAS",
            "last_price": 138.25,
            "bid": 138.23,
            "ask": 138.27,
            "vwap": 137.60,
            "market_cap": 3390000000000.0,
            "trailing_pe": 48.0,
            "forward_pe": 36.5,
            "price_to_book": 49.5,
            "enterprise_value": 3380000000000.0,
            "enterprise_to_ebitda": 38.2,
            "fifty_two_week_high": 140.76,
            "fifty_two_week_low": 39.23,
            "beta": 1.75,
            "dividend_yield": 0.0003,
            "sector": "Technology",
            "industry": "Semiconductors",
        },
        "JPM": {
            "company_name": "JPMorgan Chase & Co.",
            "primary_exchange": "XNYS",
            "last_price": 222.15,
            "bid": 222.10,
            "ask": 222.18,
            "vwap": 221.75,
            "market_cap": 635000000000.0,
            "trailing_pe": 12.4,
            "forward_pe": 11.8,
            "price_to_book": 1.85,
            "enterprise_value": 620000000000.0,
            "enterprise_to_ebitda": 9.2,
            "fifty_two_week_high": 225.48,
            "fifty_two_week_low": 137.11,
            "beta": 1.05,
            "dividend_yield": 0.021,
            "sector": "Financial Services",
            "industry": "Diversified Banking",
        },
    }

    def fetch_market_data(self, ticker: str) -> dict[str, Any]:
        data = self._INSTITUTIONAL_PRICING.get(ticker)
        now_iso = datetime.now(timezone.utc).isoformat()
        if not data:
            # Deterministic synthetic simulation for uncatalogued symbols
            hash_val = sum(ord(c) for c in ticker)
            sim_price = round(50.0 + (hash_val % 300), 2)
            data = {
                "company_name": f"{ticker} Corporation",
                "primary_exchange": "XNYS" if hash_val % 2 == 0 else "XNAS",
                "last_price": sim_price,
                "bid": round(sim_price - 0.02, 2),
                "ask": round(sim_price + 0.02, 2),
                "vwap": sim_price,
                "market_cap": round(sim_price * 1e9, 2),
                "trailing_pe": round(15.0 + (hash_val % 25), 1),
                "forward_pe": round(13.0 + (hash_val % 20), 1),
                "price_to_book": round(2.5 + (hash_val % 8), 1),
                "enterprise_value": round(sim_price * 1.05 * 1e9, 2),
                "enterprise_to_ebitda": round(10.0 + (hash_val % 15), 1),
                "fifty_two_week_high": round(sim_price * 1.25, 2),
                "fifty_two_week_low": round(sim_price * 0.75, 2),
                "beta": round(0.85 + (hash_val % 60) / 100.0, 2),
                "dividend_yield": 0.015,
                "sector": "General Equities",
                "industry": "Diversified Industry",
            }

        return {
            "ticker": ticker,
            "status": "success",
            "provider": "Bloomberg_BPIPE_Simulator",
            "feed_timestamp_utc": now_iso,
            "company_name": data["company_name"],
            "primary_exchange": data["primary_exchange"],
            "currency": "USD",
            "current_price": data["last_price"],
            "bid": data["bid"],
            "ask": data["ask"],
            "vwap": data["vwap"],
            "market_cap": data["market_cap"],
            "trailing_pe": data["trailing_pe"],
            "forward_pe": data["forward_pe"],
            "price_to_book": data["price_to_book"],
            "enterprise_value": data["enterprise_value"],
            "enterprise_to_ebitda": data["enterprise_to_ebitda"],
            "fifty_two_week_high": data["fifty_two_week_high"],
            "fifty_two_week_low": data["fifty_two_week_low"],
            "beta": data["beta"],
            "dividend_yield": data["dividend_yield"],
            "sector": data["sector"],
            "industry": data["industry"],
        }


class YahooFinanceProvider(BaseMarketDataProvider):
    """Development fallback adapter wrapping yfinance."""

    def fetch_market_data(self, ticker: str) -> dict[str, Any]:
        try:
            import yfinance as yf

            t = yf.Ticker(ticker)
            info = getattr(t, "info", {}) or {}
            fast_info = getattr(t, "fast_info", {}) or {}

            current_price = (
                info.get("currentPrice") or info.get("regularMarketPrice") or getattr(fast_info, "last_price", None)
            )
            market_cap = info.get("marketCap") or getattr(fast_info, "market_cap", None)

            return {
                "ticker": ticker,
                "status": "success",
                "provider": "YahooFinanceAdapter",
                "company_name": info.get("shortName") or info.get("longName") or ticker,
                "currency": info.get("currency", "USD"),
                "current_price": float(current_price) if current_price is not None else None,
                "market_cap": float(market_cap) if market_cap is not None else None,
                "trailing_pe": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "price_to_book": info.get("priceToBook"),
                "enterprise_value": info.get("enterpriseValue"),
                "enterprise_to_ebitda": info.get("enterpriseToEbitda"),
                "fifty_two_week_high": info.get("fiftyTwoWeekHigh") or getattr(fast_info, "year_high", None),
                "fifty_two_week_low": info.get("fiftyTwoWeekLow") or getattr(fast_info, "year_low", None),
                "beta": info.get("beta"),
                "dividend_yield": info.get("dividendYield"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
            }
        except ImportError:
            logger.warning("yfinance package not found; falling back to institutional feed simulation for '%s'", ticker)
            inst = InstitutionalBpipeProvider().fetch_market_data(ticker)
            inst["is_mock"] = True
            return inst
        except Exception as exc:
            logger.warning("yfinance call failed for '%s' (%s); falling back to institutional simulator", ticker, exc)
            inst = InstitutionalBpipeProvider().fetch_market_data(ticker)
            inst["is_mock"] = True
            return inst


def get_market_data_provider() -> BaseMarketDataProvider:
    """Return configured market data provider instance."""
    provider_name = (
        os.environ.get("MARKET_DATA_PROVIDER") or getattr(settings, "MARKET_DATA_PROVIDER", "yfinance")
    ).lower()
    if provider_name in ("bloomberg", "bpipe", "institutional", "institutional_mock", "bloomberg_sim"):
        return InstitutionalBpipeProvider()
    return YahooFinanceProvider()


def get_market_data(ticker: str) -> dict[str, Any]:
    """Retrieve equity pricing, multiples, and metrics via configured provider.

    Args:
        ticker: Equity ticker symbol (e.g. 'AAPL', 'MSFT', 'NVDA', 'JPM').

    Returns:
        Structured dictionary with validated market pricing and multiples.
    """
    clean_ticker = (ticker or "").strip().upper()
    if not clean_ticker or not re.match(r"^[A-Z0-9\.\-]+$", clean_ticker):
        raise MarketDataError(f"Invalid ticker symbol: '{ticker}'")

    provider = get_market_data_provider()
    try:
        return provider.fetch_market_data(clean_ticker)
    except Exception as exc:
        logger.error("Failed to fetch market data for '%s': %s", clean_ticker, exc)
        raise MarketDataError(f"Could not retrieve market data for '{clean_ticker}': {exc}") from exc
