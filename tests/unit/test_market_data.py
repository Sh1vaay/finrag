"""Unit tests for yfinance market data tool in src/tools/market_data.py."""

import unittest
from unittest.mock import MagicMock, patch

from src.tools.market_data import MarketDataError, get_market_data


class TestMarketData(unittest.TestCase):
    def test_invalid_ticker_symbols(self):
        with self.assertRaises(MarketDataError):
            get_market_data("")

        with self.assertRaises(MarketDataError):
            get_market_data("   ")

        with self.assertRaises(MarketDataError):
            get_market_data("AAPL; DROP TABLE users;")

    def test_mock_fallback_structure(self):
        """When yfinance is not installed or network is offline, returns structured payload."""
        data = get_market_data("AAPL")
        self.assertEqual(data["ticker"], "AAPL")
        self.assertIn("current_price", data)
        self.assertIn("market_cap", data)
        self.assertIn("trailing_pe", data)
        self.assertIn("fifty_two_week_high", data)
        self.assertIn("fifty_two_week_low", data)
        self.assertIn("currency", data)

    def test_with_mocked_yfinance(self):
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "shortName": "Apple Inc.",
            "currency": "USD",
            "currentPrice": 190.25,
            "marketCap": 2950000000000.0,
            "trailingPE": 31.2,
            "forwardPE": 27.5,
            "priceToBook": 38.4,
            "fiftyTwoWeekHigh": 199.62,
            "fiftyTwoWeekLow": 164.08,
            "beta": 1.05,
            "dividendYield": 0.005,
            "sector": "Technology",
            "industry": "Consumer Electronics",
        }
        mock_yf = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            res = get_market_data("AAPL")
            self.assertEqual(res["ticker"], "AAPL")
            self.assertEqual(res["company_name"], "Apple Inc.")
            self.assertEqual(res["current_price"], 190.25)
            self.assertEqual(res["trailing_pe"], 31.2)
            self.assertEqual(res["sector"], "Technology")

    def test_institutional_bpipe_provider(self):
        with patch.dict("os.environ", {"MARKET_DATA_PROVIDER": "institutional"}):
            res = get_market_data("MSFT")
            self.assertEqual(res["ticker"], "MSFT")
            self.assertEqual(res["provider"], "Bloomberg_BPIPE_Simulator")
            self.assertIn("vwap", res)
            self.assertIn("bid", res)
            self.assertIn("ask", res)
            self.assertIn("primary_exchange", res)
            self.assertEqual(res["primary_exchange"], "XNAS")


if __name__ == "__main__":
    unittest.main()
