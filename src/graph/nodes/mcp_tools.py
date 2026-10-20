"""Dedicated MCP Tools node for LangGraph orchestration.

Executes when the router classifies intent as 'market_data' or 'portfolio_risk'.
Invokes tools via `src/tools/mcp_client.py:MCPClient` with the session's `user_role`,
recording the `mcp.tool_call` span in the active trace, and populates `relevant_chunks`
and `agent_synthesis` for the downstream `generator` node.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.models.state import RAGState
from src.tools.mcp_client import MCPClient

logger = logging.getLogger(__name__)

SAMPLE_PORTFOLIO_PATH = Path(__file__).parent.parent.parent.parent / "data" / "sample_portfolio.json"

COMPANY_TICKER_MAP: dict[str, str] = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "tesla": "TSLA",
    "amazon": "AMZN",
    "google": "GOOGL",
    "meta": "META",
    "nvidia": "NVDA",
    "jpmorgan": "JPM",
    "berkshire": "BRK-B",
    "3m": "MMM",
    "boeing": "BA",
    "coca_cola": "KO",
    "costco": "COST",
    "netflix": "NFLX",
    "nike": "NKE",
    "walmart": "WMT",
}


def _resolve_ticker(query: str, target_company: str | None) -> str:
    """Resolve equity ticker from target company slug or query tokens."""
    if target_company:
        norm = target_company.strip().lower()
        if norm in COMPANY_TICKER_MAP:
            return COMPANY_TICKER_MAP[norm]
    import re

    tickers = re.findall(r"\b[A-Z]{2,5}\b", query)
    if tickers:
        ignore = {
            "SEC",
            "GAAP",
            "CAGR",
            "EBIT",
            "EBITDA",
            "EPS",
            "USD",
            "ROE",
            "ROA",
            "FY",
            "Q1",
            "Q2",
            "Q3",
            "Q4",
            "HHI",
            "PE",
            "PB",
            "FCF",
            "EV",
            "RAG",
        }
        for t in tickers:
            if t not in ignore:
                return t
    return "AAPL"


def _load_sample_holdings() -> dict[str, float]:
    """Load sample institutional holdings from data/sample_portfolio.json."""
    if SAMPLE_PORTFOLIO_PATH.exists():
        try:
            with open(SAMPLE_PORTFOLIO_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("holdings", {})
        except Exception as e:
            logger.warning("Could not read sample portfolio file: %s", e)
    return {"AAPL": 14.5, "MSFT": 12.0, "NVDA": 9.5, "AMZN": 7.5, "GOOGL": 6.5}


def mcp_tools_node(state: RAGState) -> dict:
    """Execute MCP tools directly in the graph based on router intent."""
    query = state.get("sanitized_query", "")
    target_company = state.get("target_company")
    user_role = state.get("user_role", "analyst")
    user_id = state.get("user_id", "system")
    intent = state.get("query_intent", "market_data")

    client = MCPClient(user_role=user_role, user_id=user_id)
    tools_invoked = []
    tool_results = []
    role_denials = []
    content_lines = []

    if intent == "market_data" or any(kw in query.lower() for kw in ("price", "beta", "pe", "52-week", "market cap")):
        ticker = _resolve_ticker(query, target_company)
        tools_invoked.append("get_market_data")
        res = client.call_tool("get_market_data", {"ticker": ticker})
        tool_results.append(res)
        if res.get("is_error"):
            content_lines.append(f"Market Data Error for {ticker}: {res.get('text')}")
        else:
            d = res["data"]
            market_cap = d.get("market_cap")
            market_cap_text = f"${market_cap / 1e9:.2f}B" if isinstance(market_cap, (int, float)) else "N/A"
            content_lines.append(
                f"Ticker: {d.get('ticker')} | Live Price: ${d.get('current_price', 'N/A')} | "
                f"Market Cap: {market_cap_text} | Trailing P/E: {d.get('trailing_pe', 'N/A')} | "
                f"Forward P/E: {d.get('forward_pe', 'N/A')} | Beta: {d.get('beta', 'N/A')} | "
                f"52-Week Range: ${d.get('fifty_two_week_low', 'N/A')} - ${d.get('fifty_two_week_high', 'N/A')}"
            )

    if intent == "portfolio_risk" or any(kw in query.lower() for kw in ("portfolio", "risk", "concentration", "hhi")):
        tools_invoked.append("calculate_portfolio_risk_metrics")
        holdings = _load_sample_holdings()
        res = client.call_tool("calculate_portfolio_risk_metrics", {"holdings": holdings})
        tool_results.append(res)
        if res.get("is_error"):
            role_denials.append("calculate_portfolio_risk_metrics")
            content_lines.append(
                f"[GATING ENFORCED] Portfolio Risk Calculation: {res.get('text')} (Code: {res.get('error_code')})"
            )
        else:
            d = res["data"]
            content_lines.append(
                f"Portfolio Risk Metrics (Sample Institutional Portfolio): HHI Score: {d.get('hhi_score')} "
                f"(Risk Tier: {d.get('concentration_risk')}) | Top-1 Weight: {d.get('top_1_weight_pct')}% | "
                f"Top-3 Weight: {d.get('top_3_weight_pct')}% | Top-5 Weight: {d.get('top_5_weight_pct')}% | "
                f"Effective Holdings Count: {d.get('effective_positions')}"
            )

    findings_text = "\n".join(content_lines)
    synthesis = f"## Real-Time & Deterministic MCP Findings:\n{findings_text}"

    # Build grounded chunk for generator
    virtual_chunk = {
        "content": findings_text,
        "metadata": {
            "source_file": "MCP Tool Service (Live / Deterministic)",
            "page_number": 1,
            "section_header": "MCP Tool Ingress",
            "doc_type": "market_data",
            "confidentiality": "public",
        },
        "score": 1.0,
    }

    return {
        "agent_synthesis": synthesis,
        "relevant_chunks": [virtual_chunk],
        "mcp_tools_invoked": tools_invoked,
        "mcp_tool_results": tool_results,
        "mcp_role_denials": role_denials,
    }
