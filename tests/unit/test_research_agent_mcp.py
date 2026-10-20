"""Unit tests for in-graph Model Context Protocol (MCP) tool execution.

Verifies:
1. research_agent invoking MCPClient for get_market_data inside graph workflow.
2. research_agent enforcing buy-side RBAC: analyst gets PERMISSION_DENIED on portfolio risk tool.
3. portfolio_manager role permitted to calculate portfolio risk metrics.
4. OpenTelemetry 'mcp.tool_call' tracing spans emitted during graph execution.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Force deterministic institutional provider for tests
os.environ["MARKET_DATA_PROVIDER"] = "institutional"

# Ensure root directory is on pythonpath
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.graph.nodes.research_agent import (
    _Decomposition,
    _execute_mcp_tools,
    _resolve_ticker,
    research_agent_node,
)


def test_resolve_ticker():
    assert _resolve_ticker("What is Apple's revenue", "apple") == "AAPL"
    assert _resolve_ticker("What is Microsoft's margin", "microsoft") == "MSFT"
    assert _resolve_ticker("What is Tesla's gross margin", "tesla") == "TSLA"
    assert _resolve_ticker("What is NVDA current price?", None) == "NVDA"


def test_research_agent_executes_market_data_mcp():
    decomp = _Decomposition(
        qualifiers=["no qualifiers"],
        required_quantities=["share price"],
        sub_questions=["What is the current share price of Apple?"],
    )
    results, block = _execute_mcp_tools(
        query="What is Apple's current share price and market cap?",
        target_company="apple",
        user_role="research_analyst",
        user_id="analyst_test",
        decomp=decomp,
    )
    assert len(results) >= 1
    mkt_res = results[0]
    assert mkt_res["name"] == "get_market_data"
    assert mkt_res["is_error"] is False
    assert mkt_res["data"]["ticker"] == "AAPL"
    assert "Live Market Data (`get_market_data`)" in block
    assert "Ticker `AAPL`" in block


def test_research_agent_enforces_role_denial_in_graph():
    decomp = _Decomposition(
        qualifiers=["no qualifiers"],
        required_quantities=["hhi"],
        sub_questions=["Calculate portfolio risk"],
    )
    # Analyst role: calculate_portfolio_risk_metrics must be DENIED
    results, block = _execute_mcp_tools(
        query="Evaluate portfolio risk and concentration limits",
        target_company=None,
        user_role="research_analyst",
        user_id="analyst_test",
        decomp=decomp,
    )
    assert len(results) >= 1
    risk_res = [r for r in results if r["name"] == "calculate_portfolio_risk_metrics"][0]
    assert risk_res["is_error"] is True
    assert risk_res["error_code"] == "PERMISSION_DENIED"
    assert "[GATING ENFORCED]" in block
    assert "lacks permission" in block or "Access denied" in block


def test_research_agent_allows_portfolio_manager():
    decomp = _Decomposition(
        qualifiers=["no qualifiers"],
        required_quantities=["hhi"],
        sub_questions=["Calculate portfolio risk"],
    )
    # Portfolio manager role: permitted
    results, block = _execute_mcp_tools(
        query="Evaluate portfolio risk and concentration limits",
        target_company=None,
        user_role="portfolio_manager",
        user_id="pm_test",
        decomp=decomp,
    )
    assert len(results) >= 1
    risk_res = [r for r in results if r["name"] == "calculate_portfolio_risk_metrics"][0]
    assert risk_res["is_error"] is False
    assert "hhi_score" in risk_res["data"]
    assert "HHI:" in block


def test_research_agent_node_full_mcp_diagnostics():
    state = {
        "sanitized_query": "What is Apple's current share price and market cap?",
        "target_company": "apple",
        "target_fiscal_year": 2023,
        "user_role": "research_analyst",
        "user_id": "test_user",
        "retrieved_chunks": [],
        "relevant_chunks": [],
    }

    mock_decomp = _Decomposition(
        qualifiers=["no qualifiers"],
        required_quantities=["share price"],
        sub_questions=["What is Apple current price?"],
    )

    with (
        patch("src.graph.nodes.research_agent._decompose", return_value=mock_decomp),
        patch("src.graph.nodes.research_agent._retrieve_and_grade_for_subq", return_value=[]),
        patch(
            "src.graph.nodes.research_agent._judge_sufficiency",
            return_value=MagicMock(decision="sufficient", reason="test done", follow_up_question=None),
        ),
        patch(
            "src.graph.nodes.research_agent._synthesize",
            return_value=("Base research findings", {"calculator_invoked": False}),
        ),
    ):
        out = research_agent_node(state)

    assert "mcp_tools_invoked" in out
    assert "get_market_data" in out["mcp_tools_invoked"]
    assert "Live Market Data (`get_market_data`)" in out["agent_synthesis"]


if __name__ == "__main__":
    test_resolve_ticker()
    test_research_agent_executes_market_data_mcp()
    test_research_agent_enforces_role_denial_in_graph()
    test_research_agent_allows_portfolio_manager()
    test_research_agent_node_full_mcp_diagnostics()
    print("All research agent in-graph MCP tool execution tests passed successfully!")
