"""Unit tests for Model Context Protocol (MCP) server & tool execution.

Tests protocol initialization, tool listing, execution, role-based denial vs allow,
and market data tools under MCP SDK v2 standards.
"""

try:
    import pytest
except ImportError:

    class _MockPytest:
        pass

    pytest = _MockPytest()

import json
import sys
from pathlib import Path

# Ensure root directory is on pythonpath
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.tools.mcp_server import handle_mcp_jsonrpc


def test_mcp_initialize():
    req = {
        "jsonrpc": "2.0",
        "id": "req-1",
        "method": "initialize",
        "params": {},
    }
    resp = handle_mcp_jsonrpc(req)
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == "req-1"
    assert "result" in resp
    assert resp["result"]["serverInfo"]["name"] == "finrag-investment-mcp"
    assert resp["result"]["protocolVersion"] == "2024-11-05"
    assert resp["result"]["capabilities"]["tools"]["listChanged"] is False


def test_mcp_tools_list():
    req = {
        "jsonrpc": "2.0",
        "id": "req-2",
        "method": "tools/list",
    }
    # Analyst sees research tools but NOT portfolio risk
    resp = handle_mcp_jsonrpc(req, user_role="research_analyst")
    assert resp["jsonrpc"] == "2.0"
    assert "result" in resp
    tools = resp["result"]["tools"]
    tool_names = [t["name"] for t in tools]
    assert "calculate_valuation_multiples" in tool_names
    assert "calculate_profitability_metrics" in tool_names
    assert "calculate_growth_metrics" in tool_names
    assert "get_market_data" in tool_names
    assert "evaluate_arithmetic" in tool_names
    assert "calculate_factor_exposures" in tool_names
    assert "get_sec_filings" in tool_names
    assert "calculate_portfolio_risk_metrics" not in tool_names
    assert "calculate_portfolio_var_cvar" not in tool_names
    assert "optimize_portfolio_weights" not in tool_names

    # Portfolio Manager sees all tools including portfolio risk and optimization
    resp_pm = handle_mcp_jsonrpc(req, user_role="portfolio_manager")
    pm_tools = [t["name"] for t in resp_pm["result"]["tools"]]
    assert "calculate_portfolio_risk_metrics" in pm_tools
    assert "calculate_portfolio_var_cvar" in pm_tools
    assert "optimize_portfolio_weights" in pm_tools


def test_mcp_tools_call_valuation():
    req = {
        "jsonrpc": "2.0",
        "id": "req-3",
        "method": "tools/call",
        "params": {
            "name": "calculate_valuation_multiples",
            "arguments": {
                "share_price": 100.0,
                "earnings_per_share": 5.0,
            },
        },
    }
    resp = handle_mcp_jsonrpc(req, user_role="research_analyst", user_id="analyst_1")
    assert resp["jsonrpc"] == "2.0"
    assert "result" in resp
    content = resp["result"]["content"][0]["text"]
    data = json.loads(content)
    assert data["share_price"] == 100.0
    assert data["pe_ratio"] == 20.0
    assert resp["result"]["isError"] is False


def test_mcp_tools_call_market_data():
    req = {
        "jsonrpc": "2.0",
        "id": "req-mkt",
        "method": "tools/call",
        "params": {
            "name": "get_market_data",
            "arguments": {"ticker": "AAPL"},
        },
    }
    resp = handle_mcp_jsonrpc(req, user_role="research_analyst", user_id="analyst_1")
    assert resp["jsonrpc"] == "2.0"
    assert "result" in resp
    assert resp["result"]["isError"] is False
    content = json.loads(resp["result"]["content"][0]["text"])
    assert content["ticker"] == "AAPL"
    assert "current_price" in content
    assert "market_cap" in content


def test_mcp_role_allowed_vs_denied():
    """Verify an MCP client calling calculate_portfolio_risk_metrics gets denied as analyst but allowed as PM."""
    holdings = {"AAPL": 400000.0, "MSFT": 300000.0, "NVDA": 300000.0}
    call_req = {
        "jsonrpc": "2.0",
        "id": "req-risk",
        "method": "tools/call",
        "params": {
            "name": "calculate_portfolio_risk_metrics",
            "arguments": {"holdings": holdings},
        },
    }

    # Case 1: Denied as research_analyst
    denied_resp = handle_mcp_jsonrpc(call_req, user_role="research_analyst")
    assert denied_resp["result"]["isError"] is True
    assert denied_resp["result"].get("error_code") == "PERMISSION_DENIED"
    assert "Access denied" in denied_resp["result"]["content"][0]["text"]

    # Case 2: Allowed as portfolio_manager
    allowed_resp = handle_mcp_jsonrpc(call_req, user_role="portfolio_manager")
    assert allowed_resp["result"]["isError"] is False
    result_data = json.loads(allowed_resp["result"]["content"][0]["text"])
    assert result_data["total_positions"] == 3
    assert result_data["hhi_score"] > 0


def test_mcp_tools_call_arithmetic():
    req = {
        "jsonrpc": "2.0",
        "id": "req-4",
        "method": "tools/call",
        "params": {
            "name": "evaluate_arithmetic",
            "arguments": {
                "expression": "(383285 - 394328) / 394328 * 100",
            },
        },
    }
    resp = handle_mcp_jsonrpc(req)
    assert resp["jsonrpc"] == "2.0"
    assert "result" in resp
    content = resp["result"]["content"][0]["text"]
    data = json.loads(content)
    assert round(data["result"], 2) == -2.80


def test_mcp_tools_call_var():
    req = {
        "jsonrpc": "2.0",
        "id": "req-var",
        "method": "tools/call",
        "params": {
            "name": "calculate_portfolio_var_cvar",
            "arguments": {
                "holdings": {"AAPL": 0.5, "MSFT": 0.5},
                "confidence_level": 0.95,
            },
        },
    }
    # Denied for analyst
    resp_analyst = handle_mcp_jsonrpc(req, user_role="research_analyst")
    assert resp_analyst["result"]["isError"] is True
    assert resp_analyst["result"]["error_code"] == "PERMISSION_DENIED"

    # Allowed for portfolio manager
    resp_pm = handle_mcp_jsonrpc(req, user_role="portfolio_manager")
    assert resp_pm["result"]["isError"] is False
    res = json.loads(resp_pm["result"]["content"][0]["text"])
    assert res["confidence_level"] == "95%"
    assert "parametric_var_pct" in res
    assert "conditional_var_expected_shortfall_pct" in res


def test_mcp_tools_call_factor():
    req = {
        "jsonrpc": "2.0",
        "id": "req-fact",
        "method": "tools/call",
        "params": {
            "name": "calculate_factor_exposures",
            "arguments": {"ticker": "NVDA"},
        },
    }
    resp = handle_mcp_jsonrpc(req, user_role="research_analyst")
    assert resp["result"]["isError"] is False
    res = json.loads(resp["result"]["content"][0]["text"])
    assert res["ticker"] == "NVDA"
    assert res["market_beta"] > 1.0
    assert "systematic_risk_pct" in res


def test_mcp_tools_call_not_found():
    req = {
        "jsonrpc": "2.0",
        "id": "req-5",
        "method": "tools/call",
        "params": {
            "name": "non_existent_tool",
            "arguments": {},
        },
    }
    resp = handle_mcp_jsonrpc(req)
    assert "error" in resp
    assert resp["error"]["code"] == -32601
    assert "not found" in resp["error"]["message"]


def test_mcp_invalid_jsonrpc():
    req = {"id": "1", "method": "tools/list"}  # missing jsonrpc="2.0"
    resp = handle_mcp_jsonrpc(req)
    assert "error" in resp
    assert resp["error"]["code"] == -32600


def test_mcp_client_invocation():
    from src.tools.mcp_client import MCPClient

    analyst_client = MCPClient(user_role="research_analyst", user_id="analyst_test")
    tools = analyst_client.list_tools()
    tool_names = [t["name"] for t in tools]
    assert "get_market_data" in tool_names
    assert "calculate_portfolio_risk_metrics" not in tool_names

    # Market data execution
    mkt_res = analyst_client.call_tool("get_market_data", {"ticker": "AAPL"})
    assert mkt_res["is_error"] is False
    assert mkt_res["data"]["ticker"] == "AAPL"

    # Role denial inside client
    denied_res = analyst_client.call_tool(
        "calculate_portfolio_risk_metrics",
        {"holdings": {"AAPL": 0.6, "MSFT": 0.4}},
    )
    assert denied_res["is_error"] is True
    assert denied_res["error_code"] == "PERMISSION_DENIED"
    assert "Access denied" in denied_res["text"]

    # Authorized PM client
    pm_client = MCPClient(user_role="portfolio_manager", user_id="pm_test")
    pm_tools = pm_client.list_tools()
    assert "calculate_portfolio_risk_metrics" in [t["name"] for t in pm_tools]
    pm_res = pm_client.call_tool(
        "calculate_portfolio_risk_metrics",
        {"holdings": {"AAPL": 0.6, "MSFT": 0.4}},
    )
    assert pm_res["is_error"] is False
    assert "hhi_score" in pm_res["data"]


if __name__ == "__main__":
    test_mcp_initialize()
    test_mcp_tools_list()
    test_mcp_tools_call_valuation()
    test_mcp_tools_call_market_data()
    test_mcp_role_allowed_vs_denied()
    test_mcp_tools_call_arithmetic()
    test_mcp_tools_call_var()
    test_mcp_tools_call_factor()
    test_mcp_tools_call_not_found()
    test_mcp_invalid_jsonrpc()
    test_mcp_client_invocation()
    print("All MCP server unit tests passed successfully!")
