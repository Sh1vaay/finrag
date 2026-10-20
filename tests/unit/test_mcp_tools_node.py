"""Unit tests for dedicated mcp_tools_node and direct router intent.

Verifies:
1. Direct routing: router classifies market data & portfolio queries to mcp_tools.
2. In-graph execution of get_market_data before generator.
3. In-graph execution of calculate_portfolio_risk_metrics with sample_portfolio.json.
4. Per-role enforcement (analyst denied vs portfolio manager allowed).
"""

from __future__ import annotations

import sys
import types
from importlib.machinery import ModuleSpec
from pathlib import Path
from unittest.mock import MagicMock

# Ensure root directory is on pythonpath
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class _FakeBaseModel:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _DynamicMockModule(types.ModuleType):
    def __getattr__(self, name):
        if name in ("BaseModel", "BaseSettings"):
            return _FakeBaseModel
        if name == "model_validator":
            return lambda *args, **kwargs: lambda fn: fn
        if name == "Field":
            return lambda *args, **kwargs: kwargs.get("default")
        return MagicMock()


class _DynamicMockLoader:
    def __init__(self, name):
        self.name = name

    def create_module(self, spec):
        mod = _DynamicMockModule(spec.name)
        mod.__path__ = []
        return mod

    def exec_module(self, module):
        pass


class _DynamicMockFinder:
    def find_spec(self, fullname, path, target=None):
        if any(fullname.startswith(pkg) for pkg in ["langchain", "langgraph", "pydantic"]):
            return ModuleSpec(fullname, _DynamicMockLoader(fullname), is_package=True)
        return None


sys.meta_path.insert(0, _DynamicMockFinder())

from src.graph.edges import route_after_router
from src.graph.nodes.mcp_tools import mcp_tools_node
from src.graph.nodes.router import router_node


def test_router_fast_path_market_data():
    state = {
        "sanitized_query": "What is Apple's current stock price, beta, and P/E ratio?",
        "target_company": "apple",
    }
    decision = router_node(state)
    assert decision["query_intent"] == "market_data"
    assert route_after_router(decision) == "mcp_tools"


def test_router_fast_path_portfolio_risk():
    state = {
        "sanitized_query": "Calculate the portfolio risk and HHI concentration score",
        "target_company": None,
    }
    decision = router_node(state)
    assert decision["query_intent"] == "portfolio_risk"
    assert route_after_router(decision) == "mcp_tools"


def test_mcp_tools_node_market_data():
    state = {
        "sanitized_query": "What is Apple's current share price and beta?",
        "target_company": "apple",
        "user_role": "research_analyst",
        "user_id": "test_analyst",
        "query_intent": "market_data",
    }
    out = mcp_tools_node(state)
    assert "get_market_data" in out["mcp_tools_invoked"]
    assert len(out["relevant_chunks"]) >= 1
    assert "Ticker: AAPL" in out["agent_synthesis"]
    assert "Live Price:" in out["agent_synthesis"]


def test_mcp_tools_node_portfolio_denial():
    state = {
        "sanitized_query": "Evaluate portfolio risk and HHI concentration",
        "target_company": None,
        "user_role": "research_analyst",
        "user_id": "test_analyst",
        "query_intent": "portfolio_risk",
    }
    out = mcp_tools_node(state)
    assert "calculate_portfolio_risk_metrics" in out["mcp_tools_invoked"]
    assert "calculate_portfolio_risk_metrics" in out["mcp_role_denials"]
    assert "[GATING ENFORCED]" in out["agent_synthesis"]
    assert "PERMISSION_DENIED" in out["agent_synthesis"]


def test_mcp_tools_node_portfolio_pm_allowed():
    state = {
        "sanitized_query": "Evaluate portfolio risk and HHI concentration",
        "target_company": None,
        "user_role": "portfolio_manager",
        "user_id": "test_pm",
        "query_intent": "portfolio_risk",
    }
    out = mcp_tools_node(state)
    assert "calculate_portfolio_risk_metrics" in out["mcp_tools_invoked"]
    assert len(out["mcp_role_denials"]) == 0
    assert "HHI Score:" in out["agent_synthesis"]
    assert "Top-1 Weight:" in out["agent_synthesis"]


if __name__ == "__main__":
    test_router_fast_path_market_data()
    test_router_fast_path_portfolio_risk()
    test_mcp_tools_node_market_data()
    test_mcp_tools_node_portfolio_denial()
    test_mcp_tools_node_portfolio_pm_allowed()
    print("All dedicated MCP tools node and router intent tests passed successfully!")
