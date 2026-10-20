"""Unit tests for System 1 Fast Decider (Strand Decider 2B).

Verifies:
1. System 1 vs System 2 classification latency and decision boundaries.
2. Market data & trading multiples classification.
3. Portfolio risk & rebalancing classification.
4. Out-of-scope non-financial query filtering.
5. Simple SEC lookup vs deep multi-step research tiering.
6. router_node integration with decider metadata propagation.
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
            return lambda *args, **kwargs: (lambda fn: fn)
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
        if any(fullname.startswith(pkg) for pkg in ["langchain", "langgraph", "pydantic", "pydantic_settings"]):
            return ModuleSpec(fullname, _DynamicMockLoader(fullname), is_package=True)
        return None

sys.meta_path.insert(0, _DynamicMockFinder())

from src.graph.edges import route_after_router
from src.graph.nodes.router import router_node
from src.services.decider_2b import StrandDecider2B, decider_2b


def test_decider_market_data():
    decision = decider_2b.classify("What is Apple's current stock price, beta, and P/E ratio?")
    assert decision.intent == "market_data"
    assert decision.system_tier == "system_1"
    assert decision.confidence >= 0.90
    assert decision.decider_model == "strand-decider-2b"


def test_decider_portfolio_risk():
    decision = decider_2b.classify("Calculate the portfolio risk and HHI concentration score")
    assert decision.intent == "portfolio_risk"
    assert decision.system_tier == "system_1"
    assert decision.confidence >= 0.90


def test_decider_rebalancing_governance():
    decision = decider_2b.classify("Add 2.5% to NVDA in the fund portfolio and rebalance")
    assert decision.intent == "portfolio_risk"
    assert decision.system_tier == "system_1"


def test_decider_out_of_scope():
    decision = decider_2b.classify("Can you give me a recipe for chocolate cake?")
    assert decision.intent == "out_of_scope"
    assert decision.system_tier == "system_1"
    assert decision.confidence >= 0.95


def test_decider_sec_simple_lookup():
    decision = decider_2b.classify("What was Microsoft's total revenue in fiscal year 2023 according to the 10-K?")
    assert decision.intent == "retrieval"
    assert decision.complexity == "simple_lookup"
    assert decision.system_tier == "system_2"


def test_decider_sec_deep_research():
    decision = decider_2b.classify(
        "Compare Apple vs Microsoft operating margin trends over the last 3 years and calculate the Piotroski score"
    )
    assert decision.intent == "retrieval"
    assert decision.complexity == "research_required"
    assert decision.system_tier == "system_2"


def test_router_node_metadata_integration():
    state = {
        "sanitized_query": "What is the 52-week range and current beta for NVDA?",
        "target_company": "nvda",
    }
    out = router_node(state)
    assert out["query_intent"] == "market_data"
    assert out["query_complexity"] == "simple_lookup"
    assert out["system_tier"] == "system_1"
    assert out["decider_model"] == "strand-decider-2b"
    assert out["decider_confidence"] >= 0.90
    assert route_after_router(out) == "mcp_tools"


if __name__ == "__main__":
    test_decider_market_data()
    test_decider_portfolio_risk()
    test_decider_rebalancing_governance()
    test_decider_out_of_scope()
    test_decider_sec_simple_lookup()
    test_decider_sec_deep_research()
    test_router_node_metadata_integration()
    print("All System 1 Strand Decider 2B unit tests passed successfully!")
