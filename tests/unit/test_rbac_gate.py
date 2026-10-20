"""Unit tests for the RBAC gate node in src/graph/nodes/rbac_gate.py."""

try:
    import pytest
except ImportError:
    pytest = None

import sys
from unittest.mock import MagicMock

for mod in [
    "langchain_core",
    "langchain_core.messages",
    "langchain_core.runnables",
    "langgraph",
    "langgraph.types",
    "langgraph.graph",
    "langgraph.graph.message",
]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

from src.graph.nodes.rbac_gate import rbac_gate

# ---------------------------------------------------------------------------
# Role-specific permission tests
# ---------------------------------------------------------------------------


class TestRbacGateRoles:
    """Verify that each known role produces the correct allowed_doc_types."""

    def test_research_analyst_role(self):
        state = {"user_role": "research_analyst"}
        result = rbac_gate(state)
        assert result["allowed_doc_types"] == ["10k", "10q", "earnings_transcript", "equity_research"]

    def test_portfolio_manager_role(self):
        state = {"user_role": "portfolio_manager"}
        result = rbac_gate(state)
        assert result["allowed_doc_types"] == [
            "10k",
            "10q",
            "earnings_transcript",
            "equity_research",
            "portfolio_holdings",
        ]

    def test_risk_compliance_role(self):
        state = {"user_role": "risk_compliance"}
        result = rbac_gate(state)
        assert result["allowed_doc_types"] == [
            "10k",
            "10q",
            "earnings_transcript",
            "equity_research",
            "portfolio_holdings",
        ]

    def test_admin_role(self):
        state = {"user_role": "admin"}
        result = rbac_gate(state)
        assert result["allowed_doc_types"] == ["*"]


# ---------------------------------------------------------------------------
# Fallback behavior
# ---------------------------------------------------------------------------


class TestRbacGateFallback:
    """Verify fallback permissions for unknown or missing roles."""

    def test_unknown_role_falls_back_to_analyst(self):
        state = {"user_role": "intern"}
        result = rbac_gate(state)
        assert "10k" in result["allowed_doc_types"]

    def test_empty_string_role_falls_back_to_analyst(self):
        state = {"user_role": ""}
        result = rbac_gate(state)
        assert "10k" in result["allowed_doc_types"]

    def test_missing_role_key_defaults_to_analyst(self):
        """When user_role is absent, rbac_gate defaults to 'analyst'."""
        state = {}
        result = rbac_gate(state)
        assert "10k" in result["allowed_doc_types"]


# ---------------------------------------------------------------------------
# Return shape
# ---------------------------------------------------------------------------


class TestRbacGateReturnShape:
    """Verify the structure of the dict returned by rbac_gate."""

    def test_returns_dict_with_allowed_doc_types_key(self):
        state = {"user_role": "finance"}
        result = rbac_gate(state)
        assert isinstance(result, dict)
        assert "allowed_doc_types" in result

    def test_return_contains_only_allowed_doc_types(self):
        """The node should return only the keys it is responsible for updating."""
        state = {"user_role": "finance"}
        result = rbac_gate(state)
        assert list(result.keys()) == ["allowed_doc_types"]

    def test_allowed_doc_types_is_a_list(self):
        state = {"user_role": "admin"}
        result = rbac_gate(state)
        assert isinstance(result["allowed_doc_types"], list)


if __name__ == "__main__":
    t_roles = TestRbacGateRoles()
    t_roles.test_research_analyst_role()
    t_roles.test_portfolio_manager_role()
    t_roles.test_risk_compliance_role()
    t_roles.test_admin_role()

    t_fb = TestRbacGateFallback()
    t_fb.test_unknown_role_falls_back_to_analyst()
    t_fb.test_empty_string_role_falls_back_to_analyst()
    t_fb.test_missing_role_key_defaults_to_analyst()

    t_shape = TestRbacGateReturnShape()
    t_shape.test_returns_dict_with_allowed_doc_types_key()
    t_shape.test_return_contains_only_allowed_doc_types()
    t_shape.test_allowed_doc_types_is_a_list()
    print("All RBAC gate tests passed!")
