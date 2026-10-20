"""Unit tests for Asset Management RBAC & compliance governance."""

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

from src.config.rbac_config import can_approve, get_permissions
from src.graph.nodes.hitl_gate import _check_compliance_flags


def test_research_analyst_permissions():
    perms = get_permissions("research_analyst")
    assert "10k" in perms["allowed_doc_types"]
    assert "10q" in perms["allowed_doc_types"]
    assert "earnings_transcript" in perms["allowed_doc_types"]
    assert "equity_research" in perms["allowed_doc_types"]
    assert "portfolio_holdings" not in perms["allowed_doc_types"]
    assert "public" in perms["allowed_confidentiality"]
    assert "internal" in perms["allowed_confidentiality"]
    assert "confidential" not in perms["allowed_confidentiality"]
    assert perms["requires_hitl_above"] == 100_000


def test_portfolio_manager_permissions():
    perms = get_permissions("portfolio_manager")
    assert "portfolio_holdings" in perms["allowed_doc_types"]
    assert "confidential" in perms["allowed_confidentiality"]
    assert perms["requires_hitl_above"] == 1_000_000


def test_risk_compliance_permissions():
    perms = get_permissions("risk_compliance")
    assert "portfolio_holdings" in perms["allowed_doc_types"]
    assert "confidential" in perms["allowed_confidentiality"]
    assert perms["requires_hitl_above"] is None


def test_approval_hierarchy():
    # Analysts cannot approve anyone
    assert not can_approve("research_analyst", "research_analyst")
    assert not can_approve("research_analyst", "portfolio_manager")

    # Portfolio Manager can approve analyst proposals
    assert can_approve("portfolio_manager", "research_analyst")
    # PM cannot self-approve
    assert not can_approve("portfolio_manager", "portfolio_manager")

    # Risk & Compliance can approve PM and Analyst actions
    assert can_approve("risk_compliance", "portfolio_manager")
    assert can_approve("risk_compliance", "research_analyst")

    # Admin can approve all
    assert can_approve("admin", "research_analyst")
    assert can_approve("admin", "portfolio_manager")


def test_compliance_flags_detection():
    clean_query = "What was Apple's gross margin in fiscal 2023?"
    assert _check_compliance_flags(clean_query) == []

    restricted_query = "Can we allocate $5M to security XYZ on the restricted list?"
    flags = _check_compliance_flags(restricted_query)
    assert len(flags) > 0
    assert "restricted list" in flags[0].lower()

    mnpi_query = "Analyst notes suggest potential MNPI regarding merger talks."
    flags_mnpi = _check_compliance_flags(mnpi_query)
    assert len(flags_mnpi) > 0
    assert "mnpi" in flags_mnpi[0].lower()


if __name__ == "__main__":
    test_research_analyst_permissions()
    test_portfolio_manager_permissions()
    test_risk_compliance_permissions()
    test_approval_hierarchy()
    test_compliance_flags_detection()
    print("All asset management RBAC unit tests passed successfully!")
