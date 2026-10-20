"""Role-Based Access Control (RBAC) & Governance for Asset Management.

Grounded in buy-side investment roles:
- `research_analyst`: Fundamental equity/credit research across 10-K, 10-Q, earnings calls, and internal research memos.
- `portfolio_manager`: Full access including confidential portfolio holdings & allocation strategies; can approve analyst proposals.
- `risk_compliance`: Mandate oversight, tracking error, liquidity limits, restricted list monitoring, and trade approvals.
- `admin`: Infrastructure, pipeline data engineering, and model management.

Document types:
- `10k`: Annual SEC regulatory filings
- `10q`: Quarterly SEC regulatory reports
- `earnings_transcript`: Management discussion & earnings call Q&A transcripts
- `equity_research`: Internal investment memos and broker research
- `portfolio_holdings`: Mandate weights, active exposures, and position sizes
"""

ROLE_PERMISSIONS: dict[str, dict] = {
    # --- Buy-Side Investment Roles ---
    "research_analyst": {
        "allowed_doc_types": ["10k", "10q", "earnings_transcript", "equity_research"],
        "allowed_confidentiality": ["public", "internal"],
        "max_results": 10,
        "requires_hitl_above": 100_000,
    },
    "portfolio_manager": {
        "allowed_doc_types": ["10k", "10q", "earnings_transcript", "equity_research", "portfolio_holdings"],
        "allowed_confidentiality": ["public", "internal", "confidential"],
        "max_results": 15,
        "requires_hitl_above": 1_000_000,
    },
    "risk_compliance": {
        "allowed_doc_types": ["10k", "10q", "earnings_transcript", "equity_research", "portfolio_holdings"],
        "allowed_confidentiality": ["public", "internal", "confidential"],
        "max_results": 20,
        "requires_hitl_above": None,
    },
    "admin": {
        "allowed_doc_types": ["*"],
        "allowed_confidentiality": ["*"],
        "max_results": 25,
        "requires_hitl_above": None,
    },
    # --- Compatibility Aliases ---
    "analyst": {
        "allowed_doc_types": ["10k", "10q", "earnings_transcript", "equity_research"],
        "allowed_confidentiality": ["public", "internal"],
        "max_results": 10,
        "requires_hitl_above": 100_000,
    },
    "finance": {
        "allowed_doc_types": ["10k", "10q", "earnings_transcript", "equity_research"],
        "allowed_confidentiality": ["public", "internal"],
        "max_results": 10,
        "requires_hitl_above": 100_000,
    },
    "hr": {
        "allowed_doc_types": ["equity_research"],
        "allowed_confidentiality": ["public", "internal"],
        "max_results": 5,
        "requires_hitl_above": None,
    },
    "c_level": {
        "allowed_doc_types": ["10k", "10q", "earnings_transcript", "equity_research", "portfolio_holdings"],
        "allowed_confidentiality": ["public", "internal", "confidential"],
        "max_results": 15,
        "requires_hitl_above": 1_000_000,
    },
}


def get_permissions(role: str) -> dict:
    """Return the RBAC permissions for a role (DB-first with static fallback)."""
    from src.services.roles_service import get_permissions as _db_first

    return _db_first(role)


# Buy-side HITL approval hierarchy:
# - Research analysts cannot approve their own high-conviction allocations.
# - Portfolio managers can approve analyst proposals.
# - Risk & Compliance can approve portfolio manager and analyst actions.
# - Admins have universal authority.
CAN_APPROVE_FOR: dict[str, set[str]] = {
    "research_analyst": set(),
    "analyst": set(),
    "finance": set(),
    "hr": set(),
    "portfolio_manager": {"research_analyst", "analyst", "finance"},
    "c_level": {"research_analyst", "analyst", "finance", "hr"},
    "risk_compliance": {"research_analyst", "portfolio_manager", "analyst", "finance"},
    "admin": {"*"},
}


def can_approve(approver_role: str, requester_role: str) -> bool:
    """True iff `approver_role` is authorized to approve a HITL-paused query.
    Self-approval is strictly forbidden."""
    if approver_role == requester_role:
        return False
    allowed = CAN_APPROVE_FOR.get(approver_role, set())
    return "*" in allowed or requester_role in allowed


def approvers_for(requester_role: str) -> list[str]:
    """List of role names authorized to approve `requester_role`'s queries."""
    return sorted(
        role
        for role, allowed in CAN_APPROVE_FOR.items()
        if role != requester_role and ("*" in allowed or requester_role in allowed)
    )
