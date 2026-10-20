import logging
import re
from typing import Any

try:
    from langchain_core.runnables import RunnableConfig
    from langgraph.types import interrupt
except ImportError:
    RunnableConfig = Any  # type: ignore

    def interrupt(data):  # type: ignore
        return {"decision": "approved"}


from src.config.rbac_config import get_permissions
from src.models.state import RAGState

logger = logging.getLogger(__name__)

# Regex to find dollar amounts in text
AMOUNT_PATTERN = re.compile(
    r"\$[\d,]+(?:\.\d{1,2})?(?:\s*(?:million|billion|trillion|thousand|M|B|T|k)\b)?", re.IGNORECASE
)

# No real figure in this corpus reaches $10T; larger parses come from a
# malformed generated draft (stray zero-groups) and are treated as noise so
# the gate never reports a nonsensical amount.
_SANE_MAX_AMOUNT = 1e13


def _extract_max_amount(text: str) -> float:
    """Extract the largest dollar amount mentioned in the text."""
    matches = AMOUNT_PATTERN.findall(text)
    if not matches:
        return 0.0

    max_amount = 0.0
    for match in matches:
        cleaned = match.replace("$", "").replace(",", "").strip()
        multiplier = 1.0
        for suffix, mult in [
            ("trillion", 1e12),
            ("billion", 1e9),
            ("million", 1e6),
            ("thousand", 1e3),
            ("T", 1e12),
            ("B", 1e9),
            ("M", 1e6),
            ("k", 1e3),
        ]:
            if suffix in cleaned:
                cleaned = cleaned.replace(suffix, "").strip()
                multiplier = mult
                break
        try:
            amount = float(cleaned) * multiplier
        except ValueError:
            continue
        if amount > _SANE_MAX_AMOUNT:
            continue
        max_amount = max(max_amount, amount)
    return max_amount


def _get_compliance_patterns() -> list[re.Pattern]:
    """Retrieve compliance restricted patterns dynamically from settings."""
    try:
        from src.config.settings import settings

        keywords = getattr(settings, "COMPLIANCE_RESTRICTED_KEYWORDS", None)
    except Exception:
        keywords = None
    if not keywords:
        keywords = [
            "restricted list",
            "restricted stock",
            "mnpi",
            "material non-public",
            "insider trading",
            "blackout window",
            "blackout period",
            "embargoed research",
        ]
    escaped = [re.escape(k.strip()) for k in keywords if k.strip()]
    if not escaped:
        return []
    pattern_str = r"\b(?:" + "|".join(escaped) + r")\b"
    return [re.compile(pattern_str, re.IGNORECASE)]


def _check_compliance_flags(text: str) -> list[str]:
    """Check text for regulatory compliance or trade-restriction triggers."""
    flags = []
    for pat in _get_compliance_patterns():
        match = pat.search(text)
        if match:
            flags.append(match.group(0))
    return flags


def hitl_gate_node(state: RAGState, config: RunnableConfig | None = None) -> dict:
    """Check if human approval is needed based on answer content and user role.

    Gating triggers:
    1. Financial threshold: dollar amounts in query/answer exceeding role's threshold
    2. Compliance / MNPI: detection of restricted list, material non-public information,
       or trading blackout references.

    Uses LangGraph interrupt() to pause the graph when approval is required.
    """
    answer = state.get("generated_answer", "")
    query = state.get("sanitized_query", "")
    user_role = state.get("user_role", "analyst")

    permissions = get_permissions(user_role)
    threshold = permissions.get("requires_hitl_above")

    compliance_flags = _check_compliance_flags(query) + _check_compliance_flags(answer)
    max_amount = max(_extract_max_amount(answer), _extract_max_amount(query))

    triggered_by_amount = threshold is not None and max_amount > threshold
    triggered_by_compliance = bool(compliance_flags) and user_role not in ("admin", "risk_compliance")

    if not triggered_by_amount and not triggered_by_compliance:
        return {"requires_human_approval": False, "human_decision": None}

    reason_parts = []
    if triggered_by_amount:
        reason_parts.append(
            f"Allocation references ${max_amount:,.0f} (exceeds ${threshold:,} threshold for '{user_role}')"
        )
    if triggered_by_compliance:
        reason_parts.append(f"Compliance flag detected: {', '.join(set(compliance_flags))}")
    gate_reason = " | ".join(reason_parts)

    logger.info(f"HITL triggered for role={user_role}: {gate_reason}")

    # Check if HITL persistence is available (checkpointer configured)
    hitl_enabled = (config or {}).get("metadata", {}).get("hitl_enabled", False)
    if not hitl_enabled:
        logger.warning("HITL triggered but no checkpointer available — auto-approving")
        return {"requires_human_approval": True, "human_decision": "approved"}

    from datetime import datetime, timezone

    submitted_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    decision = interrupt(
        {
            "type": "compliance_approval_required" if triggered_by_compliance else "approval_required",
            "reason": gate_reason,
            "max_amount": max_amount if triggered_by_amount else None,
            "compliance_flags": list(set(compliance_flags)),
            "threshold": threshold,
            "submitted_at": submitted_at,
        }
    )

    # Execution resumes here after the human responds. NOTE: this whole
    # function runs again from the top on resume, so the local
    # `submitted_at` was just recomputed (it's now() of the resume call,
    # NOT the original pause time). The API forwards the ORIGINAL value
    # through the resume dict's "submitted_at" field to preserve the
    # audit trail. Fall back to the local value only if the approver used
    # the legacy bare-string resume.
    logger.info(f"HITL decision received: {decision}")
    if isinstance(decision, dict):
        human_decision = "approved" if decision.get("decision") == "approved" else "rejected"
        decision_at = decision.get("decided_at")
        decision_by = decision.get("decided_by")
        decision_by_role = decision.get("decided_by_role")
        decision_reason = decision.get("reason") or ""
        original_submitted_at = decision.get("submitted_at") or submitted_at
    else:
        human_decision = "approved" if decision == "approved" else "rejected"
        decision_at = None
        decision_by = None
        decision_by_role = None
        decision_reason = ""
        original_submitted_at = submitted_at

    return {
        "requires_human_approval": True,
        "human_decision": human_decision,
        "human_decision_at": decision_at,
        "human_decision_by": decision_by,
        "human_decision_by_role": decision_by_role,
        "human_decision_reason": decision_reason,
        "hitl_submitted_at": original_submitted_at,
    }
