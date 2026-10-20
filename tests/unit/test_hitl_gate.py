import sys
from pathlib import Path
from unittest.mock import MagicMock

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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

from src.graph.nodes.hitl_gate import _extract_max_amount


def test_extracts_plain_amount():
    assert _extract_max_amount("Approve a $500,000 reimbursement") == 500_000.0


def test_extracts_suffix_amounts():
    assert _extract_max_amount("revenue of $383.3 billion") == 383_300_000_000.0
    assert _extract_max_amount("a $5 million payment") == 5_000_000.0


def test_picks_largest_of_several():
    assert _extract_max_amount("$1,000 fee on a $2,500,000 deal") == 2_500_000.0


def test_no_amount_returns_zero():
    assert _extract_max_amount("what is the revenue trend") == 0.0


def test_ignores_oversized_artifact():
    # A malformed generated draft can emit a nonsensical figure (stray
    # zero-groups); it must not be reported. The legitimate $197,653 wins.
    text = "Total $500,000,000,000,000,000 ... actual due $197,653.12"
    assert _extract_max_amount(text) == 197_653.12


def test_detects_compliance_flags_mnpi_and_blackout():
    from src.graph.nodes.hitl_gate import _check_compliance_flags

    # MNPI detection
    flags_mnpi = _check_compliance_flags("What is the company's internal MNPI projection for Q4?")
    assert len(flags_mnpi) > 0
    assert any("mnpi" in f.lower() for f in flags_mnpi)

    # Blackout period / window detection
    flags_blackout = _check_compliance_flags("Can an insider execute transactions during the blackout period?")
    assert len(flags_blackout) > 0
    assert any("blackout period" in f.lower() for f in flags_blackout)

    flags_window = _check_compliance_flags("Are executives subject to the trading blackout window?")
    assert len(flags_window) > 0
    assert any("blackout window" in f.lower() for f in flags_window)

    # Restricted list
    flags_restricted = _check_compliance_flags("Verify if ticker XYZ is on the institutional restricted list.")
    assert len(flags_restricted) > 0
    assert any("restricted list" in f.lower() for f in flags_restricted)

    # Clean query should return zero flags
    clean_flags = _check_compliance_flags("What was Apple's total net sales in fiscal year 2023?")
    assert len(clean_flags) == 0


def test_hitl_gate_triggers_on_compliance_keywords():
    from src.graph.nodes.hitl_gate import hitl_gate_node

    # For research_analyst, blackout window triggers approval requirement
    state = {
        "sanitized_query": "Can we purchase shares before earnings during the blackout period?",
        "generated_answer": "Standard trading guidelines apply.",
        "user_role": "analyst",
    }
    result = hitl_gate_node(state)
    assert result["requires_human_approval"] is True

    # For risk_compliance or admin, compliance keywords do NOT self-block
    admin_state = {
        "sanitized_query": "Audit review of MNPI protocols and trading restrictions",
        "generated_answer": "Audit report complete.",
        "user_role": "risk_compliance",
    }
    admin_result = hitl_gate_node(admin_state)
    assert admin_result["requires_human_approval"] is False


def test_hitl_gate_triggers_on_allocation_threshold():
    from src.graph.nodes.hitl_gate import hitl_gate_node

    # Analyst threshold is $100,000. $2,500,000 exceeds threshold
    state = {
        "sanitized_query": "Execute portfolio rebalance with $2,500,000 capital",
        "generated_answer": "Rebalance planned.",
        "user_role": "analyst",
    }
    result = hitl_gate_node(state)
    assert result["requires_human_approval"] is True


if __name__ == "__main__":
    import inspect
    import sys
    current_module = sys.modules[__name__]
    test_funcs = [
        obj for name, obj in inspect.getmembers(current_module)
        if inspect.isfunction(obj) and name.startswith("test_")
    ]
    print(f"Running {len(test_funcs)} HITL gate unit tests...")
    passed = 0
    for fn in test_funcs:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"FAILED: {fn.__name__} - {e}")
            raise
    print(f"All {passed}/{len(test_funcs)} tests passed successfully!")


