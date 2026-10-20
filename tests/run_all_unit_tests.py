"""Master unit test runner for FinRAG platform.

Runs all standalone unit test suites across quantitative tools, MCP protocols,
RBAC security gates, and ingestion components.

Usage:
    python tests/run_all_unit_tests.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

TEST_SUITES = [
    ("Quantitative Investment Metrics", "tests/unit/test_investment_metrics.py"),
    ("Role-Gated MCP-Style Tool Service", "tests/unit/test_mcp_server.py"),
    ("In-Graph MCP Tool Calling & Traced RBAC", "tests/unit/test_research_agent_mcp.py"),
    ("Institutional Market Data Provider", "tests/unit/test_market_data.py"),
    ("BGE-M3 Dense Embeddings", "tests/unit/test_embeddings.py"),
    ("Arize Phoenix & OpenInference Tracing", "tests/unit/test_tracing.py"),
    ("Asset Management RBAC & Governance", "tests/unit/test_asset_management_rbac.py"),
    ("RBAC LangGraph Gate Node", "tests/unit/test_rbac_gate.py"),
    ("Metadata Extractor & Doc Detection", "tests/unit/test_metadata_extractor.py"),
    ("Recursive & Contextual Chunker", "tests/unit/test_chunker.py"),
    ("AST-Restricted Financial Calculator", "tests/unit/test_calculator.py"),
    ("Candidate Entity/Year Validator", "tests/unit/test_candidate_validator.py"),
    ("Company Registry Canonicalization", "tests/unit/test_company_registry.py"),
    ("HITL Gate & Compliance / MNPI / Blackout", "tests/unit/test_hitl_gate.py"),
    ("Dedicated MCP Tools Node & Router Intent", "tests/unit/test_mcp_tools_node.py"),
    ("Quantitative Decision Models & Engine", "tests/unit/test_decision_engine.py"),
    ("System 1 Strand Decider 2B", "tests/unit/test_decider_2b.py"),
    ("SEC EDGAR Audited Fundamentals & Filings", "tests/unit/test_sec_edgar.py"),
    ("Model Risk Governance (Fed SR 11-7)", "tests/unit/test_mrg_governance.py"),
]


def main():
    print("=" * 70)
    print("  FinRAG — Institutional Asset Management Test Matrix")
    print("=" * 70)
    print()

    env = dict(os.environ)
    env["PYTHONPATH"] = "."

    passed = 0
    failed = 0
    start_time = time.time()

    for name, path in TEST_SUITES:
        print(f"▶ Running: {name:<38} [{path}]... ", end="", flush=True)
        res = subprocess.run([sys.executable, path], capture_output=True, text=True, env=env)
        if res.returncode == 0:
            print("✓ PASSED")
            passed += 1
        else:
            print("✗ FAILED")
            print("-" * 60)
            print(f"Command: {sys.executable} {path}")
            print(f"Exit code: {res.returncode}")
            if res.stdout:
                print(res.stdout)
            if res.stderr:
                print(res.stderr)
            print("-" * 60)
            failed += 1

    elapsed = time.time() - start_time
    print()
    print("=" * 70)
    if failed == 0:
        print(f"  RESULT: ALL {passed}/{len(TEST_SUITES)} TEST SUITES PASSED ({elapsed:.2f}s)")
    else:
        print(f"  RESULT: {failed} SUITES FAILED, {passed} passed ({elapsed:.2f}s)")
    print("=" * 70)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
