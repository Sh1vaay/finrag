"""Unit test suite for Model Risk Governance (MRG / SR 11-7) automated validation."""

import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.governance.mrg_validator import ModelRiskAuditor


def test_mrg_audit_battery():
    auditor = ModelRiskAuditor()
    results = auditor.run_all()
    summary = results["summary"]
    assert summary["total_tests"] >= 8
    assert summary["failed"] == 0
    assert summary["passed"] == summary["total_tests"]

    # Verify report emission
    output_path = Path(__file__).parent.parent.parent / "docs" / "mrg_validation_report.md"
    auditor.generate_markdown_report(output_path)
    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert "Federal Reserve SR 11-7" in content
    assert "Model Risk Governance" in content
    assert "Zero-Hallucination Deterministic Enforcement" in content


if __name__ == "__main__":
    test_mrg_audit_battery()
    print("All Model Risk Governance (SR 11-7) validation tests passed successfully!")
