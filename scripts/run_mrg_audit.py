"""CLI runner for Model Risk Governance (MRG / SR 11-7) automated audit.

Usage:
    python scripts/run_mrg_audit.py [--output docs/mrg_validation_report.md]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure project root is on PYTHONPATH
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.governance.mrg_validator import ModelRiskAuditor


def main():
    parser = argparse.ArgumentParser(description="Run SR 11-7 Model Risk Audit")
    parser.add_argument(
        "--output",
        type=Path,
        default=_PROJECT_ROOT / "docs" / "mrg_validation_report.md",
        help="Path to generate markdown validation dossier",
    )
    args = parser.parse_args()

    auditor = ModelRiskAuditor()
    results = auditor.run_all()
    auditor.generate_markdown_report(args.output)

    summary = results["summary"]
    print("=" * 65)
    print("  FinRAG — SR 11-7 Model Risk Audit Complete")
    print(f"  Tests: {summary['passed']}/{summary['total_tests']} PASSED, {summary['failed']} FAILED")
    print(f"  Report generated: {args.output}")
    print("=" * 65)

    sys.exit(0 if summary["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
