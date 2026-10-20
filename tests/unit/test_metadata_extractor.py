try:
    import pytest
except ImportError:
    pytest = None
import sys
from pathlib import Path

from src.ingestion.metadata_extractor import (
    _detect_doc_type,
    _extract_company,
    _extract_company_slug,
    extract_metadata,
)

# ---------------------------------------------------------------------------
# _detect_doc_type tests
# ---------------------------------------------------------------------------


def test_detect_doc_type_10k_in_filename():
    """Filenames containing '10-k' or '10k' patterns are detected as '10k'."""
    assert _detect_doc_type("apple_10-k_2024", "") == "10k"


def test_detect_doc_type_annual_report_in_filename():
    """'annual report' pattern in filename is detected as '10k' via regex."""
    # The pattern list includes "annual report" which uses re.search on combined text
    assert _detect_doc_type("annual report filing", "") == "10k"


def test_detect_doc_type_10q_in_filename():
    """Filenames containing '10-q' or '10q' patterns are detected as '10q'."""
    assert _detect_doc_type("msft_10-q_q2_2024", "") == "10q"


def test_detect_doc_type_earnings_transcript():
    """Earnings calls and transcripts are detected as 'earnings_transcript'."""
    assert _detect_doc_type("aapl_earnings_call_q3", "") == "earnings_transcript"
    assert _detect_doc_type("transcript_memo", "prepared remarks by ceo and q&a session") == "earnings_transcript"


def test_detect_doc_type_equity_research():
    """Analyst notes and equity research are detected as 'equity_research'."""
    assert _detect_doc_type("morgan_stanley_equity_research", "") == "equity_research"
    assert _detect_doc_type("broker_note", "price target $240 and initiation of coverage") == "equity_research"


def test_detect_doc_type_portfolio_holdings():
    """Portfolio and fund holdings sheets are detected as 'portfolio_holdings'."""
    assert _detect_doc_type("global_tech_portfolio_holdings", "") == "portfolio_holdings"
    assert _detect_doc_type("fund_review", "asset allocation and active weight analysis") == "portfolio_holdings"


def test_detect_doc_type_falls_back_to_text_content():
    """When the filename has no match, doc type is detected from text content."""
    assert _detect_doc_type("generic_document", "this is a 10-K filing for fiscal year ended") == "10k"


def test_detect_doc_type_returns_unknown_for_unrecognized():
    """Unrecognized filenames with no matching text patterns return 'unknown'."""
    assert _detect_doc_type("random_notes", "some unrelated content about weather") == "unknown"


# ---------------------------------------------------------------------------
# _extract_company tests (legacy — returns display name)
# ---------------------------------------------------------------------------


def test_extract_company_from_filename():
    """Company display name is extracted when the slug appears in the filename."""
    result = _extract_company("apple_10k_2024", "")
    assert result == "Apple Inc."


def test_extract_company_from_text_content():
    """Company display name is extracted from text when filename has no match."""
    result = _extract_company("generic_report", "this report covers microsoft quarterly earnings")
    assert result == "Microsoft Inc."


def test_extract_company_tesla():
    """Tesla is correctly identified and formatted."""
    result = _extract_company("tesla_annual", "")
    assert result == "Tesla Inc."


def test_extract_company_returns_unknown_when_no_match():
    """Returns 'Unknown' (display) when no known company is found."""
    result = _extract_company("quarterly_report", "some numbers and data about widgets")
    assert result == "Unknown"


# ---------------------------------------------------------------------------
# _extract_company_slug tests — the version used for Qdrant payload filtering
# ---------------------------------------------------------------------------


def test_extract_company_slug_apple():
    assert _extract_company_slug("apple_10k_2024", "") == "apple"


def test_extract_company_slug_microsoft_from_text():
    assert _extract_company_slug("generic_report", "microsoft reported revenue") == "microsoft"


def test_extract_company_slug_returns_unknown_when_no_match():
    assert _extract_company_slug("widgets_report", "data about widgets") == "unknown"


# ---------------------------------------------------------------------------
# Confidentiality detection (tested via extract_metadata since it is inline)
# ---------------------------------------------------------------------------


def test_confidentiality_detects_confidential():
    """Text containing 'confidential' results in 'internal' confidentiality."""
    doc = {"text": "This document is strictly confidential and should not be shared."}
    result = extract_metadata(Path("report.pdf"), doc)
    assert result["confidentiality"] == "internal"


def test_confidentiality_detects_internal_use_only():
    """Text containing 'internal use only' results in 'internal' confidentiality."""
    doc = {"text": "For internal use only. Distribution is restricted."}
    result = extract_metadata(Path("memo.pdf"), doc)
    assert result["confidentiality"] == "internal"


def test_confidentiality_defaults_to_public():
    """Text with no confidentiality markers defaults to 'public'."""
    doc = {"text": "General public information about the quarterly results."}
    result = extract_metadata(Path("public_report.pdf"), doc)
    assert result["confidentiality"] == "public"


# ---------------------------------------------------------------------------
# extract_metadata integration tests
# ---------------------------------------------------------------------------


def test_extract_metadata_uses_doc_type_override():
    """When doc_type_override is provided, it takes precedence over detection."""
    doc = {"text": "This is an internal equity research memo."}
    result = extract_metadata(Path("random_file.pdf"), doc, doc_type_override="equity_research")
    assert result["doc_type"] == "equity_research"


def test_extract_metadata_returns_complete_dict():
    """extract_metadata returns a dict with all required keys including company slug + display name."""
    doc = {"text": "Apple Inc. annual report for fiscal year.", "num_pages": 42}
    result = extract_metadata(Path("apple_10k_2024.pdf"), doc)

    required_keys = {
        "doc_type",
        "company",
        "company_name",
        "confidentiality",
        "source_file",
        "num_pages",
        "fiscal_year",
    }
    assert required_keys == set(result.keys()), f"Missing keys: {required_keys - set(result.keys())}"
    assert result["doc_type"] == "10k"
    assert result["company"] == "apple"  # slug for filtering
    assert result["company_name"] == "Apple Inc."  # display name
    assert result["confidentiality"] == "public"
    assert result["source_file"] == "apple_10k_2024.pdf"
    assert result["num_pages"] == 42


def test_extract_metadata_source_file_is_filename_only():
    """source_file should be just the filename, not the full path."""
    doc = {"text": "Some content."}
    result = extract_metadata(Path("/data/raw/tesla_10k_2023.pdf"), doc)
    assert result["source_file"] == "tesla_10k_2023.pdf"


def test_extract_metadata_num_pages_defaults_to_zero():
    """When num_pages is not in the document dict, it defaults to 0."""
    doc = {"text": "Some content."}
    result = extract_metadata(Path("report.pdf"), doc)
    assert result["num_pages"] == 0


if __name__ == "__main__":
    import inspect

    current_module = sys.modules[__name__] if "sys" in globals() else __import__("sys").modules[__name__]
    test_funcs = [
        obj for name, obj in inspect.getmembers(current_module) if inspect.isfunction(obj) and name.startswith("test_")
    ]
    print(f"Running {len(test_funcs)} metadata extractor unit tests...")
    passed = 0
    for fn in test_funcs:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"FAILED: {fn.__name__} - {e}")
            raise
    print(f"All {passed}/{len(test_funcs)} tests passed successfully!")
