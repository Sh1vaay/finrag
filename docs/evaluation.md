# Pipeline Evaluation

This document details the evaluation methodology, metrics, quality gates, and benchmarking procedures for the FinRAG multi-agent pipeline.

## Overview

The FinRAG evaluation framework assesses end-to-end retrieval and generation quality against structured ground-truth financial question-and-answer benchmarks.

The primary evaluation harness lives in [`tests/evaluation/run_evaluation.py`](../tests/evaluation/run_evaluation.py), operating over the curated SEC filing benchmark in [`tests/evaluation/eval_dataset.json`](../tests/evaluation/eval_dataset.json).

## Core Evaluation Metrics

The pipeline is benchmarked across industry-standard RAG quality dimensions using RAGAS and DeepEval:

| Metric | Target Threshold | Description |
|---|:---:|---|
| **Faithfulness** | ≥ 0.70 | Measures whether all claims in the generated response are grounded directly in retrieved contexts (anti-hallucination). |
| **Answer Relevancy** | ≥ 0.75 | Measures how directly and completely the generated response addresses the user's specific financial question. |
| **Context Precision** | ≥ 0.65 | Measures the signal-to-noise ratio in retrieved context, evaluating if relevant chunks appear near the top of the ranked list. |
| **Context Recall** | ≥ 0.60 | Measures whether all necessary context required to produce the ground-truth answer was retrieved. |

Configured thresholds and CI gates are codified in [`tests/evaluation/eval_config.py`](../tests/evaluation/eval_config.py).

## Benchmark Dataset

The primary benchmark dataset (`tests/evaluation/eval_dataset.json`) covers financial analysis queries across public SEC 10-K filings (Apple, Microsoft, Tesla), categorized into specific financial analytical slices:

- **10-K Revenue & Growth**: Topline revenue, year-over-year growth, product segment breakdowns.
- **10-K Net Income & Profitability**: Operating income, margins, tax provisions, net earnings.
- **10-K Balance Sheet & Liquidity**: Cash and equivalents, marketable securities, total assets, current ratio.
- **10-K Cash Flow & Capital Allocation**: Operating cash flows, capital expenditures, free cash flow, share repurchases.
- **Comparative & Multi-Hop**: Multi-company comparisons and cross-statement deductions.
- **Calculations**: Derived ratios and percentage changes requiring precise numeric extraction.

## Running Evaluations

### Standard Evaluation Run

Run the full evaluation pipeline and output metrics to a JSON snapshot:

```bash
python tests/evaluation/run_evaluation.py --output tests/evaluation/eval_results/eval_latest.json
```

### CI Quality Gate Check

To run evaluation as a blocking CI gate that exits with status code 0 only if all thresholds pass:

```bash
python tests/evaluation/run_evaluation.py --check-thresholds
```

### Reusing Cached Pipeline Outputs

To re-run metric scoring without re-executing LLM inference on the pipeline (e.g., when testing metric variations or judge prompts):

```bash
python tests/evaluation/run_evaluation.py --skip-pipeline --output tests/evaluation/eval_results/eval_rescore.json
```

## Diagnostic Slicing

Evaluation runs produce detailed diagnostic breakdowns to pinpoint performance across different query types:

1. **Refusal Analysis**: Distinguishes between legitimate refusals (e.g. guardrail blocks or absent data) and false refusals.
2. **Category-Level Performance**: Aggregates precision, recall, and faithfulness by financial topic.
3. **Execution Profiling**: Records pipeline latency and evaluator time per query to monitor inference cost and response times.

Snapshot results stored in `tests/evaluation/eval_results/` can be surfaced in the Admin Dashboard under the **Evaluations** tab.

---

## Model Risk Governance (Federal Reserve SR 11-7)

Institutional asset management deployments require formal model risk validation per **Federal Reserve SR 11-7** / **OCC 2011-12**. FinRAG includes an automated model validation battery that audits:

1. **AST Arithmetic Sandboxing**: Ensures absolute zero math hallucinations and code-injection immunity in AST calculators.
2. **Storage-Level RBAC Boundaries**: Validates vector database payload filters to prevent cross-mandate information leakage.
3. **Regulatory & Adversarial Stress Tests**: Validates automated triggers for Material Non-Public Information (MNPI), blackout windows, and high-stakes capital allocation limits.
4. **Quantitative Model Invariants**: Confirms statistical coherence (e.g. Expected Shortfall $CVaR \ge VaR$, single-position portfolio weights $\le$ mandate caps).

### Running the MRG Validation Audit

Execute the automated model governance battery to generate an updated validation dossier:

```bash
python scripts/run_mrg_audit.py --output docs/mrg_validation_report.md
```

Generated validation artifacts are persisted in [`docs/mrg_validation_report.md`](mrg_validation_report.md).
