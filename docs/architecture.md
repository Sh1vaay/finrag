# Architecture

## Overview

The FinRAG Multi-Agent Investment Platform is a `StateGraph` built with LangGraph. Every query flows through a 20-node directed graph connected by conditional edge routers. State is a `TypedDict` ([src/models/state.py](../src/models/state.py)) that accumulates across the pipeline; per-turn fields are reset on each new invocation, while `messages` is appended via the `add_messages` reducer to preserve conversation history.

## Graph Topology

![FinRAG Agent architecture](diagrams/architecture.png)

`START → rbac_gate → guardrails → entity_extractor → router`. The router (backed by System 1 fast SLM triage) directs queries along three active paths:
1. **Direct Document Retrieval**: `retrieval → reranker → retrieval_evaluator → grader → decision_engine`
2. **Selective Research Agent**: `research_agent → decision_engine`
3. **Dedicated Quantitative MCP Tools**: `mcp_tools → decision_engine`

All execution branches converge on `decision_engine → generator → hallucination_checker → hitl_gate → response_formatter → END`. Conditional retries loop retrieval (via `query_rewriter`) and generation; four terminal nodes (`blocked_response`, `out_of_scope_response`, `clarification_response`, `no_info_response`) short-circuit safely to `END`.

## Component diagrams

Deep-dives into the subsystems described below.

**Role-based access control** — roles resolve to a Qdrant payload filter; enforcement happens at the storage layer, so agentic queries cannot bypass it.
![RBAC](diagrams/excalidraw/art/02-rbac.png)

**Human-in-the-loop approval** — high-stakes answers pause via `interrupt()`, persist to Postgres, and require a different party to approve (self-approval blocked).
![HITL](diagrams/excalidraw/art/03-hitl.png)

**Conversation memory** — follow-up questions are rewritten into standalone queries from thread history before routing.
![Conversation memory](diagrams/excalidraw/art/04-memory.png)

**Retrieval → rerank → grade** — hybrid dense+BM25 (RRF) to a top-50 pool, cross-encoder rerank to top-8, then a 3-stage grader.
![Retrieval pipeline](diagrams/excalidraw/art/05-retrieval-rerank-grade.png)

**Research-agent subgraph** — selective decomposition for research-required queries, bounded to ≤5 LLM turns.
![Research-agent subgraph](diagrams/excalidraw/art/06-research-agent.png)

**Guardrails cascade** — PII redaction then a 3-layer injection defense escalating by cost.
![Guardrails cascade](diagrams/excalidraw/art/07-guardrails.png)

**Evaluation methodology** — how the 72.7% financial benchmark pass rate is measured under a calibrated judge.
![Evaluation methodology](diagrams/excalidraw/art/08-eval-methodology.png)

## Node Responsibilities

### Entry and Safety

| Node | File | Purpose |
|------|------|---------|
| `rbac_gate` | [src/graph/nodes/rbac_gate.py](../src/graph/nodes/rbac_gate.py) | Maps JWT role → allowed doc types/confidentiality |
| `guardrails` | [src/graph/nodes/guardrails.py](../src/graph/nodes/guardrails.py) | PII redaction (Presidio), 3-layer injection defense, query contextualization |
| `entity_extractor` | [src/graph/nodes/entity_extractor.py](../src/graph/nodes/entity_extractor.py) | Two-tier company/fiscal-year extraction (dictionary pass + LLM fallback) used as a Qdrant payload filter |

The guardrails node runs three injection checks in order of cost: (1) regex heuristics, (2) LLM Guard's PromptInjection scanner, (3) an LLM classifier — only invoked when LLM Guard returns a borderline score (0.5 ≤ x < 0.9). After safety checks pass, it runs a contextualization step that rewrites coreferential follow-ups (e.g., "what about Microsoft?") into standalone questions using prior message history.

### Routing

| Node | File | Purpose |
|------|------|---------|
| `router` | [src/graph/nodes/router.py](../src/graph/nodes/router.py) | Classifies intent (retrieval / clarification / out_of_scope) and complexity (simple lookup vs research-required) |

Structured output via Pydantic. Uses Llama 3.3 70B on Groq (free tier) with automatic fallback to GPT-4o-mini. The complexity classification is what selects the direct retrieval path versus the `research_agent` subgraph.

### Retrieval and Correction

| Node | File | Purpose |
|------|------|---------|
| `retrieval` | [src/graph/nodes/retrieval.py](../src/graph/nodes/retrieval.py) | Qdrant hybrid search — dense + BM25 sparse fused via RRF (k=60), RBAC payload filter applied at query time, top-50 pool |
| `reranker` | [src/graph/nodes/reranker.py](../src/graph/nodes/reranker.py) | BAAI/bge-reranker-v2-m3 cross-encoder; reranks the pool down to top-8 |
| `retrieval_evaluator` | [src/graph/nodes/retrieval_evaluator.py](../src/graph/nodes/retrieval_evaluator.py) | Optional CRAG-style sufficiency check before grading |
| `grader` | [src/graph/nodes/grader.py](../src/graph/nodes/grader.py) | 3-stage relevance: entity_match (deterministic) → optional LTR gate → 8-way parallel LLM grading |
| `query_rewriter` | [src/graph/nodes/query_rewriter.py](../src/graph/nodes/query_rewriter.py) | Rewrites the query if grading failed |

Retrieval pulls a top-50 pool, the cross-encoder reranks to top-8, and the grader filters those for relevance in three escalating stages (free deterministic entity match → optional learning-to-rank gate → parallel LLM grading). If fewer than `GRADING_MIN_RELEVANT_CHUNKS` are relevant, the graph loops back via `query_rewriter` up to `MAX_RETRIEVAL_RETRIES` (default 2) times. After retries are exhausted, the graph terminates with `no_info_response`. The reranker uses the stock `bge-reranker-v2-m3`; a LoRA adapter hook exists (`RERANKER_ADAPTER_PATH`) but is inactive in the canonical configuration.

### Research-agent subgraph

| Node | File | Purpose |
|------|------|---------|
| `research_agent` | [src/graph/nodes/research_agent.py](../src/graph/nodes/research_agent.py) | Selective multi-turn subgraph for research-required queries |

### Quantitative Decision Engine & MCP Tool Nodes

| Node | File | Purpose |
|------|------|---------|
| `mcp_tools` | [src/graph/nodes/mcp_tools.py](../src/graph/nodes/mcp_tools.py) | Direct fast-path execution of Model Context Protocol (MCP) tools for live pricing, valuation multiples, and SEC filings |
| `decision_engine` | [src/graph/nodes/decision_engine.py](../src/graph/nodes/decision_engine.py) | Deterministic financial analytics: VaR/CVaR risk models, Ledoit-Wolf portfolio optimization, factor exposures, Altman Z, and Piotroski F-Score |

The `decision_engine` intercepts outputs from `grader`, `research_agent`, or `mcp_tools`. It executes deterministic financial algorithms over structured facts (resolving tickers dynamically via SEC EDGAR XBRL) and populates `quantitative_analysis` in state, ensuring the downstream `generator` receives pre-calculated, verified mathematical artifacts.

### Generation and Verification

| Node | File | Purpose |
|------|------|---------|
| `generator` | [src/graph/nodes/generator.py](../src/graph/nodes/generator.py) | Claude Sonnet 4.6 generates the answer from relevant chunks (prompt-cached system prompt); appends AIMessage to state |
| `hallucination_checker` | [src/graph/nodes/hallucination.py](../src/graph/nodes/hallucination.py) | Claude Sonnet 4.6 verifies the answer is grounded in retrieved sources (separate high-stakes path, also Sonnet 4.6) |

If the hallucination check returns `grounded=False` with confidence below `HALLUCINATION_THRESHOLD` (0.7), the graph loops back to `generator` up to `MAX_GENERATION_RETRIES` (default 2) times. After retries, the answer is returned with a disclaimer.

### Approval and Formatting

| Node | File | Purpose |
|------|------|---------|
| `hitl_gate` | [src/graph/nodes/hitl_gate.py](../src/graph/nodes/hitl_gate.py) | Interrupts graph if dollar amount exceeds role threshold |
| `response_formatter` | [src/graph/nodes/response_formatter.py](../src/graph/nodes/response_formatter.py) | Builds final response with deduplicated source list |

The HITL gate extracts dollar amounts from the generated answer via regex (handles `$100k`, `$2.5M`, `$383.3 billion`, etc.), compares against `requires_hitl_above` from [rbac_config.py](../src/config/rbac_config.py), and calls `interrupt()` when over threshold. Execution resumes via `POST /hitl/approve` or `/hitl/reject`, which invoke the graph with `Command(resume="approved"|"rejected")`.

### Terminal Nodes

All four terminal nodes live in [src/graph/nodes/terminal_nodes.py](../src/graph/nodes/terminal_nodes.py):

- `blocked_response` — guardrails or HITL rejection
- `out_of_scope_response` — router classified query as off-topic
- `clarification_response` — router needs more info
- `no_info_response` — retrieval+grading failed after retries

## Conditional Edge Logic

All routing logic is in [src/graph/edges.py](../src/graph/edges.py):

| Router | From | Targets |
|--------|------|---------|
| `route_after_guardrails` | `guardrails` | `clean` → `entity_extractor`, `blocked` → `blocked_response` |
| `route_after_router` | `router` | `retrieval`, `research_required` → `research_agent`, `clarification`, `out_of_scope` |
| `route_after_retrieval_evaluator` | `retrieval_evaluator` | `accept` → `grader`, `retry` → `query_rewriter` |
| `route_after_grading` | `grader` | `sufficient` → `generator`, `retry` → `query_rewriter`, `no_info` → `no_info_response` |
| `route_after_hallucination` | `hallucination_checker` | `grounded`/`disclaimer` → `hitl_gate`, `retry` → `generator` |
| `route_after_hitl` | `hitl_gate` | `no_approval_needed`/`approved` → `response_formatter`, `rejected` → `blocked_response` |

## State Management

`RAGState` ([src/models/state.py](../src/models/state.py)) has ~21 fields grouped by concern:

- **Input**: `messages` (with `add_messages` reducer)
- **Auth**: `user_id`, `user_role`, `allowed_doc_types`
- **Guardrails**: `guardrail_status`, `detected_pii_entities`, `sanitized_query`
- **Routing**: `query_intent`
- **Retrieval**: `retrieved_chunks`, `retrieval_query`
- **Grading**: `relevant_chunks`, `grading_results`
- **Generation**: `generated_answer`
- **Hallucination**: `hallucination_status`, `hallucination_score`
- **HITL**: `requires_human_approval`, `human_decision`
- **Control flow**: `retrieval_retry_count`, `generation_retry_count`
- **Output**: `final_response`, `response_metadata`

The `messages` field uses LangGraph's `add_messages` reducer, which *appends* rather than replaces. This means when the same `thread_id` is re-invoked, prior messages are retained automatically (via PostgresSaver checkpointing), enabling multi-turn conversations.

## LLM Strategy

Managed by [LLMFactory](../src/services/llm_factory.py). Provider selection per node:

| Node | Primary | Fallback | Why |
|------|---------|----------|-----|
| `router` | Groq Llama 3.3 70B | OpenAI GPT-4o-mini | Classification task, free tier |
| `grader` | Groq Llama 3.3 70B | OpenAI GPT-4o-mini | Binary classification, high volume |
| `query_rewriter` | Groq Llama 3.3 70B | OpenAI GPT-4o-mini | Simple rewriting |
| `guardrails` (contextualizer, injection layer 3) | Groq Llama 3.3 70B | OpenAI GPT-4o-mini | Classification/rewriting |
| `generator` | Claude Sonnet 4.6 | — | Financial accuracy matters; system prompt is prompt-cached |
| `hallucination_checker` | Claude Sonnet 4.6 | — | Nuanced grounding assessment; high-stakes path also Sonnet 4.6 (configurable to Opus) |
| `research_agent` (decompose, sufficiency) | OpenAI GPT-4o-mini | — | High-volume planning turns, cost-sensitive |
| `research_agent` (synthesize) | Claude Sonnet 4.6 | — | Final synthesis quality matters |

Classification-tier providers (router/grader/query_rewriter/guardrails) fall back automatically on exceptions (rate limits, outages); the Anthropic generation tier does not fall back.

## Ingestion Pipeline

Separate from the query graph. Lives in [src/ingestion/](../src/ingestion/):

1. [docling_loader.py](../src/ingestion/docling_loader.py) — PDF → per-page text (pypdf) + markdown (Docling)
2. [metadata_extractor.py](../src/ingestion/metadata_extractor.py) — Detects `doc_type`, `company`, `confidentiality`
3. [chunker.py](../src/ingestion/chunker.py) — Recursive splitter (~800 chars, 150 overlap), chunks each page independently so every chunk carries a `page_number`
4. [qdrant_uploader.py](../src/ingestion/qdrant_uploader.py) — Embeds (voyage-finance-2 canonical, OpenAI text-embedding-3-small as the 1536-dim fallback) and upserts to Qdrant in batches

Run via `python scripts/seed_qdrant.py --sample` or `python scripts/internal/data_prep/ingest_documents.py --input data/raw/`.

## Model Context Protocol (MCP) Server & Tool Registry

Built to the official **Model Context Protocol (MCP)** JSON-RPC 2.0 standard ([src/tools/mcp_server.py](../src/tools/mcp_server.py)), the agent acts as an MCP Tool Host for Claude Desktop, agentic IDEs, and internal investment toolkits.

### MCP Endpoint Surface:
- `POST /v1/mcp`: Official JSON-RPC 2.0 endpoint handling `initialize`, `tools/list`, and `tools/call`.
- `GET /v1/mcp/tools`: REST discovery endpoint returning tool schemas.
- `POST /v1/mcp/tools/{tool_name}`: REST execution endpoint with role-based access validation.

```mermaid
flowchart LR
    A[Front-Office Analyst / PM] -->|Query| B[FastAPI /v1/mcp]
    B --> C[MCP JSON-RPC Server]
    C --> D{RBAC Gate}
    D -->|Authorized| E[Investment Tools Engine]
    D -->|Forbidden| F[403 Authorization Error]
    E --> G[Valuation Multiples]
    E --> H[Profitability & Return]
    E --> I[Growth & Variance]
    E --> J[Portfolio Concentration & HHI]
    E --> K[AST-Safe Arithmetic]
    E --> L[VaR / CVaR Engine]
    E --> M[Ledoit-Wolf Optimizer]
    E --> N[Fama-French Factor Attribution]
    E --> O[SEC EDGAR XBRL Pipeline]
    E --> P[SR 11-7 MRG Validator]
```

## Quantitative Investment Tools Engine

Financial applications require deterministic numerical precision. Generative LLMs hallucinate complex arithmetic; this system offloads all math to dedicated, unit-tested quantitative tools ([src/tools/investment_metrics.py](../src/tools/investment_metrics.py), [src/tools/decision_models.py](../src/tools/decision_models.py), [src/services/sec_edgar_service.py](../src/services/sec_edgar_service.py)):

1. **Valuation Multiples Engine (`calculate_valuation_multiples`)**:
   - Calculates Trailing & Forward P/E, Enterprise Value (EV), EV/EBITDA, Price-to-Book (P/B), and Free Cash Flow (FCF) Yield.
   - Handles capital structure shifts (net debt, cash, minority interest).

2. **Profitability & Capital Efficiency (`calculate_profitability_metrics`)**:
   - Calculates Gross, Operating, and Net Profit Margins.
   - Computes Return on Equity (ROE), Return on Assets (ROA), and Debt-to-Equity (D/E).

3. **Growth & Variance Analysis (`calculate_growth_metrics`)**:
   - Computes Year-over-Year (YoY) and Quarter-over-Quarter (QoQ) revenue/earnings growth.
   - Calculates margin expansion/compression in basis points (bps) and Compound Annual Growth Rate (CAGR).

4. **Portfolio Risk & Concentration (`calculate_portfolio_risk_metrics`)**:
   - Computes Herfindahl-Hirschman Index (HHI) concentration scores (<1,500 diversified, >2,500 highly concentrated).
   - Assesses Top-5 and Top-10 holding weight concentrations.
   - Checks compliance against standard regulatory diversification thresholds (e.g. UCITS 5/10/40 rule and 1940 Act diversification tests).

5. **AST-Safe Arithmetic Calculator (`src/tools/calculator.py`)**:
   - Parses mathematical formulas via Python AST without unsafe `eval()`.
   - Guaranteed deterministic execution preventing calculation hallucinations.

6. **Buy-Side Parametric & Historical VaR/CVaR (`calculate_portfolio_var_cvar`)**:
   - Calculates Value at Risk (VaR) and Conditional VaR (Expected Shortfall) at 95% and 99% confidence intervals.
   - Multi-horizon scaling across 1-day, 10-day (Basel III market risk standard), and 1-year horizons.
   - Parametric Gaussian distribution and empirical historical simulation with Cornish-Fisher expansion awareness. Restricted to `portfolio_manager` role.

7. **Ledoit-Wolf Covariance Shrinkage & Mean-Variance Optimizer (`optimize_portfolio_weights`)**:
   - Modern Portfolio Theory (MPT) mean-variance optimization solving for Maximum Sharpe Ratio or Minimum Volatility.
   - Ledoit-Wolf analytical shrinkage target towards identity matrix to eliminate ill-conditioned covariance matrices in small-n sample spaces.
   - Enforces non-negative long-only simplex projection with configurable single-position capacity constraints (`max_single_cap`). Restricted to `portfolio_manager` role.

8. **Fama-French Multi-Factor Attribution (`calculate_factor_exposures`)**:
   - Multi-factor OLS regression calculating portfolio alpha, beta, and factor sensitivities (Market MKT-RF, Size SMB, Value HML, Momentum MOM, Quality RMW, Investment CMA).
   - Reports residual volatility and idiosyncratic risk decomposition.

9. **Automated SEC EDGAR XBRL Pipeline & Auto-CIK Resolution (`get_sec_filings`)**:
   - Direct integration with SEC EDGAR REST API and company facts endpoint (`data.sec.gov/api/xbrl/companyfacts/`).
   - Dynamic ticker-to-CIK mapping cached in SQLite, User-Agent compliant headers (`SEC_EDGAR_USER_AGENT`), and structured 10-K, 10-Q, 8-K fact extraction.

10. **Federal Reserve SR 11-7 / OCC 2011-12 Model Risk Governance (`src/governance/mrg_validator.py`)**:
    - Embedded validation hooks testing quantitative models for conceptual soundness, input data verification, parameter sensitivity benchmarking, and outcome analysis.
    - Generates institutional Model Risk Governance validation reports ([docs/mrg_validation_report.md](mrg_validation_report.md)).

## System 1 SLM Hybrid Decider Routing (`src/services/decider_2b.py`)

A local 2B parameter Small Language Model acts as the fast "System 1" intent router (e.g. Strands Decider / Gemma 2 2B / Qwen 2.5 1.5B via vLLM/Ollama):
- Categorizes queries into `RETRIEVAL`, `QUANTITATIVE_TOOL`, `DECISION_MODEL`, `HITL_APPROVAL`, or `DIRECT_ANSWER`.
- Operates under sub-50ms latency.
- Graceful fallback: If local SLM endpoint is unavailable or unreachable, automatically routes through primary LLM classifier without pipeline interruption.

## Persistence

PostgreSQL stores LangGraph checkpoints for HITL resumption and conversation history. Initialized in [src/api/main.py](../src/api/main.py) lifespan via `AsyncPostgresSaver`. On API startup, the checkpointer tables are created with `CREATE INDEX CONCURRENTLY`, which requires an autocommit connection (handled separately from the runtime pool).

If PostgreSQL is unavailable, the app logs an error and continues with HITL disabled — the graph runs without a checkpointer and interrupts are auto-approved.

## Observability

LangSmith tracing is always on when `LANGCHAIN_API_KEY` is set. Every graph invocation is tagged with:
- `run_name`: `"rag_query"` or `"rag_query_stream"`
- `tags`: `["api", f"role:{user.role}"]`
- `metadata`: `{"user_id", "role", "thread_id", "hitl_enabled"}`

Project names are environment-specific via `settings.langchain_project_name` (e.g., `rag-agent-dev`, `rag-agent-prod`).

For LLM cost and usage observability, every model call is routed through a LiteLLM proxy that forwards traces to a self-hosted Langfuse v3 instance ([settings.py](../src/config/settings.py) `LANGFUSE_*`). The admin cost endpoints ([src/api/routes/admin.py](../src/api/routes/admin.py)) query Langfuse's API to aggregate spend by user, model, and trace name, so cost attribution lives in Langfuse rather than in the agent itself.

## Evaluation

Current financial benchmark results, methodology, the calibrated judge (Cohen's κ = 0.932), and reproduction commands live in [docs/evaluation.md](evaluation.md). CI regression thresholds are defined in [tests/evaluation/eval_config.py](../tests/evaluation/eval_config.py).
