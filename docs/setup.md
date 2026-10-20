# Setup & Development

This document covers everything needed to run the project locally — backend API, CLI client, sample data, and the API surface.

## Prerequisites

- Python 3.12+
- Docker and Docker Compose
- Model & API Configuration:
  - **Option 1 (Placeholder / Mock)**: `LLM_PROVIDER_TYPE=placeholder` — runs the entire pipeline locally without any API keys or model servers.
  - **Option 2 (Open-Source / Local)**: `LLM_PROVIDER_TYPE=open_source` — point `OPEN_SOURCE_LLM_BASE_URL` to Ollama, vLLM, LM Studio, or LocalAI.
  - **Option 3 (Closed-Source)**: `LLM_PROVIDER_TYPE=closed_source` — provide `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`.
  - **Option 4 (Hybrid)**: Default production profile using Groq fast-path + Claude generator.

## Local development — backend

```bash
git clone https://github.com/Sh1vaay/finrag.git
cd finrag

# Using uv (recommended) or pip
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

cp .env.example .env
# Edit .env with your API keys + JWT_SECRET

# Bring up the substrate (Qdrant + Postgres + observability stack)
docker compose up -d

# Generate and seed sample financial documents
uv run python scripts/internal/data_prep/download_sample_data.py
uv run python scripts/seed_qdrant.py --sample

# Run the full test suite
uv run tests/run_all_unit_tests.py

# Start the API server with hot reload
make run                    # http://localhost:8000
```

## Local development — FinRAG CLI

The institutional command-line interface provides real-time streaming analysis, slash commands (`/analyst`, `/filings`, `/calc`, `/health`), and interactive sessions.

```bash
# Direct terminal execution via uv
uv run finrag chat --user analyst

# Or run via Python module
uv run python -m cli.main chat --user analyst
```

## Test accounts (development only)

| Username | Password | Role | Asset Management Function & Scope |
|----------|----------|------|-----------------------------------|
| `analyst` | `analyst123` | `research_analyst` | Fundamental Equity / Credit Research (10-Ks, 10-Qs, transcripts, equity research) |
| `pm` | `pm123` | `portfolio_manager` | Strategy, Portfolio Allocation & Execution (confidential portfolio holdings) |
| `compliance` | `compliance123` | `risk_compliance` | Mandate Limits, MNPI, Restricted Lists & Audit Sign-off |
| `admin` | `admin123` | `admin` | Platform MLOps, Model Registry & Universal MCP Access |
| `finance` | `finance123` | `research_analyst` | Backward-compatibility alias for equity research |
| `clevel` | `clevel123` | `portfolio_manager` | Backward-compatibility alias for executive portfolio management |

These are defined in `src/api/routes/auth.py` for local development. Replace with an institutional IdP (SSO / SAML / OAuth2) in production environments.

## Project layout

```
finrag-agent/
cli/                Command-line client (`finrag`)
src/
  api/
    routes/         FastAPI route handlers (chat, auth, admin, threads, roles, mcp)
    dependencies.py Auth + DB session dependencies
    main.py         FastAPI app, CORS, lifespan, startup validator
  config/           Settings, RBAC config, prompt templates
  governance/       Model Risk Governance (SR 11-7 / OCC 2011-12 validation hooks)
  graph/
    nodes/          20 LangGraph nodes including decision_engine, mcp_tools, research_agent
    edges.py        Conditional edge routing functions
    builder.py      StateGraph construction and compilation
  ingestion/        Document pipeline (parser -> chunk -> embed -> Qdrant)
  models/           Pydantic models (RAGState, schemas, auth, governance)
  services/         llm_factory, decider_2b (SLM), market_data_service,
                    sec_edgar_service, embeddings, vector_store, reranker_service,
                    guardrails_service, ltr_gate_service, candidate_validator,
                    cost_tracker, request_context, result_cache, llm_retry,
                    roles_service, thread_service
  tools/            AST-restricted calculator, quantitative investment metrics,
                    decision_models (VaR/CVaR, Ledoit-Wolf, Fama-French),
                    Model Context Protocol (MCP) server

migrations/         Alembic — roles table + system role seed
tests/
  unit/             Unit tests (mocked LLMs, quantitative models, MRG checks)
  integration/      Integration tests
  evaluation/       RAGAS + DeepEval + correctness pipeline benchmarks

scripts/            Data download, ingestion, JWT generation,
                    reranker training data builder, LoRA training script,
                    failure-analysis utilities
data/
  sample/           8 curated sample financial PDFs (committed)
  training/         Reranker training data manifests + jsonl
```

## Key environment variables

See [`.env.example`](../.env.example) for the full list.

| Variable | Required | Description |
|---|---|---|
| `LLM_PROVIDER_TYPE` | no | `hybrid` (default), `open_source` (Ollama/vLLM), `closed_source`, or `placeholder` |
| `OPEN_SOURCE_LLM_BASE_URL` | no | Base URL for open-source LLM server (default: `http://localhost:11434/v1`) |
| `OPEN_SOURCE_LLM_MODEL` | no | Model name for open-source runner (e.g. `llama3.3:70b`, `mistral`, `qwen2.5`) |
| `DECIDER_2B_BASE_URL` | no | Base URL for local 2B parameter SLM routing engine (default: `http://localhost:11434/v1`) |
| `DECIDER_2B_MODEL` | no | Model identifier for System 1 fast-path decider (default: `strands-decider:2b`) |
| `MARKET_DATA_PROVIDER` | no | Market data provider: `auto`, `direct`, `sec_edgar`, or `fallback` |
| `SEC_EDGAR_USER_AGENT` | conditional | Required by SEC EDGAR fair-access policy (`SampleApp user@domain.com`) |
| `OPENAI_API_KEY` | conditional | OpenAI API key (for OpenAI embeddings or closed-source models) |
| `ANTHROPIC_API_KEY` | conditional | Anthropic API key (for Claude Sonnet generator) |
| `GROQ_API_KEY` | no | Free routing/grading; falls back to OpenAI or placeholder |
| `VOYAGE_API_KEY` | no | voyage-finance-2 embeddings (canonical for financial evaluation) |
| `EMBEDDING_PROVIDER` | no | `openai` (default), `voyage` (canonical), `abaci` (experimental) |
| `RERANKER_ADAPTER_PATH` | no | LoRA adapter path; empty preserves stock BGE |
| `LITELLM_URL` | no | LiteLLM gateway URL; empty bypasses gateway (direct-provider mode) |
| `JWT_SECRET` | yes (prod) | Must be changed from default in production |
| `POSTGRES_PASSWORD` | yes (prod) | Must be changed from default in production |
| `ENVIRONMENT` | no | `dev` (default), `staging`, or `production` |
| `CORS_ORIGINS` | no | JSON array of allowed origins |
