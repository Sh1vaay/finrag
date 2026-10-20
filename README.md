# FinRAG: Institutional Multi-Agent Investment Research & Quantitative Decision Platform

[![CI Matrix](https://img.shields.io/badge/CI-Passing-brightgreen.svg)](.github/workflows/ci.yml)
[![Python 3.11 | 3.12](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-20--Node%20Topology-000000.svg?logo=langchain&logoColor=white)](https://github.com/langchain-ai/langgraph)
[![Model Context Protocol](https://img.shields.io/badge/MCP-JSON--RPC%202.0-8A2BE2.svg)](https://modelcontextprotocol.io/)
[![Qdrant Hybrid Search](https://img.shields.io/badge/Qdrant-Hybrid%20Dense%20%2B%20BM25-DC2626.svg?logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Model Risk Governance](https://img.shields.io/badge/MRG-Fed%20SR%2011--7%20%2F%20OCC%202011--12-EAB308.svg)](docs/mrg_validation_report.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

An institutional-grade, multi-agent AI investment research and quantitative decision platform engineered for buy-side asset managers, hedge funds, sovereign wealth funds, and quantitative research teams. Powered by **LangGraph**, **Model Context Protocol (MCP)**, **Hybrid Dense-Sparse Vector Retrieval**, **Cross-Encoder Neural Reranking**, and **Deterministic Quantitative Models**, FinRAG automates fundamental equity analysis, SEC EDGAR XBRL ingestion, portfolio risk estimation (VaR/CVaR, Ledoit-Wolf shrinkage, factor attribution), and fiduciary compliance governance.

---

## Table of Contents

- [Executive Overview & Problem Statement](#executive-overview--problem-statement)
- [System Architecture](#system-architecture)
- [Request Lifecycle & Data Flow](#request-lifecycle--data-flow)
- [Core Platform Capabilities](#core-platform-capabilities)
- [Quantitative Decision Suite & Risk Governance (Fed SR 11-7)](#quantitative-decision-suite--risk-governance-fed-sr-11-7)
- [Model Context Protocol (MCP) Server Specification](#model-context-protocol-mcp-server-specification)
- [Buy-Side Role-Based Access Control (RBAC)](#buy-side-role-based-access-control-rbac)
- [Human-in-the-Loop (HITL) Fiduciary Governance](#human-in-the-loop-hitl-fiduciary-governance)
- [REST & Protocol API Reference](#rest--protocol-api-reference)
- [Security Architecture & Threat Mitigation](#security-architecture--threat-mitigation)
- [Performance & Scalability Benchmarks](#performance--scalability-benchmarks)
- [Observability, Tracing & Cost Accounting](#observability-tracing--cost-accounting)
- [Technology Stack](#technology-stack)
- [Installation & Local Development](#installation--local-development)
- [Production Deployment & Hardening](#production-deployment--hardening)
- [Contributing Guidelines](#contributing-guidelines)
- [License](#license)

---

## Executive Overview & Problem Statement

### The Institutional Challenge
Front-office investment analysts, risk officers, and portfolio managers face four fundamental roadblocks when deploying generative AI in institutional asset management:

1. **Arithmetic & Formula Hallucination**: Generative language models are probabilistic text predictors. They routinely hallucinate financial multiples, miscalculate accounting restatements, and invert risk metrics (e.g. producing negative VaR).
2. **Context Window Saturation & Entity Drift**: Regulatory filings (10-Ks, 10-Qs, 8-Ks) span hundreds of dense pages with complex multi-year financial tables. Standard RAG pipelines confuse fiscal years, mix entity facts, and fail on tabular comparisons.
3. **Data Contamination & Entitlement Breaches**: Asset managers operate under fiduciary information barriers. Private mandate holdings, restricted watchlists, and confidential investment memos must never leak across departments or unauthorized roles.
4. **Model Risk Governance Mandates**: Under **Federal Reserve SR 11-7** and **OCC 2011-12**, models used in investment decision-making must undergo strict conceptual soundness testing, input boundary validation, matrix conditioning checks, and deterministic audit tracking.

### The FinRAG Solution
- **Deterministic Offloading**: All numerical calculations, portfolio optimizations, and risk simulations are strictly offloaded from generative LLMs to AST-isolated, unit-tested quantitative mathematical engines.
- **Dual-System Cognitive Architecture**: A fast System 1 Small Language Model (SLM) triages queries in sub-50ms, while a System 2 deep research LangGraph orchestrator decomposes complex multi-statement inquiries across 20 specialized nodes.
- **Storage-Layer RBAC Filtering**: Role entitlements are enforced directly within Qdrant vector index payload filters. Unauthorized chunks are physically excluded from index scans and never touch LLM context windows.
- **Automated SR 11-7 Validation Battery**: Built-in automated governance validators continuously benchmark mathematical outputs, test boundary conditions, check covariance matrix conditioning, and emit institutional audit dossiers.

---

## System Architecture

FinRAG is structured around an institutional, multi-layered architecture designed for sub-second financial reasoning, deterministic mathematical accuracy, and strict fiduciary information barriers.

### Architectural Component Decomposition

```mermaid
flowchart TD
    %% ==========================================
    %% 1. CLIENT & PROTOCOL LAYER
    %% ==========================================
    subgraph FrontendComponents["1. Frontend & Client Surface"]
        CLI["finrag CLI\n(Typer · Prompt Toolkit · Rich TUI · Slash Commands)"]
        ClaudeDesktop["Claude Desktop / External LLM\n(Model Context Protocol JSON-RPC Client)"]
        WebDashboard["Front-Office UI / BI Systems\n(REST & SSE API Consumers)"]
    end

    %% ==========================================
    %% 2. API & GATEWAY LAYER
    %% ==========================================
    subgraph APIGatewayLayer["2. API Gateway & Ingress Layer (/v1)"]
        FastAPIApp["FastAPI Application Gateway\n(Asynchronous Uvicorn / ASGI Server)"]
        LiteLLMGateway["LiteLLM Proxy & Load Balancer\n(Provider Router & Semantic Cache)"]
        
        subgraph Endpoints["Exposed REST & RPC Endpoints"]
            AuthEP["/v1/auth (login, me)"]
            ChatEP["/v1/chat (SSE streaming, sync)"]
            HITLEP["/v1/approvals (review, approve)"]
            DocsEP["/v1/documents (PDF citations)"]
            IngestEP["/v1/ingest (upload, vectorize)"]
            MCPEP["/v1/mcp (JSON-RPC 2.0 tools)"]
            AdminEP["/v1/admin (audit, roles, costs)"]
        end
    end

    %% ==========================================
    %% 3. SECURITY & BACKEND SERVICES
    %% ==========================================
    subgraph BackendServices["3. Core Backend Services & Guardrails"]
        AuthService["Auth & RBAC Service\n(Timing-Safe HS256 JWT & Role Claims)"]
        
        subgraph GuardrailCascade["3-Tier Guardrails Cascade"]
            G_L1["Layer 1: Heuristic Regex\n(Prompt Injection & Keyword Filter)"]
            G_L2["Layer 2: ONNX Runtime Scanner\n(CPU-Optimized LLM Guard Scanner)"]
            G_L3["Layer 3: Presidio & spaCy NER\n(PII Detection & Deanonymization)"]
        end

        CognitiveTriage["Dual-System Cognitive Triage"]
        subgraph TriageEngines["Cognitive Triage Engines"]
            Sys1["System 1: Strand Decider 2B SLM\n(Sub-50ms Fast Intent Classifier)"]
            Sys2["System 2: Groq LLaMA-3.3-70B\n(Adaptive Decomposing Router)"]
        end

        QuantService["Deterministic Quantitative Decision Engine\n(AST Sandbox · Covariance Shrinkage · VaR/CVaR)"]
        IngestionService["Document Ingestion & Contextual Chunker\n(FastEmbed BM25 · Dense Vectorizer · Docling)"]
    end

    %% ==========================================
    %% 4. LANGGRAPH 20-NODE ORCHESTRATION ENGINE
    %% ==========================================
    subgraph LangGraphDAG["4. LangGraph 20-Node Orchestration Engine"]
        StartNode(["START"])
        RBACGateNode["rbac_gate"]
        GuardNode["guardrails"]
        ExtractorNode["entity_extractor"]
        RouterNode{"router\n(System 1 / 2)"}
        
        %% Branches
        DirectNode["direct_response\n/ terminal_nodes"]
        MCPToolsNode["mcp_tools\n(Tool Execution)"]
        ResearchNode["research_agent\n(Sub-query Decompose)"]
        
        subgraph RetrievalLoop["Adaptive Retrieval Loop"]
            RetrievalNode["retrieval\n(Hybrid RRF k=60)"]
            RerankNode["reranker\n(BGE Cross-Encoder)"]
            EvalNode["retrieval_evaluator\n(LTR Gating)"]
            GraderNode["grader\n(3-Stage Relevancy)"]
            RewriterNode["query_rewriter\n(Context Reformulation)"]
        end

        DecisionNode["decision_engine\n(Quantitative Models)"]
        GeneratorNode["generator\n(Grounded Claude Sonnet)"]
        HallucinationNode["hallucination_checker\n(Faithfulness Audit)"]
        HITLNode{"hitl_gate\n(Fiduciary Check)"}
        FormatterNode["response_formatter\n(Audited Citations)"]
        EndNode(["END"])
    end

    %% ==========================================
    %% 5. DATABASE & STORAGE SUBSTRATE
    %% ==========================================
    subgraph DatabaseLayer["5. Database & Storage Substrate"]
        QdrantDB[("Qdrant Vector Database\n- Dense Vectors (1536-dim)\n- BM25 Sparse Vectors\n- Storage-Layer RBAC Payload Filters")]
        PostgresDB[("PostgreSQL 16 Database\n- LangGraph Thread State Checkpoints\n- System Roles & Entitlements\n- Conversation History")]
        RedisCache[("Redis 7.2 Database\n- Sub-Component Result Cache (TTL 7d)\n- LiteLLM Semantic Cache")]
    end

    %% ==========================================
    %% 6. EXTERNAL INTEGRATIONS & PROVIDERS
    %% ==========================================
    subgraph ExternalIntegrations["6. External Integrations & Model APIs"]
        SEC_EDGAR["SEC EDGAR XBRL Audited API\n(CIK Resolver · 10-K/10-Q Cache)"]
        MarketDataFeed["Institutional Market Data Feed\n(B-PIPE / FactSet Simulation & yfinance)"]
        AnthropicAPI["Anthropic API\n(Claude Sonnet 4.6 / Haiku 4.5)"]
        OpenAIAPI["OpenAI API\n(GPT-4o / GPT-4o-mini / Embeddings)"]
        GroqAPI["Groq Cloud API\n(LLaMA-3.3-70B Fast-Path)"]
        LocalSLM["Local SLM Host\n(Ollama / vLLM Endpoint)"]
    end

    %% ==========================================
    %% 7. MLOPS & OBSERVABILITY
    %% ==========================================
    subgraph ObservabilityStack["7. Enterprise MLOps & Observability"]
        Phoenix["Arize Phoenix\n(OpenTelemetry Distributed Traces & Spans)"]
        Langfuse["Langfuse v3\n(Per-User Token & Cost Attribution)"]
        AuditLedger["Audit Trail Ledger\n(Append-Only JSONL Event Logs)"]
    end

    %% Connections
    FrontendComponents --> APIGatewayLayer
    APIGatewayLayer --> BackendServices
    BackendServices --> LangGraphDAG
    
    %% Ingress to DAG
    StartNode --> RBACGateNode --> GuardNode --> ExtractorNode --> RouterNode
    RouterNode -->|Direct Lookup| DirectNode --> EndNode
    RouterNode -->|Quantitative Tool| MCPToolsNode --> DecisionNode
    RouterNode -->|Multi-Hop Research| ResearchNode --> DecisionNode
    RouterNode -->|Document Retrieval| RetrievalNode
    
    RetrievalNode --> RerankNode --> EvalNode --> GraderNode
    GraderNode -->|Relevant Chunks| DecisionNode
    GraderNode -->|Needs Refinement| RewriterNode --> RetrievalNode
    
    DecisionNode --> GeneratorNode --> HallucinationNode --> HITLNode
    HITLNode -->|Approved / Low Risk| FormatterNode --> EndNode
    HITLNode -->|High Stakes Interrupt| PostgresDB
    
    %% Storage & Integrations
    RetrievalNode <--> QdrantDB
    IngestionService --> QdrantDB
    LangGraphDAG <--> PostgresDB
    LangGraphDAG <--> RedisCache
    QuantService <--> MarketDataFeed
    QuantService <--> SEC_EDGAR
    GeneratorNode <--> AnthropicAPI
    ResearchNode <--> OpenAIAPI
    RouterNode <--> GroqAPI
    Sys1 <--> LocalSLM
    
    %% Observability
    FastAPIApp --> AuditLedger
    LangGraphDAG --> Phoenix
    LangGraphDAG --> Langfuse
```

---

## Authentication & Authorization Flow

FinRAG implements timing-safe authentication and strict storage-level Role-Based Access Control (RBAC). Document entitlements are applied directly to Qdrant vector database query filters, guaranteeing that unauthorized chunks are physically omitted from vector index traversals and never enter the LLM context window.

```mermaid
sequenceDiagram
    autonumber
    actor User as Client (CLI / Front-Office UI)
    participant Auth as POST /v1/auth/login
    participant AuthService as AuthService
    participant RolesDB as PostgreSQL (roles table)
    participant API as Authenticated API (/v1/chat)
    participant Qdrant as Qdrant Hybrid Storage

    User->>Auth: Submit username & password
    Auth->>AuthService: Timing-safe credential verification
    AuthService->>RolesDB: Query role entitlements (allowed_doc_types, confidentiality, limits)
    RolesDB-->>AuthService: Return role permission matrix
    AuthService-->>Auth: Issue HS256 JWT Token (Claims: user_id, role, department, exp)
    Auth-->>User: TokenResponse (access_token, user metadata, permissions)

    Note over User,API: Subsequent requests attach Authorization: Bearer <JWT> header

    User->>API: POST /v1/chat (Financial Query + Bearer JWT)
    API->>AuthService: Decode & verify token signature + expiration
    AuthService-->>API: Active user context (Role: portfolio_manager, Dept: Asset Management)
    
    API->>Qdrant: Build storage payload filter based on user role
    Note over Qdrant: Filter: doc_type in ['10k', '10q', 'portfolio_holdings']<br/>AND confidentiality in ['public', 'internal', 'confidential']
    Qdrant-->>API: Return physically entitled document candidate chunks only
```

---

## Request Lifecycle & Data Flow

Every investment research query follows an audited multi-stage path with 3-tier security screening, dual-system cognitive triage, deterministic mathematical offloading, hallucination verification, and human-in-the-loop authorization gates.

```mermaid
sequenceDiagram
    autonumber
    actor Analyst as Buy-Side Analyst
    participant API as FastAPI Gateway
    participant Guard as Safety & Guardrails Cascade
    participant Sys1 as System 1 SLM Triage (Strand 2B)
    participant Graph as LangGraph 20-Node DAG
    participant Retrieval as Hybrid Retrieval & BGE Reranker
    participant Quant as Deterministic Decision Suite
    participant Gen as Claude Sonnet Generator
    participant Verifier as Hallucination & Citation Checker
    participant HITL as HITL Fiduciary Gate
    actor PM as Portfolio Manager (Approver)

    Analyst->>API: POST /v1/chat (Financial Query + Bearer JWT)
    API->>Guard: Validate HS256 Token & Role Entitlements
    
    rect rgb(240, 248, 255)
        Note over Guard: 3-Tier Security Screening
        Guard->>Guard: Layer 1: Heuristic Regex Prompt Injection Check
        Guard->>Guard: Layer 2: ONNX Runtime CPU Scanner (LLM Guard)
        Guard->>Guard: Layer 3: Presidio / spaCy PII Analyzer & Anonymizer
    end

    alt Malicious Injection or Policy Breach
        Guard-->>API: 403 Forbidden / Sanitized Rejection
        API-->>Analyst: Structured Security Rejection
    else Clean Query
        Guard->>Sys1: Fast-Path Intent & Complexity Triage (<50ms)
        
        alt Quantitative / Arithmetic Request
            Sys1->>Quant: AST-Sandboxed Quantitative Engine
            Note over Quant: Calculate VaR / CVaR, Ledoit-Wolf Shrinkage, Altman Z'', Factor OLS
            Quant-->>Graph: Verified Math Output + SR 11-7 Metadata
        else Deep Filings Research & Multi-Hop Query
            Sys1->>Graph: LangGraph Multi-Agent Research Topology
            Graph->>Retrieval: Hybrid Search (Dense 1536-dim + BM25 Sparse, Top-50)
            Retrieval-->>Graph: BGE-Reranker-v2-m3 Cross-Encoder Candidates (Top-8)
            Graph->>Graph: 3-Stage Document Relevance Grader
            Graph->>Gen: Synthesize Response with Audited Citations
            Gen-->>Graph: Grounded Candidate Answer
        end

        Graph->>Verifier: Hallucination & Faithfulness Audit
        Verifier-->>Graph: Citation Verified & Numerically Grounded

        Graph->>HITL: Fiduciary Risk Check ($100k+ / MNPI / Restricted List)
        alt Fiduciary Threshold Exceeded ($100k+ or Restricted Keyword)
            HITL->>Graph: Trigger interrupt() & Save Frozen State to PostgreSQL
            Graph-->>API: Status 200 OK (paused: pending_approval)
            API-->>Analyst: Query Paused: Awaiting Portfolio Manager Authorization
            PM->>API: POST /v1/hitl/approve (Thread ID + PM Authorization)
            API->>Graph: Resume Execution from PostgreSQL Checkpoint
        end

        Graph-->>API: Formatted Answer + Audited Source Citations + Latency/Cost
        API-->>Analyst: Server-Sent Events (SSE) Token Stream or JSON ChatResponse
    end
```

---

## Core Platform Capabilities

- **20-Node LangGraph State Orchestration**: Stateful DAG execution topology with conditional looping for query reformulation, multi-candidate document grading, reflection, hallucination suppression, and human authorization interrupts.
- **Dual-System Cognitive Routing**: System 1 local 2B SLM resolves pricing, arithmetic, and basic ratio requests in <50ms with zero token cost, reserving frontier models (Claude Sonnet 4.6, GPT-4o) for complex multi-statement research.
- **Official Model Context Protocol (MCP) Server**: Full JSON-RPC 2.0 implementation exposing institutional quantitative calculations, factor models, and filings analysis to Claude Desktop and external autonomous agents.
- **Hybrid Dense-Sparse Retrieval**: OpenAI / Voyage dense vectors combined with fastembed BM25 sparse keyword representations fused via Reciprocal Rank Fusion ($k=60$) across top-50 candidates, refined to top-8 by `bge-reranker-v2-m3`.
- **AST-Sandboxed Deterministic Arithmetic**: Mathematical expressions evaluate via an Abstract Syntax Tree (AST) parser whitelisting basic operations and rejecting execution of arbitrary code, imports, attributes, and dunders.
- **Storage-Layer RBAC Segregation**: Entitlement filters execute inside Qdrant index queries, guaranteeing unauthorized document chunks never touch application RAM or LLM context windows.
- **SEC EDGAR Audited XBRL Pipeline**: Direct integration with SEC EDGAR REST APIs, CIK resolution, local caching, and financial statement extraction from audited 10-K and 10-Q reports.
- **Enterprise MLOps & Observability**: Distributed OpenTelemetry tracing via Arize Phoenix and token/cost accounting via Langfuse v3 with per-user spend tracking.

---

## Quantitative Decision Suite & Risk Governance (Fed SR 11-7)

FinRAG implements an audited mathematical decision suite aligned with **Federal Reserve SR 11-7** and **OCC 2011-12** standards:

| Quantitative Engine | Mathematical Formulation | Regulatory / Mandate Function | Authorized Roles |
|---|---|---|---|
| **Parametric VaR** | $\text{VaR}_\alpha = -V_0 (\mu \Delta t + z_\alpha \sigma \sqrt{\Delta t})$ | Baseline market risk across 1d, 10d (Basel III), and 1y horizons at 95% and 99% confidence. | `portfolio_manager`, `risk_compliance`, `admin` |
| **Conditional VaR (CVaR)** | $\text{CVaR}_\alpha = \mathbb{E}[L \mid L \ge \text{VaR}_\alpha]$ | Expected Shortfall quantifying tail risk severity beyond VaR threshold. | `portfolio_manager`, `risk_compliance`, `admin` |
| **Ledoit-Wolf Covariance Optimizer** | $\hat{\Sigma}_{\text{LW}} = \hat{\delta} F + (1 - \hat{\delta}) S$ | Shrinkage target towards identity; guarantees positive-definite covariance in ill-conditioned sample spaces. Simplex projection with `max_single_cap`. | `portfolio_manager`, `risk_compliance`, `admin` |
| **Fama-French Factor Attribution** | $R_{it} - R_{ft} = \alpha_i + \sum_{k=1}^K \beta_{ik} F_{kt} + \epsilon_{it}$ | Multi-factor attribution across market, size (SMB), value (HML), momentum, and quality. | `research_analyst`, `portfolio_manager`, `risk_compliance`, `admin` |
| **Herfindahl-Hirschman Index (HHI)** | $\text{HHI} = \sum_{i=1}^N (w_i \times 100)^2$ | Flags portfolio concentration risk (<1,500 diversified, >2,500 concentrated) and validates UCITS 5/10/40 rule compliance. | `research_analyst`, `portfolio_manager`, `risk_compliance`, `admin` |
| **Piotroski F-Score & Altman Z-Score** | 9-factor accounting momentum & multi-ratio distress classification | Evaluates fundamental balance sheet quality, earnings momentum, and credit distress likelihood from SEC filings. | `research_analyst`, `portfolio_manager`, `risk_compliance`, `admin` |

Execute the automated model risk validation battery via:
```bash
python scripts/run_mrg_audit.py --output docs/mrg_validation_report.md
```

---

## Model Context Protocol (MCP) Server Specification

FinRAG implements the official **Model Context Protocol (MCP)** specification over JSON-RPC 2.0:

### Protocol Endpoints
- `POST /v1/mcp`: Official JSON-RPC 2.0 protocol endpoint
- `GET /v1/mcp/tools`: Tool discovery catalog (scoped to caller's RBAC role)
- `POST /v1/mcp/tools/{tool_name}`: Direct REST execution endpoint

### Example: Calling Portfolio VaR/CVaR via MCP JSON-RPC 2.0
```json
{
  "jsonrpc": "2.0",
  "id": "req-001",
  "method": "tools/call",
  "params": {
    "name": "calculate_portfolio_var_cvar",
    "arguments": {
      "weights": [0.4, 0.3, 0.3],
      "returns_matrix": [
        [0.012, -0.015, 0.008],
        [-0.005, 0.021, -0.010],
        [0.018, -0.004, 0.015]
      ],
      "portfolio_value": 10000000.0,
      "confidence_level": 0.99,
      "horizon_days": 10
    }
  }
}
```

---

## Buy-Side Role-Based Access Control (RBAC)

Access control enforces strict buy-side investment boundaries:

| Role | Permitted Document Types | Confidentiality Levels | Max Chunks | HITL Allocation Threshold | Approval Authority |
|---|---|---|---|---|---|
| `research_analyst` | `10k`, `10q`, `earnings_transcript`, `equity_research` | `public`, `internal` | 10 | $100,000 | None (no self-approval) |
| `portfolio_manager` | `10k`, `10q`, `earnings_transcript`, `equity_research`, `portfolio_holdings` | `public`, `internal`, `confidential` | 15 | $1,000,000 | `research_analyst` |
| `risk_compliance` | `10k`, `10q`, `earnings_transcript`, `equity_research`, `portfolio_holdings` | `public`, `internal`, `confidential` | 20 | None (auditor role) | `portfolio_manager`, `research_analyst` |
| `admin` | `*` (universal) | `*` (universal) | 25 | None | Universal (`*`) |

---

## Human-in-the-Loop (HITL) Fiduciary Governance

Asset management mandates require multi-party human verification for high-value allocations and sensitive events:
1. **Financial Allocation Ceilings**: Any proposal or investment model referencing amounts exceeding a role's threshold (`$100,000` for analysts, `$1,000,000` for PMs) triggers a LangGraph `interrupt()`.
2. **Compliance & Regulatory Filters**:
   - **Restricted List Mentions**: Securities currently under blackout or restricted list status.
   - **Material Non-Public Information (MNPI)**: Triggers an immediate hold if unreleased insider indicators are detected.
   - **Regulatory Concentration Limits**: Flags violations of UCITS 5/10/40 or US 1940 Act diversification tests.
3. **Approval Multi-Party Workflow**: State is securely frozen in PostgreSQL. Authorized approvers review proposals via the interactive CLI review inbox (`finrag approvals review`) or the REST API. Self-approval is strictly forbidden.

---

## REST & Protocol API Reference

Interactive Swagger documentation is available at `http://localhost:8000/docs`.

| Category | Method | Path | Description | Auth Required |
|---|---|---|---|---|
| **Auth** | `POST` | `/v1/auth/login` | Exchange credentials for HS256 JWT | None |
| **Auth** | `GET` | `/v1/auth/me` | Fetch active user identity and RBAC permissions | Bearer JWT |
| **Chat** | `POST` | `/v1/chat` | Send investment query (supports SSE streaming) | Bearer JWT |
| **HITL** | `GET` | `/v1/approvals` | List pending review items for caller's role | Bearer JWT (Approver) |
| **HITL** | `POST` | `/v1/approvals/{id}/approve` | Approve or reject paused investment thread | Bearer JWT (Approver) |
| **Threads** | `GET` | `/v1/threads` | List historical conversation sessions | Bearer JWT |
| **Threads** | `GET` | `/v1/threads/{id}` | Inspect conversation state & checkpoints | Bearer JWT |
| **Documents** | `POST` | `/v1/documents/upload` | Ingest PDF filing into Qdrant collection | Bearer JWT (`admin`) |
| **Documents** | `GET` | `/v1/documents/{filename}` | RBAC-checked citation PDF viewer | Bearer JWT |
| **MCP** | `POST` | `/v1/mcp` | Unified JSON-RPC 2.0 MCP protocol endpoint | Bearer JWT |
| **MCP** | `GET` | `/v1/mcp/tools` | Discover available quantitative MCP tools | Bearer JWT |
| **Admin** | `GET` | `/v1/admin/costs` | Inspect per-user / per-model token expenditures | Bearer JWT (`admin`) |
| **Admin** | `GET` | `/v1/admin/roles` | Inspect and dynamically adjust role permissions | Bearer JWT (`admin`) |
| **Health** | `GET` | `/v1/health` | Service liveliness and database readiness probe | None |

---

## Security Architecture & Threat Mitigation

FinRAG implements defense-in-depth across the entire application lifecycle:

1. **Timing-Safe Authentication**: Login authentication evaluates passwords using Python's `secrets.compare_digest` to eliminate side-channel timing attacks.
2. **3-Layer Prompt Injection Defense**:
   - *Layer 1 (Sub-ms Regex)*: Rejects known jailbreak signatures, system prompt override tags, and delimiter manipulation.
   - *Layer 2 (ONNX CPU Classifier)*: Deep semantic injection classification using an ONNX-runtime model (`protectai/deberta-v3-base-prompt-injection-v2`).
   - *Layer 3 (PII Anonymization)*: spaCy NER pipeline redacts SSNs, credit card numbers, email addresses, and personal phone numbers before prompt synthesis.
3. **AST Execution Sandbox**: Mathematical expressions are evaluated without access to `__builtins__`, file descriptors, or network sockets.
4. **Non-Root Container Hardening**: The Docker production image runs under a dedicated `appuser` (UID 1000) with read-only root filesystems and drop-capabilities (`cap_drop: ALL`).
5. **Secret Hygiene**: Startup validators in `src/config/settings.py` block container initialization if default passwords (`devpassword`, placeholder JWT secrets) are detected in non-development environments.

---

## Performance & Scalability Benchmarks

- **Retrieval Latency**: Hybrid Qdrant retrieval completes in <15ms across 100k+ chunks using HNSW indexing and sparse BM25 payload indexes.
- **Neural Reranking Budget**: BAAI `bge-reranker-v2-m3` evaluates 25 candidate chunks in <180ms on CPU (sub-50ms on CUDA/MPS).
- **Sub-Component Caching**: Redis 7.2 caches deterministic sub-results (entity extraction, query routing, XBRL facts) with a 7-day TTL, eliminating redundant LLM calls on repeated queries.
- **Asynchronous Connection Pooling**: PostgreSQL connections are managed via `psycopg-pool` with non-blocking async I/O.
- **Horizontal Scaling**: The FastAPI application is fully stateless. State persists in PostgreSQL checkpoints, allowing horizontal scaling behind standard load balancers (AWS ALB, NGINX, Cloudflare).

---

## Observability, Tracing & Cost Accounting

- **OpenTelemetry Distributed Traces**: End-to-end spans capture latency across router, retrieval, reranker, synthesis, and quantitative execution via Arize Phoenix.
- **Per-User Cost Attribution**: Append-only cost logs (`cost_logs/cost_log.jsonl`) record exact prompt tokens, completion tokens, and model expenditures per query.
- **Tier-1 Boot Provenance**: Every application start emits an immutable component banner logging Git commit SHA, active models, vector collection, and security settings.

---

## Technology Stack

- **Orchestration**: LangGraph 1.x (20-Node StateGraph), LangChain Core 0.3+
- **Vector Database**: Qdrant 1.13.2 (hybrid dense HNSW + sparse BM25)
- **Neural Reranking**: BAAI `bge-reranker-v2-m3` via HuggingFace Sentence-Transformers
- **Protocols & Frameworks**: Model Context Protocol (MCP) JSON-RPC 2.0, FastAPI, SSE Starlette
- **State & Caching**: PostgreSQL 16 (LangGraph checkpoints & roles), Redis 7.2 (caching)
- **Quantitative Engine**: NumPy, SciPy (quadratic optimization), scikit-learn
- **Package Management**: `uv` (modern, ultra-fast Python package manager)
- **Observability**: Arize Phoenix, Langfuse v3, OpenTelemetry, Prometheus

---

## Installation & Local Development

### 1. Prerequisites
- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/) package manager
- Docker and Docker Compose

### 2. Clone & Virtual Environment Setup
```bash
git clone https://github.com/Sh1vaay/finrag.git
cd finrag

# Create virtual environment and install all dependencies via uv
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 3. Environment Configuration
```bash
cp .env.example .env
# Edit .env and supply your OPENAI_API_KEY, ANTHROPIC_API_KEY, and GROQ_API_KEY
```

### 4. Launch Services & Seed Data
```bash
# Start Docker substrate (Qdrant, PostgreSQL, Redis)
docker compose -f compose.minimal.yml up -d

# Seed the vector database with sample financial filings
uv run python scripts/seed_qdrant.py --sample

# Start the FastAPI server with hot-reload
make run
```

### 5. Launch CLI Interactive Session
```bash
# Login as research analyst (password: analyst123)
uv run finrag login -u analyst

# Launch interactive streaming chat
uv run finrag chat
```

### 6. Run Test Matrix
```bash
# Execute unit & integration test matrix
uv run pytest tests/unit/ -v

# Execute standalone quantitative test runner
uv run tests/run_all_unit_tests.py
```

---

## Production Deployment & Hardening

1. **Environment Mode**: Set `ENVIRONMENT=production` in `.env` to enforce runtime validators.
2. **Secret Rotation**: Rotate `JWT_SECRET` and `POSTGRES_PASSWORD` away from development defaults; startup validators block execution with default credentials.
3. **CORS Restrictions**: Explicitly define `CORS_ORIGINS` (e.g. `["https://app.yourfund.com"]`); wildcard `*` is rejected in production.
4. **SEC Fair Access Policy**: Declare your fund's legitimate user-agent in `SEC_EDGAR_USER_AGENT` (`FundName contact@fund.com`).
5. **Observability Stack**: Start the full 11-service observability stack via `docker compose up -d` to launch the Langfuse trace server.

---

## Contributing Guidelines

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/quantitative-model-expansion`).
3. Verify all tests and linters pass:
   ```bash
   uv run ruff check src/ tests/
   uv run pytest tests/unit/ -v
   uv run tests/run_all_unit_tests.py
   ```
4. Commit your changes with descriptive commit messages.
5. Submit a Pull Request.

---

## License

This project is licensed under the [MIT License](LICENSE).
