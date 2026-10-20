# Asset Management Role-Based Access Control (RBAC) & Governance Matrix

Role-based access control is enforced **at the vector database layer** via Qdrant payload filters — unauthorized documents are never retrieved, preventing any risk of prompt leakage or model context contamination.

---

## Role → Permissions Matrix

Source: [src/config/rbac_config.py](../src/config/rbac_config.py)

| Role | Allowed Document Types | Confidentiality Levels | Top-K Chunks | HITL Allocation Threshold | Approval Authority |
|------|------------------------|------------------------|--------------|---------------------------|-------------------|
| `research_analyst` | `10k`, `10q`, `earnings_transcript`, `equity_research` | `public`, `internal` | 10 | $100,000 | None (no self-approval) |
| `portfolio_manager` | `10k`, `10q`, `earnings_transcript`, `equity_research`, `portfolio_holdings` | `public`, `internal`, `confidential` | 15 | $1,000,000 | `research_analyst` |
| `risk_compliance` | `10k`, `10q`, `earnings_transcript`, `equity_research`, `portfolio_holdings` | `public`, `internal`, `confidential` | 20 | None (mandate oversight) | `portfolio_manager`, `research_analyst` |
| `admin` | `*` (all) | `*` (all) | 25 | None (system administrator) | Universal (`*`) |

*Note: Backward-compatible aliases (`analyst`, `finance`, `hr`, `c_level`) are maintained in `rbac_config.py` for legacy compatibility.*

---

## Quantitative Tools & MCP Authorization Matrix

Institutional tools exposed via Model Context Protocol (MCP) enforce strict role gating before execution:

| Quantitative Tool Name | Function / Description | Authorized Roles |
|---|---|---|
| `calculate_valuation_multiples` | P/E, EV/EBITDA, P/B, FCF yield metrics | `research_analyst`, `portfolio_manager`, `risk_compliance`, `admin` |
| `calculate_profitability_metrics` | Gross/Operating/Net margins, ROE, ROA, D/E | `research_analyst`, `portfolio_manager`, `risk_compliance`, `admin` |
| `calculate_growth_metrics` | YoY/QoQ revenue & EPS growth, margin expansion bps | `research_analyst`, `portfolio_manager`, `risk_compliance`, `admin` |
| `calculate_portfolio_risk_metrics` | Herfindahl-Hirschman Index (HHI) concentration & Top-N weights | `research_analyst`, `portfolio_manager`, `risk_compliance`, `admin` |
| `calculate_portfolio_var_cvar` | Parametric & Historical VaR/CVaR at 95%/99% confidence (1d/10d/1y) | `portfolio_manager`, `risk_compliance`, `admin` |
| `optimize_portfolio_weights` | Ledoit-Wolf shrinkage mean-variance optimizer (Max Sharpe, Min Vol) | `portfolio_manager`, `risk_compliance`, `admin` |
| `calculate_factor_exposures` | Fama-French multi-factor risk attribution & alpha/beta sensitivities | `research_analyst`, `portfolio_manager`, `risk_compliance`, `admin` |
| `get_sec_filings` | Automated SEC EDGAR XBRL ingestion & dynamic CIK facts extraction | `research_analyst`, `portfolio_manager`, `risk_compliance`, `admin` |

---

## Document Types & Taxonomy

| Document Type | Investment Scope & Contents | Typical Confidentiality |
|---------------|-----------------------------|-------------------------|
| `10k` | SEC Annual Filings — audited financials, MD&A, risk factors, note disclosures | `public` |
| `10q` | SEC Quarterly Reports — interim financial statements, quarterly momentum | `public` |
| `earnings_transcript` | CEO/CFO prepared remarks and Wall Street sell-side analyst Q&A | `public` |
| `equity_research` | Proprietary DCF models, investment memos, price targets, thesis briefs | `internal` |
| `portfolio_holdings` | Live mandate weights, active cash positions, position sizing, concentration metrics | `confidential` |

---

## Confidentiality Levels

Classified during ingestion ([src/ingestion/metadata_extractor.py](../src/ingestion/metadata_extractor.py)) and validated at retrieval time:

- **`public`**: Regulatory disclosures, EDGAR filings, press releases, public call transcripts.
- **`internal`**: Proprietary equity research memos, internal valuation summaries, meeting notes.
- **`confidential`**: Portfolio holdings, active trade strategies, risk analytics, and blackout mandates.

---

## Human-in-the-Loop (HITL) Governance & Compliance Gates

Asset management workflows handle high-stakes fiduciary decisions. The agent enforces two layers of deterministic gates before emitting investment outputs:

### 1. Financial Allocation Thresholds
- Queries referencing financial figures exceeding a role's threshold (`$100,000` for analysts, `$1,000,000` for PMs) trigger LangGraph `interrupt()`.
- The execution state is safely persisted in the checkpointer (`PostgreSQL` / `MemorySaver`) until reviewed.

### 2. Regulatory & Compliance Monitoring
The gate scans queries and generated investment notes for:
- **Restricted List Violations**: Mentions of companies or tickers currently under blackout or restricted list status.
- **Material Non-Public Information (MNPI)**: Triggers an immediate hold if unreleased insider indicators are flagged.
- **Portfolio Concentration Warnings**: Flags portfolios exceeding regulatory concentration rules (e.g. UCITS 5/10/40 or 1940 Act limits).

### 3. Multi-Level Approval Hierarchy
Enforced by `can_approve(approver_role, requester_role)`:
- **Research Analysts** cannot approve their own allocation proposals.
- **Portfolio Managers** are authorized to approve analyst proposals.
- **Risk & Compliance Officers** hold sign-off authority over PM allocations and analyst models.
- **Admins** hold infrastructure-level override capabilities.

---

## Storage-Layer Enforcement

1. **`rbac_gate` Node**: Resolves the user's role and retrieves authorized `allowed_doc_types` and `allowed_confidentiality`.
2. **`build_rbac_filter()`**: Constructs a deterministic Qdrant `Filter` with `FieldCondition` clauses for `doc_type` and `confidentiality`.
3. **Index-Level Pruning**: HNSW search only executes over points matching the RBAC filter payload. Unauthorized data never touches the LLM context window.

---

## Development Test Credentials

Defined in [src/api/routes/auth.py](../src/api/routes/auth.py):

| Username | Password | Role | Asset Management Function |
|----------|----------|------|---------------------------|
| `analyst` | `analyst123` | `research_analyst` | Fundamental Equity / Credit Research |
| `pm` | `pm123` | `portfolio_manager` | Strategy, Portfolio Allocation & Execution |
| `compliance` | `compliance123` | `risk_compliance` | Mandate Limits, MNPI & Regulatory Oversight |
| `admin` | `admin123` | `admin` | Platform Engineering & MLOps |
