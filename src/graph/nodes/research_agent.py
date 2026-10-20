"""Research agent — selective subgraph for `research_required` queries.

Runs ONLY for queries the router classifies as `research_required` (calc with
formula, multi-section synthesis, comparative, applicability-judgment).
Lookup queries stay on the fast path.

Internal loop (turn-budgeted at 5 LLM-driven turns total):

  1. **decompose**       — breaks the query into 2–4 atomic sub-questions
                           and surfaces qualifiers + required quantities.
  2. **retrieve+grade**  — for each sub-question, calls the existing retrieval
                           pipeline (retrieval_node + reranker_node + grader_node)
                           via state-mutation so RBAC, entity filter, and
                           progressive relaxation all flow through unchanged.
                           Agent retrieval delegates to the existing path; it does
                           NOT call hybrid_search directly.
  3. **sufficiency**     — judges whether collected evidence covers every
                           required quantity. If not, emits one targeted
                           follow-up sub-question (subject to turn budget).
  4. **synthesize**      — writes a structured-markdown context block
                           that calls out qualifiers explicitly + lists each
                           required quantity with value + source. The main
                           generator consumes this alongside `relevant_chunks`
                           and produces the final answer; the hallucination
                           checker still grounds against `relevant_chunks`.

Output (merged into RAGState):
  - relevant_chunks: deduped union across all sub-question retrievals
  - grading_results: pre-marked "agent" so downstream router skips re-grading
  - agent_synthesis: structured markdown block (also flows into the generator)
  - agent_turns_used: int (for diagnostics)
  - agent_sub_questions: list[str]
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

# Path to project root
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = Any  # type: ignore

try:
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:
    class HumanMessage:  # type: ignore
        def __init__(self, content):
            self.content = content
    class SystemMessage:  # type: ignore
        def __init__(self, content):
            self.content = content

try:
    from pydantic import BaseModel, Field
except ImportError:
    class BaseModel:  # type: ignore
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    def Field(*args, **kwargs):  # type: ignore
        default = kwargs.get("default")
        default_factory = kwargs.get("default_factory")
        if default_factory is not None:
            return default_factory()
        return default

from src.config.prompts import (
    AGENT_SYNTHESIZER_SYSTEM_PROMPT,
    AGENT_SYNTHESIZER_SYSTEM_PROMPT_CALC,
    DECOMPOSE_SYSTEM_PROMPT,
    SUFFICIENCY_SYSTEM_PROMPT,
)
from src.config.settings import settings
from src.graph.nodes.grader import grader_node
from src.graph.nodes.reranker import reranker_node
from src.graph.nodes.retrieval import retrieval_node
from src.models.state import RAGState
from src.services.llm_factory import LLMFactory
from src.tools.calculator import CalculatorError, compute

logger = logging.getLogger(__name__)


MAX_LLM_TURNS = 5  # decompose + N retrievals + sufficiency + synthesize <= 5
MAX_SUB_QUESTIONS = 5
MAX_FOLLOWUP_ROUNDS = 2  # how many sufficiency-driven follow-ups to run


# ────────────────────────────────────────────────────────────
# Pydantic schemas for structured output (agent-internal only)
# ────────────────────────────────────────────────────────────


class _Decomposition(BaseModel):
    """Output of the decompose step."""

    qualifiers: list[str] = Field(
        description="Question qualifiers that change WHAT counts as a correct answer "
                    "(e.g. 'exclude M&A', 'organic', 'year over year direction'). "
                    "Use ['no qualifiers'] if none."
    )
    required_quantities: list[str] = Field(
        description="The 2+ inputs the question's answer requires (current assets, "
                    "current liabilities, operating income FY-1, etc)."
    )
    sub_questions: list[str] = Field(
        description="2–4 atomic sub-questions, each retrievable from a single section."
    )


class _SufficiencyVerdict(BaseModel):
    """Output of the sufficiency-judge step."""

    decision: str = Field(description="'sufficient' or 'need_more'")
    missing_quantity: str | None = Field(default=None)
    follow_up_question: str | None = Field(default=None)
    reason: str = Field(description="One sentence explaining the verdict.")


class _AgentSynthesis(BaseModel):
    """Output of the synthesize step (includes calculator arithmetic expression field)."""

    synthesis: str = Field(
        description="Structured-markdown context block for the generator."
    )
    arithmetic_expression: str | None = Field(
        default=None,
        description=(
            "Pure arithmetic expression to compute the numerical answer, OR null "
            "for non-numeric / lookup / not-found cases. Allowed: digits, '.', "
            "'+', '-', '*', '/', '//', '()'. No variables, functions, commas, "
            "units, or trailing '%'."
        ),
    )


# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────


def _claude_with_caching() -> ChatAnthropic:
    """Generator-tier Claude (Sonnet 4.6) — kept for back-compat.

    Task-specific factory calls allow each sub-step (decompose, sufficiency,
    synthesize) to be configured to a different model independently:

        LLMFactory.get_research_decompose_llm()
        LLMFactory.get_research_sufficiency_llm()
        LLMFactory.get_research_synthesize_llm()

    Note: cache_control on agent system prompts will start firing here once
    accumulated agent context exceeds 1024 tokens.
    """
    return LLMFactory.get_generator_llm()


def _system_block(prompt_text: str, llm) -> SystemMessage:
    """Anthropic cache_control marker on system text when llm is ChatAnthropic."""
    if isinstance(llm, ChatAnthropic):
        return SystemMessage(content=[{
            "type": "text",
            "text": prompt_text,
            "cache_control": {"type": "ephemeral"},
        }])
    return SystemMessage(content=prompt_text)


def _chunk_id(chunk: dict) -> tuple[str, Any]:
    """Stable identity key for dedup across sub-question retrievals."""
    meta = chunk.get("metadata", {})
    return (
        meta.get("source_file", "?"),
        meta.get("chunk_index", chunk.get("id", id(chunk))),
    )


def _evidence_summary(chunks: list[dict], max_chars_per: int = 400) -> str:
    """Compact summarization of collected chunks for the sufficiency prompt.

    The sufficiency judge doesn't need full chunks — just enough to confirm
    each required quantity is present. Truncating each chunk keeps the prompt
    bounded as the agent collects more evidence across turns.
    """
    if not chunks:
        return "(no evidence collected yet)"
    parts = []
    for i, c in enumerate(chunks, 1):
        meta = c.get("metadata", {})
        src = meta.get("source_file", "?")
        page = meta.get("page_number", "?")
        text = (c.get("raw_content") or c.get("content", ""))[:max_chars_per]
        parts.append(f"[{i}] {src} p{page}: {text}")
    return "\n".join(parts)


def _full_evidence(chunks: list[dict]) -> str:
    """Full chunk text for the synthesizer (no truncation)."""
    if not chunks:
        return "(no evidence collected)"
    parts = []
    for i, c in enumerate(chunks, 1):
        meta = c.get("metadata", {})
        src = meta.get("source_file", "?")
        page = meta.get("page_number", "?")
        text = c.get("raw_content") or c.get("content", "")
        parts.append(f"[Source {i}: {src}, Page {page}]\n{text}")
    return "\n\n---\n\n".join(parts)


def _retrieve_and_grade_for_subq(state: RAGState, sub_question: str) -> list[dict]:
    """Run the sub-question through the EXISTING retrieval/reranker/grader path.

    This is the RBAC-preservation lockdown: by mutating only `retrieval_query`
    on a copy of the parent state and routing through retrieval_node, we
    inherit RBAC filtering, target_company filter, year filter, and the
    progressive-relaxation backoff for free.
    """
    # Make a shallow sub-state with the sub-question as the retrieval query.
    sub_state = dict(state)
    sub_state["retrieval_query"] = sub_question

    # 1) retrieval
    retrieval_out = retrieval_node(sub_state)
    sub_state["retrieved_chunks"] = retrieval_out.get("retrieved_chunks", [])
    if not sub_state["retrieved_chunks"]:
        return []

    # 2) reranker
    rerank_out = reranker_node(sub_state)
    sub_state["reranked_chunks"] = rerank_out.get("reranked_chunks", [])
    sub_state["candidate_diagnostics"] = rerank_out.get("candidate_diagnostics", [])
    if not sub_state["reranked_chunks"]:
        return []

    # 3) grader (uses gpt-4o-mini, parallelized)
    grade_out = grader_node(sub_state)
    return grade_out.get("relevant_chunks", []) or []


# ────────────────────────────────────────────────────────────
# LLM-driven steps
# ────────────────────────────────────────────────────────────


def _decompose(query: str, target_company: str | None, target_fiscal_year: int | None) -> _Decomposition:
    """Turn 1 — decompose the query into atomic sub-questions.

    Uses the dedicated decompose-task LLM (configurable via
    `RESEARCH_AGENT_DECOMPOSE_MODEL`). Default: `claude-sonnet-4-6`.
    """
    llm = LLMFactory.get_research_decompose_llm()
    structured = llm.with_structured_output(_Decomposition)
    prompt = DECOMPOSE_SYSTEM_PROMPT.format(
        query=query,
        target_company=target_company or "(not specified)",
        target_fiscal_year=target_fiscal_year or "(not specified)",
    )
    try:
        result: _Decomposition = structured.invoke([
            _system_block(prompt, llm),
            HumanMessage(content=f"Decompose this question: {query}"),
        ])
        # Defensive truncation
        result.sub_questions = result.sub_questions[:MAX_SUB_QUESTIONS]
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Decompose failed, falling back to single-question agent: {exc}")
        return _Decomposition(
            qualifiers=["no qualifiers"],
            required_quantities=[query],
            sub_questions=[query],
        )


def _judge_sufficiency(
    query: str,
    decomposition: _Decomposition,
    chunks: list[dict],
) -> _SufficiencyVerdict:
    """Sufficiency check — does the evidence cover every required quantity?

    Uses the dedicated sufficiency-task LLM (configurable via
    `RESEARCH_AGENT_SUFFICIENCY_MODEL`). Default: `claude-sonnet-4-6`.
    """
    llm = LLMFactory.get_research_sufficiency_llm()
    structured = llm.with_structured_output(_SufficiencyVerdict)
    decomp_text = (
        f"Qualifiers: {decomposition.qualifiers}\n"
        f"Required quantities: {decomposition.required_quantities}\n"
        f"Sub-questions: {decomposition.sub_questions}"
    )
    prompt = SUFFICIENCY_SYSTEM_PROMPT.format(
        query=query,
        decomposition=decomp_text,
        n_chunks=len(chunks),
        evidence_summary=_evidence_summary(chunks),
    )
    try:
        verdict: _SufficiencyVerdict = structured.invoke([
            _system_block(prompt, llm),
            HumanMessage(content="Judge sufficiency now."),
        ])
        return verdict
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Sufficiency judge failed, defaulting to sufficient: {exc}")
        return _SufficiencyVerdict(
            decision="sufficient",
            reason=f"judge_error: {type(exc).__name__}",
        )


def _empty_calc_diagnostics() -> dict:
    return {
        "calculator_invoked": False,
        "calculator_expression": None,
        "calculator_value": None,
        "calculator_error": None,
    }


def _synthesize(
    query: str,
    decomposition: _Decomposition,
    chunks: list[dict],
) -> tuple[str, dict]:
    """Final turn — writes the structured context block for the generator.

    Dispatches on `settings.ENABLE_CALCULATOR_TOOL`:
      - False (default): plain-text path. The synthesizer returns a markdown
        string without the calculator tool.
      - True: structured-output path, calls the calculator on the LLM-emitted
        expression, appending verified arithmetic to the synthesis.

    Returns:
        (synthesis_markdown, calc_diagnostics) — same shape on both paths so
        the caller doesn't branch on the flag.
    """
    if settings.ENABLE_CALCULATOR_TOOL:
        return _synthesize_with_calculator(query, decomposition, chunks)
    return _synthesize_legacy(query, decomposition, chunks)


def _synthesize_legacy(
    query: str,
    decomposition: _Decomposition,
    chunks: list[dict],
) -> tuple[str, dict]:
    """Synthesizer: plain-text return, no calculator.

    Uses the dedicated synthesize-task LLM (configurable via
    `RESEARCH_AGENT_SYNTHESIZE_MODEL`). Default: `claude-sonnet-4-6`.
    """
    llm = LLMFactory.get_research_synthesize_llm()
    decomp_text = (
        f"Qualifiers: {decomposition.qualifiers}\n"
        f"Required quantities: {decomposition.required_quantities}\n"
        f"Sub-questions executed: {decomposition.sub_questions}"
    )
    prompt = AGENT_SYNTHESIZER_SYSTEM_PROMPT.format(
        query=query,
        decomposition=decomp_text,
        n_chunks=len(chunks),
        evidence=_full_evidence(chunks),
    )
    try:
        result = llm.invoke([
            _system_block(prompt, llm),
            HumanMessage(content="Synthesize now."),
        ])
        synthesis = result.content if isinstance(result.content, str) else str(result.content)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Synthesizer failed, falling back to chunk concatenation: {exc}")
        synthesis = "## Research findings\n\n(synthesizer error — main generator will use raw chunks)"
    return synthesis, _empty_calc_diagnostics()


def _synthesize_with_calculator(
    query: str,
    decomposition: _Decomposition,
    chunks: list[dict],
) -> tuple[str, dict]:
    """Calculator-aware synthesizer (gated behind ENABLE_CALCULATOR_TOOL).

    Uses `RESEARCH_AGENT_SYNTHESIZE_MODEL` (same as legacy path).
    """
    llm = LLMFactory.get_research_synthesize_llm()
    decomp_text = (
        f"Qualifiers: {decomposition.qualifiers}\n"
        f"Required quantities: {decomposition.required_quantities}\n"
        f"Sub-questions executed: {decomposition.sub_questions}"
    )
    prompt = AGENT_SYNTHESIZER_SYSTEM_PROMPT_CALC.format(
        query=query,
        decomposition=decomp_text,
        n_chunks=len(chunks),
        evidence=_full_evidence(chunks),
    )
    diagnostics = _empty_calc_diagnostics()

    structured = llm.with_structured_output(_AgentSynthesis)
    try:
        result: _AgentSynthesis = structured.invoke([
            _system_block(prompt, llm),
            HumanMessage(content="Synthesize now."),
        ])
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Synthesizer failed, falling back to chunk concatenation: {exc}")
        return (
            "## Research findings\n\n(synthesizer error — main generator will use raw chunks)",
            diagnostics,
        )

    synthesis = result.synthesis or ""
    expr = (result.arithmetic_expression or "").strip()
    if expr:
        diagnostics["calculator_expression"] = expr
        try:
            value = compute(expr)
            diagnostics["calculator_invoked"] = True
            diagnostics["calculator_value"] = value
            synthesis = (
                synthesis.rstrip()
                + f"\n\n**Verified arithmetic** (calculator-evaluated): "
                f"`{expr}` = **{value:.6g}**"
            )
            logger.info(f"Calculator: '{expr}' = {value:.6g}")
        except CalculatorError as exc:
            diagnostics["calculator_error"] = f"{type(exc).__name__}: {exc}"
            logger.warning(f"Calculator rejected '{expr}': {exc}")

    return synthesis, diagnostics


# ────────────────────────────────────────────────────────────
# In-Graph MCP Tool Calling via MCPClient
# ────────────────────────────────────────────────────────────

COMPANY_TICKER_MAP: dict[str, str] = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "tesla": "TSLA",
    "amazon": "AMZN",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "meta": "META",
    "facebook": "META",
    "nvidia": "NVDA",
    "jpmorgan": "JPM",
    "jpm": "JPM",
    "berkshire": "BRK-B",
    "berkshire_hathaway": "BRK-B",
    "3m": "MMM",
    "boeing": "BA",
    "coca_cola": "KO",
    "coca-cola": "KO",
    "coke": "KO",
    "costco": "COST",
    "netflix": "NFLX",
    "nike": "NKE",
    "walmart": "WMT",
}


def _load_sample_portfolio_holdings() -> dict[str, float]:
    """Load sample institutional portfolio holdings from data/sample_portfolio.json."""
    sample_path = _PROJECT_ROOT / "data" / "sample_portfolio.json"
    if sample_path.exists():
        try:
            with open(sample_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                holdings = data.get("holdings", {})
                if holdings:
                    return holdings
        except Exception as e:
            logger.warning(f"Failed to read sample_portfolio.json in research_agent: {e}")
    return {"AAPL": 14.5, "MSFT": 12.0, "NVDA": 9.5, "AMZN": 7.5, "GOOGL": 6.5}


def _resolve_ticker(query: str, target_company: str | None) -> str | None:
    """Resolve equity ticker from target_company slug or query token heuristics."""
    if target_company:
        norm = target_company.strip().lower()
        if norm in COMPANY_TICKER_MAP:
            return COMPANY_TICKER_MAP[norm]
    import re
    tickers = re.findall(r"\b[A-Z]{2,5}\b", query)
    if tickers:
        ignore = {
            "SEC", "GAAP", "CAGR", "EBIT", "EBITDA", "EPS", "USD", "ROE", "ROA",
            "FY", "Q1", "Q2", "Q3", "Q4", "HHI", "PE", "PB", "FCF", "EV", "RAG"
        }
        for t in tickers:
            if t not in ignore:
                return t
    return None


def _execute_mcp_tools(
    query: str,
    target_company: str | None,
    user_role: str,
    user_id: str,
    decomp: _Decomposition,
) -> tuple[list[dict[str, Any]], str]:
    """Execute permitted MCP tools through MCPClient inside the graph run context.

    Records spans under the active trace ('mcp.tool_call') alongside retrieval & LLM.
    Enforces buy-side RBAC: role-denied tool invocations surface explicitly in the graph trace
    and synthesis findings.
    """
    from src.tools.mcp_client import MCPClient

    client = MCPClient(user_role=user_role, user_id=user_id)
    results: list[dict[str, Any]] = []
    text_blocks: list[str] = []

    q_lower = query.lower()

    # 1. Market Data Tool (get_market_data)
    market_data_keywords = (
        "market data", "stock price", "share price", "market cap", "beta",
        "52-week", "52 week", "trading price", "live price", "current price",
        "pe ratio", "p/e", "forward p/e", "trailing p/e"
    )
    wants_market_data = any(kw in q_lower for kw in market_data_keywords) or any(
        any(kw in sq.lower() for kw in market_data_keywords) for sq in decomp.sub_questions
    )

    if wants_market_data:
        ticker = _resolve_ticker(query, target_company)
        if ticker:
            logger.info("Research agent invoking MCP tool 'get_market_data' for ticker=%s (role=%s)", ticker, user_role)
            res = client.call_tool("get_market_data", {"ticker": ticker})
            results.append(res)
            if not res["is_error"] and isinstance(res["data"], dict):
                d = res["data"]
                price_str = f"${d.get('current_price', 'N/A')}"
                mkt_cap = f"${d.get('market_cap', 0) / 1e9:.2f}B" if d.get("market_cap") else "N/A"
                text_blocks.append(
                    f"- **Live Market Data (`get_market_data`)**: Ticker `{d.get('ticker')}` | "
                    f"Price: {price_str} | Market Cap: {mkt_cap} | Trailing P/E: {d.get('trailing_pe', 'N/A')} | "
                    f"Beta: {d.get('beta', 'N/A')} | 52W Range: {d.get('fifty_two_week_low', 'N/A')} - {d.get('fifty_two_week_high', 'N/A')}"
                )
            else:
                text_blocks.append(f"- **MCP Tool `get_market_data`**: {res.get('text', 'Execution failed')}")

    # 2. Portfolio Risk Tool (calculate_portfolio_risk_metrics: Requires portfolio_manager role)
    portfolio_keywords = ("portfolio risk", "concentration", "hhi", "herfindahl", "portfolio weight", "holdings risk")
    wants_portfolio_risk = any(kw in q_lower for kw in portfolio_keywords)
    if wants_portfolio_risk:
        logger.info("Research agent invoking MCP tool 'calculate_portfolio_risk_metrics' (role=%s)", user_role)
        holdings = _load_sample_portfolio_holdings()
        res = client.call_tool("calculate_portfolio_risk_metrics", {"holdings": holdings})
        results.append(res)
        if res["is_error"]:
            text_blocks.append(
                f"- **Portfolio Risk Tool (`calculate_portfolio_risk_metrics`)**: "
                f"[GATING ENFORCED] {res.get('text')} (error_code={res.get('error_code')})"
            )
        else:
            d = res["data"]
            text_blocks.append(
                f"- **Portfolio Risk Tool (`calculate_portfolio_risk_metrics`)**: "
                f"HHI: {d.get('hhi_score')} | Concentration: Top-3={d.get('top_3_weight_pct')}% | "
                f"Effective Holdings: {d.get('effective_positions')}"
            )

    if not text_blocks:
        return results, ""

    formatted_block = "### MCP Tool Suite Findings (Live Data & Institutional Analytics):\n" + "\n".join(text_blocks)
    return results, formatted_block


# ────────────────────────────────────────────────────────────
# Top-level node
# ────────────────────────────────────────────────────────────


def research_agent_node(state: RAGState) -> dict:
    """Selective research agent for complex queries.

    Turn budget rationale:
      - Turn 1: decompose
      - Turns 2..(2+K): K sub-question retrievals (K=2..4)
      - Turn 2+K..: sufficiency + follow-ups, up to MAX_FOLLOWUP_ROUNDS
      - Final: synthesize + MCP tool execution
    Total LLM-driven turns capped at MAX_LLM_TURNS (5).
    """
    query = state.get("sanitized_query", "")
    if not query:
        return {"relevant_chunks": [], "agent_synthesis": None}

    target_company = state.get("target_company")
    target_fiscal_year = state.get("target_fiscal_year")

    # Turn 1: decompose
    decomp = _decompose(query, target_company, target_fiscal_year)
    sub_questions = decomp.sub_questions
    logger.info(
        f"Agent decomposed into {len(sub_questions)} sub-questions; "
        f"qualifiers={decomp.qualifiers}; required={decomp.required_quantities}"
    )
    from src.services.event_log import emit
    emit("research_agent.decompose", n_sub_questions=len(sub_questions),
         qualifiers=list(decomp.qualifiers), required_quantities=list(decomp.required_quantities),
         sub_questions=list(sub_questions))

    # Turns 2..(2+K): retrieve + grade per sub-question
    chunks_by_id: dict[tuple, dict] = {}
    for subq in sub_questions:
        for c in _retrieve_and_grade_for_subq(state, subq):
            cid = _chunk_id(c)
            if cid not in chunks_by_id:
                chunks_by_id[cid] = c

    logger.info(f"Agent collected {len(chunks_by_id)} unique chunks across {len(sub_questions)} sub-questions")

    # Sufficiency loop — at most MAX_FOLLOWUP_ROUNDS additional retrievals
    sufficiency_history: list[dict] = []
    for round_idx in range(MAX_FOLLOWUP_ROUNDS):
        chunks = list(chunks_by_id.values())
        verdict = _judge_sufficiency(query, decomp, chunks)
        sufficiency_history.append({
            "decision": verdict.decision,
            "missing": verdict.missing_quantity,
            "follow_up": verdict.follow_up_question,
            "reason": verdict.reason,
        })
        if verdict.decision == "sufficient":
            logger.info(f"Agent: sufficient after round {round_idx} ({verdict.reason})")
            emit("research_agent.sufficiency", decision="sufficient", round=round_idx,
                 reason=verdict.reason, n_chunks_so_far=len(chunks_by_id))
            break
        if not verdict.follow_up_question:
            logger.info(f"Agent: 'need_more' but no follow-up — exiting loop")
            emit("research_agent.sufficiency", decision="need_more_no_followup",
                 round=round_idx, reason=verdict.reason, n_chunks_so_far=len(chunks_by_id))
            break
        logger.info(f"Agent follow-up [{round_idx}]: {verdict.follow_up_question} ({verdict.reason})")
        emit("research_agent.sufficiency", decision="need_more", round=round_idx,
             follow_up=verdict.follow_up_question, missing=verdict.missing_quantity,
             reason=verdict.reason)
        for c in _retrieve_and_grade_for_subq(state, verdict.follow_up_question):
            cid = _chunk_id(c)
            if cid not in chunks_by_id:
                chunks_by_id[cid] = c

    # Step: Execute MCP Tools via MCPClient within the active graph execution context
    user_role = state.get("user_role", "analyst")
    user_id = state.get("user_id", "research_agent")
    mcp_results, mcp_block = _execute_mcp_tools(
        query=query,
        target_company=target_company,
        user_role=user_role,
        user_id=user_id,
        decomp=decomp,
    )

    # Final turn: synthesize (also computes arithmetic via the calculator tool)
    relevant_chunks = list(chunks_by_id.values())
    synthesis, calc_diagnostics = _synthesize(query, decomp, relevant_chunks)
    if mcp_block:
        synthesis = f"{synthesis}\n\n{mcp_block}" if synthesis else mcp_block

    # Compute total LLM turns: 1 decompose + len(sufficiency_history) + 1 synthesize
    turns_used = 1 + len(sufficiency_history) + 1

    logger.info(
        f"Agent done: {len(relevant_chunks)} chunks, {turns_used} Claude turns, "
        f"sub_questions={sub_questions}, "
        f"calc_invoked={calc_diagnostics['calculator_invoked']}, "
        f"mcp_tools_invoked={len(mcp_results)}"
    )
    emit("research_agent.synthesize", n_chunks=len(relevant_chunks), turns_used=turns_used,
         calculator_invoked=calc_diagnostics.get("calculator_invoked", False),
         calculator_value=calc_diagnostics.get("calculator_value"),
         mcp_tools_invoked=[r["name"] for r in mcp_results],
         synthesis_chars=len(synthesis) if synthesis else 0)

    # Mark grading_results so the main grader path doesn't re-grade.
    grading_results = [{"agent_curated": True} for _ in relevant_chunks]

    mcp_diagnostics = {
        "mcp_tools_invoked": [r["name"] for r in mcp_results],
        "mcp_tool_results": mcp_results,
        "mcp_role_denials": [r["name"] for r in mcp_results if r.get("error_code") == "PERMISSION_DENIED"],
    }

    return {
        "relevant_chunks": relevant_chunks,
        "grading_results": grading_results,
        "agent_synthesis": synthesis,
        "agent_turns_used": turns_used,
        "agent_sub_questions": sub_questions,
        **calc_diagnostics,
        **mcp_diagnostics,
    }
