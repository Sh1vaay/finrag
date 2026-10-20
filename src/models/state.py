from typing import Annotated, Any, Optional, TypedDict

try:
    from langchain_core.messages import BaseMessage
    from langgraph.graph.message import add_messages
except ImportError:
    BaseMessage = Any  # type: ignore

    def add_messages(left, right):  # type: ignore
        return (left or []) + (right or [])


class RAGState(TypedDict):
    """Shared state flowing through every node in the RAG agent graph."""

    # --- Input ---
    messages: Annotated[list[BaseMessage], add_messages]

    # --- Auth ---
    user_id: str
    user_role: str  # "research_analyst", "portfolio_manager", "risk_compliance", "admin"
    allowed_doc_types: list[str]  # ["10k", "10q", "earnings_transcript", "equity_research", "portfolio_holdings", ...]

    # --- Guardrails ---
    guardrail_status: str  # "clean", "pii_detected", "injection_detected", "out_of_scope"
    detected_pii_entities: list[dict]
    sanitized_query: str

    # --- Routing ---
    query_intent: str  # "retrieval", "market_data", "portfolio_risk", "clarification", "out_of_scope"
    # Complexity classification: "simple_lookup" (single-fact retrieval,
    # fast path) vs "research_required" (multi-section synthesis, calc with formula,
    # comparative — routed to the research agent subgraph). None means classifier
    # didn't set it (treat as simple_lookup downstream).
    query_complexity: str | None
    system_tier: str | None  # "system_1" (fast decider path) vs "system_2" (deep reasoning)
    decider_confidence: float | None
    decider_model: str | None

    # --- Entity extraction ---
    # Lowercase slug matching a `company` payload value in Qdrant, or None if the
    # query isn't scoped to a specific company (e.g. "compare Apple and MSFT").
    target_company: str | None
    target_fiscal_year: int | None

    # --- Retrieval ---
    retrieved_chunks: list[dict]  # Hybrid candidates (top ~50 from dense+BM25 RRF)
    retrieval_query: str

    # --- Reranking ---
    reranked_chunks: list[dict]  # Cross-encoder top-K selected from retrieved_chunks
    candidate_diagnostics: list[dict]  # Optional per-candidate validation/LTR diagnostics

    # --- Optional selective retrieval evaluator ---
    retrieval_evaluator_confidence: float | None
    retrieval_evaluator_decision: str | None  # "accept", "retry"

    # --- Grading ---
    relevant_chunks: list[dict]
    grading_results: list[dict]

    # --- Empty-context fallback diagnostics ---
    # Set to True when the retrieval node had to drop its entity filter to find
    # any chunks at all (retrieval-side fallback), or when the grader had to
    # pass through top-K reranked chunks because all chunks failed grading
    # (grader-side fallback). Either signal indicates a "best-effort" answer
    # rather than a confidently-grounded one — useful for failure analysis.
    retrieval_fallback_used: bool | None
    grader_fallback_used: bool | None

    # --- Research agent ---
    # Set when the research agent runs (query_complexity == "research_required").
    # `agent_synthesis` is a structured-text reasoning block the agent produces
    # to guide the main generator; raw `relevant_chunks` are still populated and
    # remain the substrate the hallucination checker grounds against.
    agent_synthesis: str | None
    agent_turns_used: int | None
    agent_sub_questions: list[str] | None

    # --- Calculator tool ---
    # Set when the synthesizer emits an arithmetic expression and we evaluated
    # it with src/tools/calculator.py. Populated only on the research-agent path.
    calculator_invoked: bool | None
    calculator_expression: str | None
    calculator_value: float | None
    calculator_error: str | None

    # --- MCP Tool Suite (in-graph tool calling via MCPClient) ---
    mcp_tools_invoked: list[str] | None
    mcp_tool_results: list[dict] | None
    mcp_role_denials: list[str] | None

    # --- Quantitative Decision Analytics (Piotroski, Altman Z, Factor Scoring) ---
    quantitative_analysis: dict | None

    # --- Generation ---
    generated_answer: str

    # --- Hallucination Check ---
    hallucination_status: str  # "grounded", "hallucinated"
    hallucination_score: float

    # --- HITL ---
    requires_human_approval: bool
    human_decision: Optional[str]  # "approved", "rejected", None
    # HITL audit metadata — populated when an authorized approver resumes
    # the paused graph via /v1/hitl/approve or /v1/hitl/reject. Surfaced to the
    # requester via /v1/chat/result so they can see who decided their request,
    # when, and (on reject) why.
    human_decision_at: Optional[str]  # ISO-8601 timestamp
    human_decision_by: Optional[str]  # approver user_id
    human_decision_by_role: Optional[str]  # approver role
    human_decision_reason: Optional[str]  # rejection reason (mandatory on reject)
    hitl_submitted_at: Optional[str]  # ISO-8601 when the gate fired

    # --- Control Flow ---
    retrieval_retry_count: int
    generation_retry_count: int

    # --- Output ---
    final_response: str
    response_metadata: dict
