from pydantic import model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- Environment ---
    ENVIRONMENT: str = "dev"
    LOG_LEVEL: str = "INFO"

    # --- CORS ---
    # Safe explicit default origins for local development; wildcard is strictly forbidden in production.
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:5173", "http://localhost:8000"]

    # --- LLM Provider Selection & Strategy ---
    # Strategy mode:
    #   - "hybrid": task-specialized providers (fast-path routing + specialized generation)
    #   - "open_source": routes calls to an open-source endpoint (Ollama, vLLM, LM Studio, etc.)
    #   - "closed_source": routes calls to a selected closed-source provider (OpenAI, Anthropic, etc.)
    #   - "placeholder": mock mode for executing graph pipeline without active API keys or endpoints
    LLM_PROVIDER_TYPE: str = "hybrid"

    # --- Open-Source / Local LLM Placeholders (Ollama, vLLM, LM Studio, etc.) ---
    OPEN_SOURCE_LLM_BASE_URL: str = "http://localhost:11434/v1"
    OPEN_SOURCE_LLM_MODEL: str = "placeholder-open-model"
    OPEN_SOURCE_LLM_API_KEY: str = "placeholder-key"

    # --- System 1 Fast Decider (Strand Decider 2B) & Market Data ---
    ENABLE_SYSTEM_ONE_DECIDER: bool = True
    DECIDER_2B_MODEL: str = "strand-decider-2b"
    DECIDER_2B_BASE_URL: str = "http://localhost:11434/v1"
    MARKET_DATA_PROVIDER: str = "institutional"  # "institutional" | "yfinance"
    SEC_EDGAR_USER_AGENT: str = "FinRAG-AssetManagement/1.0 (compliance@finrag-institutional.com)"

    # --- Closed-Source LLM Placeholders (OpenAI, Anthropic, etc.) ---
    CLOSED_SOURCE_LLM_PROVIDER: str = "openai"  # "openai" | "anthropic"
    CLOSED_SOURCE_LLM_MODEL: str = "placeholder-closed-model"
    CLOSED_SOURCE_LLM_API_KEY: str = ""

    # Provider API Keys
    OPENAI_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    FIREWORKS_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""
    VOYAGE_API_KEY: str = ""
    ABACI_NLP_API_KEY: str = ""

    # --- External eval overrides ---
    # Override: force all LLM calls through OpenAI (used to bypass rate limits during
    # bulk evaluation runs). Default false = prod hybrid Groq + OpenAI + Anthropic.
    FORCE_OPENAI_ONLY: bool = False
    # Use Groq for the high-volume "fast path" nodes (router, grader, query rewriter).
    # Set false to send those to OpenAI even when GROQ_API_KEY is set.
    # Default true preserves the production latency profile.
    USE_GROQ_FAST_PATH: bool = True

    # --- Tracing & Observability ---
    # Selector: "phoenix" (default open-source OpenTelemetry), "langfuse", or "none"
    TRACING_BACKEND: str = "phoenix"
    PHOENIX_COLLECTOR_ENDPOINT: str = "http://localhost:6006/v1/traces"
    PHOENIX_PROJECT_NAME: str = "finrag-investment"

    # --- LangSmith ---
    LANGCHAIN_TRACING_V2: bool = True
    LANGCHAIN_PROJECT: str = "rag-agent-dev"
    LANGCHAIN_API_KEY: str = ""

    # --- Auth ---
    JWT_SECRET: str = "dev-secret-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_HOURS: int = 24

    # --- Qdrant ---
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "financial_docs"

    # --- Documents (Frontend PDF citation clickthrough) ---
    # Filesystem root from which `GET /documents/{filename}` serves PDF files.
    # Filenames are validated against path-traversal; only basenames matching
    # files inside this directory are served (no recursive lookup).
    DOCUMENTS_ROOT: str = "data/sample"

    # --- LiteLLM gateway ---
    # Single proxy fronting every LLM call. When set, src/services/llm_factory.py
    # routes Anthropic / OpenAI / Groq calls through this URL instead of hitting
    # provider APIs directly. Default empty = direct-provider behavior.
    # Set to `http://litellm:4000` inside compose or `http://localhost:4000` for host-based dev.
    LITELLM_URL: str = ""

    # --- Langfuse ---
    # Self-hosted or cloud Langfuse instance. Supply via environment variables (.env).
    LANGFUSE_HOST: str = "http://langfuse-web:3000"
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""

    # --- PostgreSQL ---
    POSTGRES_DB: str = "rag_agent"
    POSTGRES_USER: str = "rag_user"
    POSTGRES_PASSWORD: str = "devpassword"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432

    @property
    def postgres_url(self) -> str:
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # --- Agent Thresholds ---
    HALLUCINATION_THRESHOLD: float = 0.7
    GRADING_MIN_RELEVANT_CHUNKS: int = 1
    MAX_RETRIEVAL_RETRIES: int = 2
    MAX_GENERATION_RETRIES: int = 2
    HITL_AMOUNT_THRESHOLD: int = 100_000
    # Compliance & Regulatory Monitoring triggers (automatically initiates HITL review)
    COMPLIANCE_RESTRICTED_KEYWORDS: list[str] = [
        "restricted list",
        "restricted stock",
        "mnpi",
        "material non-public",
        "insider trading",
        "blackout window",
        "blackout period",
        "embargoed research",
    ]
    # Retrieval: cast a wide net (hybrid dense+BM25 with RRF) so the reranker has
    # enough variety to pick from.
    RETRIEVAL_TOP_K: int = 50
    # Reranker: cross-encoder narrows to the final set that feeds the grader + generator.
    RERANKER_TOP_K: int = 8
    # Non-LLM post-reranker validation
    ENABLE_DETERMINISTIC_VALIDATOR: bool = True
    VALIDATOR_MIN_KEEP: int = 3
    # Grader empty-context fallback: when grader yields 0
    # relevant chunks, pass through top-K reranker chunks (best-effort) instead
    # of refusing. Default OFF; can be re-enabled per-deployment if downstream
    # is more tolerant of low-confidence chunks.
    ENABLE_GRADER_EMPTY_CONTEXT_FALLBACK: bool = False
    # Calculator tool in research-agent synthesizer.
    # Enables structured-output calculator evaluation on arithmetic expressions.
    ENABLE_CALCULATOR_TOOL: bool = False
    # Learned LTR gate (Phase 2)
    ENABLE_LTR_GATE: bool = False
    LTR_GATE_MODEL_PATH: str = "data/models/ltr_gate.json"
    LTR_GATE_HIGH_CONFIDENCE: float = 0.9
    LTR_GATE_LOW_CONFIDENCE: float = 0.1
    # Optional selective retrieval evaluator (Phase 4)
    ENABLE_SELECTIVE_RETRIEVAL_EVALUATOR: bool = False
    RETRIEVAL_EVALUATOR_MIN_CONFIDENCE: float = 0.55

    # Deterministic generation seed and guardrails sensitivity
    LLM_SEED: int = 42
    GUARDRAILS_INJECTION_THRESHOLD: float = 0.9

    # --- Multi-HyDE ---
    # Generates N hypothetical answers per query, embeds each, runs hybrid_search
    # across all (orig + N) paths, RRF-fuses results, dedupes, returns top-K.
    # Targets the vocabulary gap from question-phrasing → document-phrasing.
    ENABLE_MULTI_HYDE: bool = False
    MULTI_HYDE_N: int = 3
    MULTI_HYDE_MODEL: str = "gpt-4o-mini"
    # Temperature > 0 — we explicitly want the N hypotheticals to be different.
    MULTI_HYDE_TEMPERATURE: float = 0.3

    # --- Embedding ---
    # Provider dispatch: "openai" (default), "voyage", "bge-m3" (local open-source 1024-dim),
    # or "abaci". When "bge-m3", embeddings execute locally without API calls and
    # target an isolated 1024-dim collection ("financial_docs_bgem3").
    EMBEDDING_PROVIDER: str = "openai"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS: int = 1536

    # --- LLM Models ---
    # Kept on Groq for latency-critical classification / routing
    ROUTER_MODEL: str = "llama-3.3-70b-versatile"
    GRADER_MODEL: str = "llama-3.3-70b-versatile"
    # Generator uses Claude Sonnet 4.6 for stronger instruction-following + grounding.
    # OpenAI remains the fallback.
    GENERATOR_MODEL: str = "claude-sonnet-4-6"
    # --- Per-task model tiering ---
    HALLUCINATION_MODEL: str = "claude-sonnet-4-6"
    # HITL high-stakes: verification model for high-dollar actions.
    HIGH_STAKES_HALLUCINATION_MODEL: str = "claude-sonnet-4-6"
    # --- Research-agent sub-models ---
    # decompose + sufficiency: structured-output classifier tasks; gpt-4o-mini
    # handles them with high speed and low cost.
    RESEARCH_AGENT_DECOMPOSE_MODEL: str = "gpt-4o-mini"
    RESEARCH_AGENT_SUFFICIENCY_MODEL: str = "gpt-4o-mini"
    RESEARCH_AGENT_SYNTHESIZE_MODEL: str = "claude-sonnet-4-6"
    # Legacy OpenAI fallback model — used when Anthropic fails or FORCE_OPENAI_ONLY is on
    OPENAI_FALLBACK_MODEL: str = "gpt-4o-mini"

    # Opt-in route the grader through Llama-3.3-70B-Instruct at an OpenAI-compatible provider.
    # USE_LLAMA_GRADER turns it on; LLAMA_GRADER_PROVIDER picks the backend:
    #   - "openrouter" (default)
    #   - "fireworks"
    USE_LLAMA_GRADER: bool = False
    LLAMA_GRADER_PROVIDER: str = "openrouter"  # "openrouter" | "fireworks"
    LLAMA_GRADER_MODEL_OPENROUTER: str = "meta-llama/llama-3.3-70b-instruct"
    LLAMA_GRADER_MODEL_FIREWORKS: str = "accounts/fireworks/models/llama-v3p3-70b-instruct"

    @property
    def langchain_project_name(self) -> str:
        """Return env-specific LangSmith project name."""
        if self.LANGCHAIN_PROJECT != "rag-agent-dev":
            return self.LANGCHAIN_PROJECT
        env_suffix = {"dev": "dev", "staging": "staging", "production": "prod"}
        return f"rag-agent-{env_suffix.get(self.ENVIRONMENT, self.ENVIRONMENT)}"

    @model_validator(mode="after")
    def validate_production_settings(self):
        """Enforce safety checks for production and staging environments."""
        if self.ENVIRONMENT in ("production", "staging"):
            if self.JWT_SECRET == "dev-secret-change-in-production":
                raise ValueError(f"JWT_SECRET must be changed from default in {self.ENVIRONMENT}!")
            if self.POSTGRES_PASSWORD == "devpassword":
                raise ValueError(f"POSTGRES_PASSWORD must be changed from default in {self.ENVIRONMENT}!")
            if "*" in self.CORS_ORIGINS:
                raise ValueError(f"CORS_ORIGINS must not be wildcard '*' in {self.ENVIRONMENT}!")
        return self

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
