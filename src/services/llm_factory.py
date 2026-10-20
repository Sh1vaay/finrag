"""LLM factory with provider selection, open/closed-source routing, and fallback chains.

Supported provider strategies:
1. **Hybrid / Specialized** (default):
   - Fast path (router, grader, query rewriter): Groq / OpenAI
   - Generation (answer synthesis): Claude Sonnet / OpenAI
   - Verification (hallucination checks): Claude Sonnet / Haiku / OpenAI
2. **Open-Source / Local** (`LLM_PROVIDER_TYPE="open_source"`):
   - Routes through an OpenAI-compatible endpoint (Ollama, vLLM, LM Studio, LocalAI, TGI)
   - Configured via `OPEN_SOURCE_LLM_BASE_URL` and `OPEN_SOURCE_LLM_MODEL`
3. **Closed-Source** (`LLM_PROVIDER_TYPE="closed_source"`):
   - Routes through chosen commercial provider (`CLOSED_SOURCE_LLM_PROVIDER="openai"|"anthropic"`)
4. **Placeholder / Dry-Run** (`LLM_PROVIDER_TYPE="placeholder"`):
   - Returns mock responses and valid structured routing objects without requiring any live API keys.
"""

import logging
import os
from typing import Any, Iterator

from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from src.config.settings import settings
from src.services.cost_tracker import get_cost_handler
from src.services.request_context import current_user_id

logger = logging.getLogger(__name__)


class PlaceholderChatModel(BaseChatModel):
    """Fallback placeholder model for development, dry-run, and unconfigured states.

    Allows graph execution, testing, and API verification without live API keys
    or running model backends.
    """

    model_name: str = "placeholder-model"
    placeholder_text: str = (
        "[FinRAG Placeholder] No LLM model configured yet. "
        "Set OPEN_SOURCE_LLM_* (e.g. Ollama/vLLM) or CLOSED_SOURCE_LLM_* (e.g. OpenAI/Anthropic) "
        "in your .env file."
    )

    @property
    def _llm_type(self) -> str:
        return "placeholder"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = AIMessage(content=self.placeholder_text)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        tokens = self.placeholder_text.split(" ")
        for i, token in enumerate(tokens):
            chunk_text = token if i == len(tokens) - 1 else token + " "
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=chunk_text))
            if run_manager:
                run_manager.on_llm_new_token(chunk_text)
            yield chunk

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        """Return a runnable that creates a default structured object for graph traversal."""
        fallback_text = self.placeholder_text

        class _StructuredPlaceholderRunnable:
            def __init__(self, target_schema: Any):
                self._schema = target_schema

            def invoke(self, input_val: Any, config: Any = None) -> Any:
                if isinstance(self._schema, type) and issubclass(self._schema, BaseModel):
                    name = self._schema.__name__
                    if name == "RouterDecision":
                        return self._schema(
                            intent="retrieval",
                            reason="[Placeholder router decision — placeholder LLM active]",
                            complexity="simple_lookup",
                            system_tier="system_2",
                            confidence=1.0,
                            decider_model="placeholder",
                        )
                    if name == "GradeResult":
                        return self._schema(
                            relevant=True,
                            reason="[Placeholder chunk relevant]",
                        )
                    if name == "HallucinationCheck":
                        return self._schema(
                            grounded=True,
                            score=1.0,
                            explanation="[Placeholder verified grounded]",
                        )
                    if name == "InjectionCheck":
                        return self._schema(
                            is_injection=False,
                            confidence=0.0,
                            reason="[Placeholder clean query]",
                        )
                    if name == "RetrievalEvalDecision":
                        return self._schema(
                            confidence=0.9,
                            is_actionable=True,
                            reason="[Placeholder confident retrieval]",
                        )
                    if name == "EntityExtraction":
                        return self._schema(
                            company="apple",
                            fiscal_year=2023,
                        )
                    try:
                        return self._schema.model_construct()
                    except Exception:
                        pass
                return fallback_text

            async def ainvoke(self, input_val: Any, config: Any = None) -> Any:
                return self.invoke(input_val, config)

        return _StructuredPlaceholderRunnable(schema)


def _attribution_kwargs() -> dict:
    """Build the per-call kwargs that carry user attribution to LiteLLM/Langfuse."""
    uid = current_user_id.get()
    if not uid or not settings.LITELLM_URL:
        return {}
    return {"model_kwargs": {"user": uid}}


_OPENAI_SEED = getattr(settings, "LLM_SEED", 42)


def _openai(model: str, temperature: float = 0.0, max_tokens: int | None = None) -> ChatOpenAI:
    """OpenAI chat model. Routes through LiteLLM when LITELLM_URL is set, else direct."""
    kwargs = {
        "model": model,
        "temperature": temperature,
        "seed": _OPENAI_SEED,
        "callbacks": [get_cost_handler()],
    }
    if settings.LITELLM_URL:
        kwargs["base_url"] = f"{settings.LITELLM_URL.rstrip('/')}/v1"
        kwargs["api_key"] = "sk-litellm-proxy"
    else:
        kwargs["api_key"] = settings.OPENAI_API_KEY
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    kwargs.update(_attribution_kwargs())
    return ChatOpenAI(**kwargs)


_ANTHROPIC_NO_TEMPERATURE_MODELS: set[str] = {"claude-opus-4-7"}


def _anthropic(model: str, temperature: float = 0.0, max_tokens: int = 1024) -> ChatAnthropic:
    """Anthropic chat model."""
    kwargs = {
        "model_name": model,
        "max_tokens": max_tokens,
        "callbacks": [get_cost_handler()],
    }
    if settings.LITELLM_URL:
        kwargs["base_url"] = f"{settings.LITELLM_URL.rstrip('/')}/anthropic"
        kwargs["api_key"] = "sk-litellm-proxy"
    else:
        kwargs["api_key"] = settings.ANTHROPIC_API_KEY
    if model not in _ANTHROPIC_NO_TEMPERATURE_MODELS:
        kwargs["temperature"] = temperature
    kwargs.update(_attribution_kwargs())
    return ChatAnthropic(**kwargs)


def _open_source(
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> ChatOpenAI:
    """Open-source / local LLM runner (Ollama, vLLM, LM Studio, LocalAI, etc.)."""
    endpoint = (base_url or settings.OPEN_SOURCE_LLM_BASE_URL).rstrip("/")
    key = api_key or settings.OPEN_SOURCE_LLM_API_KEY or "placeholder-key"
    target_model = model or settings.OPEN_SOURCE_LLM_MODEL
    kwargs = {
        "model": target_model,
        "temperature": temperature,
        "base_url": endpoint,
        "api_key": key,
        "callbacks": [get_cost_handler()],
    }
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    kwargs.update(_attribution_kwargs())
    return ChatOpenAI(**kwargs)


def _groq(model: str, temperature: float = 0.0) -> ChatGroq | ChatOpenAI:
    """Groq chat model."""
    if settings.LITELLM_URL:
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            seed=_OPENAI_SEED,
            base_url=f"{settings.LITELLM_URL.rstrip('/')}/v1",
            api_key="sk-litellm-proxy",
            callbacks=[get_cost_handler()],
            **_attribution_kwargs(),
        )
    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=settings.GROQ_API_KEY,
        callbacks=[get_cost_handler()],
    )


_warned_keys: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    """Emit each warning once per process."""
    if key not in _warned_keys:
        _warned_keys.add(key)
        logger.warning(msg)


def _llm_for_task(model_name: str, temperature: float = 0.0, max_tokens: int = 2048):
    """Dispatch to the right provider based on prefix or global configuration.

    Supports:
      - "placeholder", "mock": returns PlaceholderChatModel
      - "ollama/*", "vllm/*", "local/*", "open_source/*": returns OpenAI-compatible endpoint
      - "claude-*", "anthropic/*": returns Anthropic
      - "gpt-*", "openai/*": returns OpenAI
      - "groq/*": returns Groq
      - Global LLM_PROVIDER_TYPE == "open_source" -> _open_source(model=model_name)
      - Global LLM_PROVIDER_TYPE == "placeholder" -> PlaceholderChatModel
    """
    if model_name in ("placeholder", "mock") or settings.LLM_PROVIDER_TYPE == "placeholder":
        return PlaceholderChatModel(model_name=model_name)

    # Open-source / local prefixes
    for prefix in ("ollama/", "vllm/", "local/", "open_source/"):
        if model_name.startswith(prefix):
            stripped = model_name[len(prefix):]
            return _open_source(model=stripped, temperature=temperature, max_tokens=max_tokens)

    # Provider prefixes
    if model_name.startswith("claude-"):
        return _anthropic(model_name, temperature=temperature, max_tokens=max_tokens)
    if model_name.startswith("anthropic/"):
        return _anthropic(model_name[len("anthropic/"):], temperature=temperature, max_tokens=max_tokens)
    if model_name.startswith("gpt-"):
        return _openai(model_name, temperature=temperature, max_tokens=max_tokens)
    if model_name.startswith("openai/"):
        return _openai(model_name[len("openai/"):], temperature=temperature, max_tokens=max_tokens)
    if model_name.startswith("groq/"):
        return _groq(model_name[len("groq/"):], temperature=temperature)

    # Global provider type overrides
    if settings.LLM_PROVIDER_TYPE == "open_source":
        return _open_source(model=model_name, temperature=temperature, max_tokens=max_tokens)
    if settings.LLM_PROVIDER_TYPE == "closed_source":
        if settings.CLOSED_SOURCE_LLM_PROVIDER == "anthropic" and settings.ANTHROPIC_API_KEY:
            return _anthropic(model_name, temperature=temperature, max_tokens=max_tokens)
        return _openai(model_name, temperature=temperature, max_tokens=max_tokens)

    # Fallback to OpenAI or Anthropic if keys available, else Placeholder
    if settings.OPENAI_API_KEY or settings.LITELLM_URL:
        return _openai(model_name, temperature=temperature, max_tokens=max_tokens)
    if settings.ANTHROPIC_API_KEY:
        return _anthropic(model_name, temperature=temperature, max_tokens=max_tokens)

    _warn_once(
        "unconfigured_llm_fallback",
        f"Model {model_name!r} requested but no matching API key found. "
        "Falling back to PlaceholderChatModel.",
    )
    return PlaceholderChatModel(model_name=model_name)


class LLMFactory:
    """Creates LLM instances with automatic fallback between providers."""

    @staticmethod
    def get_router_llm():
        """Classification-tier model for routing and query intent."""
        if settings.LLM_PROVIDER_TYPE == "placeholder":
            return PlaceholderChatModel(model_name="placeholder-router")
        if settings.LLM_PROVIDER_TYPE == "open_source":
            return _open_source(model=settings.OPEN_SOURCE_LLM_MODEL, temperature=0.0)
        if settings.LLM_PROVIDER_TYPE == "closed_source":
            if settings.CLOSED_SOURCE_LLM_PROVIDER == "anthropic" and settings.ANTHROPIC_API_KEY:
                return _anthropic(settings.CLOSED_SOURCE_LLM_MODEL, 0.0)
            return _openai(settings.CLOSED_SOURCE_LLM_MODEL, 0.0)

        # Hybrid default
        if settings.FORCE_OPENAI_ONLY or not settings.USE_GROQ_FAST_PATH:
            if settings.OPENAI_API_KEY or settings.LITELLM_URL:
                return _openai(settings.OPENAI_FALLBACK_MODEL, 0.0)
        if settings.GROQ_API_KEY:
            return _groq(settings.ROUTER_MODEL, 0.0)
        if settings.OPENAI_API_KEY or settings.LITELLM_URL:
            _warn_once("router_no_groq", "Groq API key not set for router, falling back to OpenAI")
            return _openai(settings.OPENAI_FALLBACK_MODEL, 0.0)

        _warn_once("router_placeholder", "No LLM keys configured for router, using PlaceholderChatModel")
        return PlaceholderChatModel(model_name="placeholder-router")

    @staticmethod
    def get_grader_llm():
        """Binary classification LLM for document chunk relevance."""
        if settings.LLM_PROVIDER_TYPE == "placeholder":
            return PlaceholderChatModel(model_name="placeholder-grader")
        if settings.LLM_PROVIDER_TYPE == "open_source":
            return _open_source(model=settings.OPEN_SOURCE_LLM_MODEL, temperature=0.0)
        if settings.LLM_PROVIDER_TYPE == "closed_source":
            if settings.CLOSED_SOURCE_LLM_PROVIDER == "anthropic" and settings.ANTHROPIC_API_KEY:
                return _anthropic(settings.CLOSED_SOURCE_LLM_MODEL, 0.0)
            return _openai(settings.CLOSED_SOURCE_LLM_MODEL, 0.0)

        if settings.USE_LLAMA_GRADER:
            provider = settings.LLAMA_GRADER_PROVIDER.lower()
            if provider == "openrouter":
                key = settings.OPENROUTER_API_KEY or os.environ.get("OPENROUTER_API_KEY")
                if key:
                    _warn_once("grader_openrouter_llama", f"Grader: OpenRouter {settings.LLAMA_GRADER_MODEL_OPENROUTER}")
                    return ChatOpenAI(
                        model=settings.LLAMA_GRADER_MODEL_OPENROUTER,
                        temperature=0.0,
                        max_tokens=1024,
                        base_url="https://openrouter.ai/api/v1",
                        api_key=key,
                        max_retries=3,
                        callbacks=[get_cost_handler()],
                        **_attribution_kwargs(),
                    )
            elif provider == "fireworks":
                key = settings.FIREWORKS_API_KEY or os.environ.get("FIREWORKS_API_KEY")
                if key:
                    _warn_once("grader_fireworks_llama", f"Grader: Fireworks {settings.LLAMA_GRADER_MODEL_FIREWORKS}")
                    return ChatOpenAI(
                        model=settings.LLAMA_GRADER_MODEL_FIREWORKS,
                        temperature=0.0,
                        max_tokens=512,
                        base_url="https://api.fireworks.ai/inference/v1",
                        api_key=key,
                        max_retries=3,
                        callbacks=[get_cost_handler()],
                        **_attribution_kwargs(),
                    )

        if settings.FORCE_OPENAI_ONLY or not settings.USE_GROQ_FAST_PATH:
            if settings.OPENAI_API_KEY or settings.LITELLM_URL:
                return _openai(settings.OPENAI_FALLBACK_MODEL, 0.0)
        if settings.GROQ_API_KEY:
            return _groq(settings.GRADER_MODEL, 0.0)
        if settings.OPENAI_API_KEY or settings.LITELLM_URL:
            _warn_once("grader_no_groq", "Groq API key not set for grader, falling back to OpenAI")
            return _openai(settings.OPENAI_FALLBACK_MODEL, 0.0)

        _warn_once("grader_placeholder", "No LLM keys configured for grader, using PlaceholderChatModel")
        return PlaceholderChatModel(model_name="placeholder-grader")

    @staticmethod
    def get_generator_llm():
        """User-facing answer generator."""
        if settings.LLM_PROVIDER_TYPE == "placeholder":
            return PlaceholderChatModel(model_name="placeholder-generator")
        if settings.LLM_PROVIDER_TYPE == "open_source":
            return _open_source(model=settings.OPEN_SOURCE_LLM_MODEL, temperature=0.1, max_tokens=2048)
        if settings.LLM_PROVIDER_TYPE == "closed_source":
            if settings.CLOSED_SOURCE_LLM_PROVIDER == "anthropic" and settings.ANTHROPIC_API_KEY:
                return _anthropic(settings.CLOSED_SOURCE_LLM_MODEL, temperature=0.1, max_tokens=2048)
            return _openai(settings.CLOSED_SOURCE_LLM_MODEL, temperature=0.1, max_tokens=2048)

        if settings.FORCE_OPENAI_ONLY:
            return _openai(settings.OPENAI_FALLBACK_MODEL, 0.1)
        if settings.ANTHROPIC_API_KEY:
            return _anthropic(settings.GENERATOR_MODEL, temperature=0.1, max_tokens=2048)
        if settings.OPENAI_API_KEY or settings.LITELLM_URL:
            logger.warning("Anthropic API key not set for generator, falling back to OpenAI")
            return _openai(settings.OPENAI_FALLBACK_MODEL, 0.1)

        _warn_once("generator_placeholder", "No LLM keys configured for generator, using PlaceholderChatModel")
        return PlaceholderChatModel(model_name="placeholder-generator")

    @staticmethod
    def get_hallucination_llm():
        """Standard grounding verification model."""
        if settings.LLM_PROVIDER_TYPE == "placeholder":
            return PlaceholderChatModel(model_name="placeholder-hallucination")
        if settings.LLM_PROVIDER_TYPE == "open_source":
            return _open_source(model=settings.OPEN_SOURCE_LLM_MODEL, temperature=0.0, max_tokens=1024)
        if settings.LLM_PROVIDER_TYPE == "closed_source":
            if settings.CLOSED_SOURCE_LLM_PROVIDER == "anthropic" and settings.ANTHROPIC_API_KEY:
                return _anthropic(settings.CLOSED_SOURCE_LLM_MODEL, temperature=0.0, max_tokens=1024)
            return _openai(settings.CLOSED_SOURCE_LLM_MODEL, temperature=0.0, max_tokens=512)

        if settings.FORCE_OPENAI_ONLY:
            return _openai(settings.OPENAI_FALLBACK_MODEL, 0.0, max_tokens=512)
        if settings.ANTHROPIC_API_KEY:
            return _anthropic(settings.HALLUCINATION_MODEL, temperature=0.0, max_tokens=2048)
        if settings.OPENAI_API_KEY or settings.LITELLM_URL:
            _warn_once(
                "hallucination_no_anthropic",
                "Anthropic API key not set for hallucination checker, falling back to OpenAI",
            )
            return _openai(settings.OPENAI_FALLBACK_MODEL, 0.0, max_tokens=512)

        _warn_once("hallucination_placeholder", "No LLM keys configured for hallucination, using PlaceholderChatModel")
        return PlaceholderChatModel(model_name="placeholder-hallucination")

    @staticmethod
    def get_high_stakes_hallucination_llm():
        """High-stakes hallucination check for threshold breaches."""
        if settings.LLM_PROVIDER_TYPE == "placeholder":
            return PlaceholderChatModel(model_name="placeholder-high-stakes")
        if settings.LLM_PROVIDER_TYPE == "open_source":
            return _open_source(model=settings.OPEN_SOURCE_LLM_MODEL, temperature=0.0, max_tokens=2048)
        if settings.LLM_PROVIDER_TYPE == "closed_source":
            if settings.CLOSED_SOURCE_LLM_PROVIDER == "anthropic" and settings.ANTHROPIC_API_KEY:
                return _anthropic(settings.CLOSED_SOURCE_LLM_MODEL, temperature=0.0, max_tokens=2048)
            return _openai(settings.CLOSED_SOURCE_LLM_MODEL, temperature=0.0, max_tokens=512)

        if settings.FORCE_OPENAI_ONLY:
            return _openai(settings.OPENAI_FALLBACK_MODEL, 0.0, max_tokens=512)
        if settings.ANTHROPIC_API_KEY:
            return _anthropic(settings.HIGH_STAKES_HALLUCINATION_MODEL, temperature=0.0, max_tokens=2048)
        return LLMFactory.get_hallucination_llm()

    @staticmethod
    def get_research_decompose_llm():
        """Research-agent decompose step."""
        if settings.LLM_PROVIDER_TYPE == "placeholder":
            return PlaceholderChatModel(model_name="placeholder-decompose")
        if settings.LLM_PROVIDER_TYPE == "open_source":
            return _open_source(model=settings.OPEN_SOURCE_LLM_MODEL, temperature=0.0, max_tokens=2048)
        if settings.LLM_PROVIDER_TYPE == "closed_source":
            return _llm_for_task(settings.CLOSED_SOURCE_LLM_MODEL, temperature=0.0, max_tokens=2048)

        if settings.FORCE_OPENAI_ONLY:
            return _openai(settings.OPENAI_FALLBACK_MODEL, 0.0, max_tokens=2048)
        if not settings.ANTHROPIC_API_KEY and settings.RESEARCH_AGENT_DECOMPOSE_MODEL.startswith("claude-"):
            if settings.OPENAI_API_KEY or settings.LITELLM_URL:
                _warn_once("decompose_no_anthropic", "Anthropic key missing, falling back to OpenAI for decompose")
                return _openai(settings.OPENAI_FALLBACK_MODEL, 0.0, max_tokens=2048)
        return _llm_for_task(settings.RESEARCH_AGENT_DECOMPOSE_MODEL, temperature=0.0, max_tokens=2048)

    @staticmethod
    def get_research_sufficiency_llm():
        """Research-agent sufficiency-judge step."""
        if settings.LLM_PROVIDER_TYPE == "placeholder":
            return PlaceholderChatModel(model_name="placeholder-sufficiency")
        if settings.LLM_PROVIDER_TYPE == "open_source":
            return _open_source(model=settings.OPEN_SOURCE_LLM_MODEL, temperature=0.0, max_tokens=2048)
        if settings.LLM_PROVIDER_TYPE == "closed_source":
            return _llm_for_task(settings.CLOSED_SOURCE_LLM_MODEL, temperature=0.0, max_tokens=2048)

        if settings.FORCE_OPENAI_ONLY:
            return _openai(settings.OPENAI_FALLBACK_MODEL, 0.0, max_tokens=2048)
        if not settings.ANTHROPIC_API_KEY and settings.RESEARCH_AGENT_SUFFICIENCY_MODEL.startswith("claude-"):
            if settings.OPENAI_API_KEY or settings.LITELLM_URL:
                _warn_once("sufficiency_no_anthropic", "Anthropic key missing, falling back to OpenAI for sufficiency")
                return _openai(settings.OPENAI_FALLBACK_MODEL, 0.0, max_tokens=2048)
        return _llm_for_task(settings.RESEARCH_AGENT_SUFFICIENCY_MODEL, temperature=0.0, max_tokens=2048)

    @staticmethod
    def get_research_synthesize_llm():
        """Research-agent synthesize step."""
        if settings.LLM_PROVIDER_TYPE == "placeholder":
            return PlaceholderChatModel(model_name="placeholder-synthesize")
        if settings.LLM_PROVIDER_TYPE == "open_source":
            return _open_source(model=settings.OPEN_SOURCE_LLM_MODEL, temperature=0.1, max_tokens=2048)
        if settings.LLM_PROVIDER_TYPE == "closed_source":
            return _llm_for_task(settings.CLOSED_SOURCE_LLM_MODEL, temperature=0.1, max_tokens=2048)

        if settings.FORCE_OPENAI_ONLY:
            return _openai(settings.OPENAI_FALLBACK_MODEL, 0.1, max_tokens=2048)
        if not settings.ANTHROPIC_API_KEY and settings.RESEARCH_AGENT_SYNTHESIZE_MODEL.startswith("claude-"):
            if settings.OPENAI_API_KEY or settings.LITELLM_URL:
                _warn_once("synthesize_no_anthropic", "Anthropic key missing, falling back to OpenAI for synthesize")
                return _openai(settings.OPENAI_FALLBACK_MODEL, 0.1, max_tokens=2048)
        return _llm_for_task(settings.RESEARCH_AGENT_SYNTHESIZE_MODEL, temperature=0.1, max_tokens=2048)
