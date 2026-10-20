import logging

from langchain_core.messages import HumanMessage

from src.config.prompts import ROUTER_PROMPT
from src.config.settings import settings
from src.models.schemas import RouterDecision
from src.models.state import RAGState
from src.services.decider_2b import decider_2b
from src.services.llm_factory import LLMFactory

logger = logging.getLogger(__name__)


def router_node(state: RAGState) -> dict:
    """Classify query intent, complexity, and system tier using Strand Decider 2B.

    Emits `query_intent`, `query_complexity`, `system_tier`, and `decider_confidence`.
    """
    query = state.get("sanitized_query", "")
    if not query:
        return {
            "query_intent": "clarification",
            "query_complexity": None,
            "system_tier": "system_1",
            "decider_confidence": 1.0,
            "decider_model": settings.DECIDER_2B_MODEL,
        }

    # System 1: Strand Decider 2B fast classification
    if settings.ENABLE_SYSTEM_ONE_DECIDER:
        try:
            decision = decider_2b.classify(query, target_company=state.get("target_company"))
            logger.info(
                f"StrandDecider2B decision: intent={decision.intent}, complexity={decision.complexity}, "
                f"tier={decision.system_tier}, confidence={decision.confidence:.2f}"
            )
            try:
                from src.services.event_log import emit
                emit(
                    "router",
                    intent=decision.intent,
                    complexity=decision.complexity,
                    system_tier=decision.system_tier,
                    confidence=decision.confidence,
                    model=decision.decider_model,
                )
            except Exception:
                pass
            return {
                "query_intent": decision.intent,
                "query_complexity": decision.complexity,
                "system_tier": decision.system_tier,
                "decider_confidence": decision.confidence,
                "decider_model": decision.decider_model,
            }
        except Exception as e:
            logger.warning(f"StrandDecider2B failed, falling back to LLMFactory: {e}")

    # Fallback path via LLMFactory if decider is disabled or fails
    try:
        llm = LLMFactory.get_router_llm()
        structured_llm = llm.with_structured_output(RouterDecision)
        result: RouterDecision = structured_llm.invoke([HumanMessage(content=ROUTER_PROMPT.format(query=query))])
        logger.info(
            f"Router decision: intent={result.intent}, complexity={result.complexity}, reason={result.reason}"
        )
        try:
            from src.services.event_log import emit
            emit("router", intent=result.intent, complexity=result.complexity, reason=result.reason)
        except Exception:
            pass
        return {
            "query_intent": result.intent,
            "query_complexity": result.complexity,
            "system_tier": getattr(result, "system_tier", "system_2"),
            "decider_confidence": getattr(result, "confidence", 1.0),
            "decider_model": getattr(result, "decider_model", "router-llm"),
        }
    except Exception as e:
        logger.error(f"Router failed, defaulting to retrieval/simple_lookup: {e}")
        return {
            "query_intent": "retrieval",
            "query_complexity": "simple_lookup",
            "system_tier": "system_2",
            "decider_confidence": 0.5,
            "decider_model": "fallback",
        }
