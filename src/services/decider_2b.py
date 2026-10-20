"""System 1 Fast Decider Service (Strands Decider 2B Client & Fast-Path Triage).

Implements the Dual-System (System 1 vs System 2) cognitive architecture:
- **System 1 (Strands Decider 2B)**: Low-latency intent & complexity triage.
  Directly integrates with local or remote small language model (SLM) endpoints
  (e.g., strands-decider:2b via Ollama, vLLM, or Hugging Face TGI at DECIDER_2B_BASE_URL).
- **Graceful Deterministic Fallback**: When the local SLM service is offline or unprovisioned,
  instantly fails over to sub-millisecond deterministic pattern matching to ensure continuous
  zero-downtime execution without cloud roundtrips.
- **System 2 (Deep Reasoning)**: The 18-node LangGraph orchestration engine with
  hybrid retrieval, cross-encoders, and quantitative decision modeling.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import httpx

from src.config.settings import settings
from src.models.schemas import RouterDecision

logger = logging.getLogger(__name__)


class StrandDecider2B:
    """System 1 Neural Decider Client (2B Architecture) with Deterministic Fallback."""

    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        confidence_threshold: float = 0.80,
        timeout_seconds: float = 0.35,
    ):
        self.model_name = model_name or getattr(settings, "DECIDER_2B_MODEL", "strand-decider-2b")
        self.base_url = (base_url or getattr(settings, "DECIDER_2B_BASE_URL", "http://localhost:11434/v1")).rstrip("/")
        self.confidence_threshold = confidence_threshold
        self.timeout_seconds = timeout_seconds

    def _query_slm_endpoint(self, query: str) -> RouterDecision | None:
        """Attempt to query local/hosted Strands Decider 2B endpoint over HTTP."""
        url = f"{self.base_url}/chat/completions"
        system_prompt = (
            "You are Strands Decider 2B, an ultra-fast System 1 classification model. "
            "Classify financial user queries into JSON format with keys: "
            "'intent' (retrieval | market_data | portfolio_risk | out_of_scope | clarification), "
            "'complexity' (simple_lookup | research_required), "
            "'system_tier' (system_1 | system_2), "
            "'confidence' (float 0.0-1.0), "
            "'reason' (short sentence)."
        )
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            "temperature": 0.0,
            "max_tokens": 100,
        }
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                resp = client.post(url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    parsed = json.loads(content)
                    return RouterDecision(
                        intent=parsed.get("intent", "retrieval"),
                        complexity=parsed.get("complexity", "simple_lookup"),
                        system_tier=parsed.get("system_tier", "system_1"),
                        confidence=float(parsed.get("confidence", 0.90)),
                        reason=parsed.get("reason", "Strands Decider 2B classification"),
                        decider_model=self.model_name,
                    )
        except Exception as exc:
            logger.debug("Strands Decider 2B endpoint query bypassed (%s: %s)", type(exc).__name__, exc)
        return None

    def classify(self, query: str, target_company: str | None = None) -> RouterDecision:
        """Classify query intent and complexity via SLM with microsecond fallback."""
        start_t = time.perf_counter()
        q_clean = query.strip()
        q_lower = q_clean.lower()

        if not q_clean:
            return RouterDecision(
                intent="clarification",
                complexity="simple_lookup",
                system_tier="system_1",
                confidence=1.0,
                reason="Empty query requires clarification.",
                decider_model=self.model_name,
            )

        # 1. Try real SLM endpoint if reachable within timeout window
        slm_decision = self._query_slm_endpoint(q_clean)
        if slm_decision:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            logger.info("Strands Decider 2B (neural): classified in %.2fms -> %s", elapsed_ms, slm_decision.intent)
            return slm_decision

        # 2. Deterministic fast-path triage fallback (sub-millisecond)
        return self._deterministic_triage(q_clean, q_lower, start_t)

    def _deterministic_triage(self, q_clean: str, q_lower: str, start_t: float) -> RouterDecision:
        """Deterministic sub-millisecond rule-based classifier fallback."""

        # 1. Out of Scope Non-Financial Detection
        non_financial_patterns = [
            r"\b(recipe|weather|movie|poem|song|joke|sports|football|cricket|game)\b",
            r"\b(who is the president of|capital of|write code for|how to cook)\b",
        ]
        if any(re.search(pat, q_lower) for pat in non_financial_patterns):
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            logger.info(f"StrandDecider2B: out_of_scope classified in {elapsed_ms:.2f}ms")
            return RouterDecision(
                intent="out_of_scope",
                complexity="simple_lookup",
                system_tier="system_1",
                confidence=0.98,
                reason="Query is not related to asset management, equities, or SEC filings.",
                decider_model=self.model_name,
            )

        # 2. Portfolio Risk & Mandate Concentration
        portfolio_patterns = [
            r"\b(portfolio risk|concentration risk|hhi|herfindahl|portfolio weight|rebalance)\b",
            r"\b(add|trim|allocate|reduce)\s+[0-9]+(?:\.[0-9]+)?\s*%\s*(?:to|in|for)?\s*[a-z]{1,5}\b",
            r"\b(top-5 concentration|single-name limit)\b",
        ]
        if any(re.search(pat, q_lower) for pat in portfolio_patterns):
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            logger.info(f"StrandDecider2B: portfolio_risk classified in {elapsed_ms:.2f}ms")
            return RouterDecision(
                intent="portfolio_risk",
                complexity="simple_lookup",
                system_tier="system_1",
                confidence=0.96,
                reason="Query requests portfolio risk metrics or rebalancing governance.",
                decider_model=self.model_name,
            )

        # 3. Real-Time Market Data & Trading Multiples
        market_patterns = [
            r"\b(live price|current price|stock price|share price|beta|market cap)\b",
            r"\b(52-week|52 week|trailing p/e|forward p/e|pe ratio|p/e ratio)\b",
            r"\b(quote for|trading multiple)\b",
        ]
        is_market_query = any(re.search(pat, q_lower) for pat in market_patterns)
        is_sec_document_query = any(k in q_lower for k in ("10-k", "10k", "10-q", "10q", "sec filing", "annual report", "audited footnote"))

        if is_market_query and not is_sec_document_query:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            logger.info(f"StrandDecider2B: market_data classified in {elapsed_ms:.2f}ms")
            return RouterDecision(
                intent="market_data",
                complexity="simple_lookup",
                system_tier="system_1",
                confidence=0.95,
                reason="Query targets live market pricing or trading multiples via market data tool.",
                decider_model=self.model_name,
            )

        # 4. Deep SEC Retrieval & Research Agent Decision
        # Check if research_required (multi-year comparison, segment breakdown, synthesis)
        research_indicators = [
            r"\b(compare|versus|vs|trend|across|over the last|between)\b",
            r"\b(segment breakdown|operating margin expansion|working capital bridge)\b",
            r"\b(cagr|piotroski|altman z|bankruptcy probability|comprehensive analysis)\b",
        ]
        is_research = any(re.search(pat, q_lower) for pat in research_indicators)
        complexity = "research_required" if is_research else "simple_lookup"

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        logger.info(f"StrandDecider2B: retrieval ({complexity}) classified in {elapsed_ms:.2f}ms")
        return RouterDecision(
            intent="retrieval",
            complexity=complexity,
            system_tier="system_2",
            confidence=0.92 if is_research else 0.88,
            reason=f"Financial query requiring {'multi-step synthesis' if is_research else 'SEC filing extraction'}.",
            decider_model=self.model_name,
        )


# Global singleton instance
decider_2b = StrandDecider2B()
