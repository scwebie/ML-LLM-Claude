"""Shared agent orchestration primitives.

Design note -- why this is not literally the OpenAI Agents SDK
----------------------------------------------------------------
The OpenAI Agents SDK's PyPI distribution (``openai-agents``) imports as
``import agents``, which collides head-on with this project's *required*
top-level package name ``agents/`` (see the repository layout in the
project brief). Depending on it here would shadow -- or be shadowed by --
our own package on every ``import agents`` in the codebase.

Instead, this module implements a small, in-house orchestration layer that
mirrors the Agents SDK's core pattern (an agent takes structured input,
returns a validated Pydantic structured output) using the plain ``openai``
client directly. Every agent's *numeric scores are always computed
deterministically* from already-calculated features -- the optional LLM
call, when an API key is configured, only ever contributes a short
natural-language ``reasoning_summary`` gloss on top of those numbers. This
keeps the ML pipeline fully reproducible (numeric features never depend on
a non-deterministic network call) while still demonstrating the intended
LLM-enhanced-research-agent pattern end to end.

Agents never place orders. They only ever return ``AgentReport`` /
agent-specific structured outputs that flow into the Feature Store.
"""

from __future__ import annotations

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)

AGENT_VERSION = "0.1.0"


def clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def safe_tanh(value: float) -> float:
    import math

    if value != value:  # NaN check without importing numpy here
        return 0.0
    return math.tanh(value)


def maybe_llm_reasoning(system_prompt: str, user_prompt: str) -> str | None:
    """Best-effort optional LLM narrative. Returns None on any failure or if
    no API key is configured -- callers must treat this as purely additive
    and never depend on it for any numeric output."""
    if not settings.openai_api_key:
        return None
    try:
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=120,
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as exc:  # noqa: BLE001 - any LLM failure must never break the pipeline
        logger.warning("llm_reasoning_failed", error=str(exc))
        return None
