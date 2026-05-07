"""
Utility functions: model factory, retry wrapper, output validation.

Provides factory functions for creating LLM model instances and Langfuse
clients. All modules should use these factories instead of constructing
clients directly — this keeps API keys and config in one place.
"""

import re
import time
import random
import logging
from typing import Callable

from langfuse import Langfuse, observe, get_client
from strands.models.openai import OpenAIModel

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    MODEL_PARAMS,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    LANGFUSE_HOST,
    MAX_RETRIES,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Langfuse singleton — initialised by init_langfuse() in main.py
# ---------------------------------------------------------------------------
_langfuse_client: Langfuse | None = None
_session_id: str = ""


def init_langfuse(session_id: str) -> None:
    """Initialise the module-level Langfuse client and store the session ID.

    Creates a Langfuse v3 client instance (which configures the get_client()
    singleton) so all @observe-decorated calls are tagged to this session.

    Args:
        session_id: The session ID for this pipeline run (used to group all traces).
    """
    global _langfuse_client, _session_id
    _session_id = session_id
    _langfuse_client = Langfuse(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        host=LANGFUSE_HOST,
    )
    logger.info("Langfuse initialised — session: %s", session_id)


def flush_langfuse() -> None:
    """Flush all pending Langfuse traces to the server."""
    try:
        get_client().flush()
    except Exception:
        pass
    if _langfuse_client:
        _langfuse_client.flush()
    logger.info("Langfuse traces flushed.")

UUID_PATTERN = re.compile(
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
    re.IGNORECASE,
)


class ContextOverflowError(Exception):
    """Raised on HTTP 400 — signals caller to halve batch and retry."""


def make_model(model_id: str, max_tokens: int | None = None) -> OpenAIModel:
    """Create an OpenRouter-routed OpenAI-compatible model instance.

    Args:
        model_id: Model identifier, e.g. 'qwen/qwen3-30b-a3b'.
        max_tokens: Override default output token limit.

    Returns:
        Configured OpenAIModel instance.
    """
    return OpenAIModel(
        client_args={
            "api_key": OPENROUTER_API_KEY,
            "base_url": OPENROUTER_BASE_URL,
        },
        model_id=model_id,
        params={
            "max_tokens": max_tokens or MODEL_PARAMS["max_tokens"],
            "temperature": MODEL_PARAMS["temperature"],
        },
    )


def estimate_tokens(text: str) -> int:
    """Rough token count estimate: 4 chars ≈ 1 token.

    Args:
        text: Input string.

    Returns:
        Estimated token count.
    """
    return len(text) // 4


def validate_output(
    response_text: str,
    expected_ids: set[str],
) -> tuple[set[str], set[str]]:
    """Check which expected transaction UUIDs appear in the response.

    Args:
        response_text: Raw LLM output text.
        expected_ids: Set of transaction UUIDs that should appear.

    Returns:
        Tuple of (found_ids, missing_ids) — both lowercased sets.
    """
    found_ids = {m.lower() for m in UUID_PATTERN.findall(response_text)}
    expected_lower = {e.lower() for e in expected_ids}
    found = found_ids & expected_lower
    missing = expected_lower - found
    return found, missing


def call_with_retry(
    call_fn: Callable[[], str],
    label: str = "llm_call",
    max_retries: int = MAX_RETRIES,
) -> str:
    """Call call_fn() with retry logic for all API failure modes.

    Raises ContextOverflowError on HTTP 400 so the caller can halve
    the batch and retry — never falls back to non-LLM logic.

    Args:
        call_fn: Zero-argument callable that returns the LLM response string.
        label: Identifier for log messages (e.g. 'comms_agent').
        max_retries: Maximum retry attempts for transient errors.

    Returns:
        Raw response string from the LLM.

    Raises:
        ContextOverflowError: On HTTP 400 (context window exceeded).
        Exception: After max_retries exhausted for other errors.
    """
    import openai

    for attempt in range(max_retries + 1):
        try:
            return call_fn()
        except openai.BadRequestError as e:
            logger.warning("%s: context overflow (400) on attempt %d: %s",
                           label, attempt + 1, e)
            raise ContextOverflowError(str(e)) from e
        except openai.RateLimitError as e:
            wait = (2 ** attempt) + random.uniform(0, 1)
            logger.warning("%s: rate limited (429), waiting %.1fs (attempt %d/%d)",
                           label, wait, attempt + 1, max_retries)
            if attempt < max_retries:
                time.sleep(wait)
            else:
                raise
        except Exception as e:
            wait = (2 ** attempt) + random.uniform(0, 1)
            logger.warning("%s: %s on attempt %d/%d, retrying in %.1fs",
                           label, type(e).__name__, attempt + 1, max_retries, wait)
            if attempt < max_retries:
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"{label}: exhausted {max_retries} retries")  # unreachable


@observe(as_type="generation")
def _traced_agent_call(agent, prompt: str, model_id: str) -> str:
    """Single Strands agent call wrapped in a Langfuse generation span.

    Args:
        agent: Configured strands Agent instance.
        prompt: User prompt to pass to the agent.
        model_id: Model identifier — reported to Langfuse for cost tracking.

    Returns:
        String response from the agent.
    """
    lf = get_client()
    if _session_id:
        lf.update_current_trace(session_id=_session_id)
    lf.update_current_generation(
        model=model_id,
        input=[{"role": "user", "content": prompt[:500]}],
    )

    result = agent(prompt)
    response = str(result)

    invocation = result.metrics.latest_agent_invocation
    usage = invocation.usage if invocation else result.metrics.accumulated_usage
    lf.update_current_generation(
        model=model_id,
        output=response,
        usage_details={
            "input": usage.get("inputTokens", 0),
            "output": usage.get("outputTokens", 0),
            "total": usage.get("totalTokens", 0),
        },
    )

    return response


def call_agent_with_retry(
    agent,
    prompt: str,
    model_id: str,
    label: str = "agent_call",
    max_retries: int = MAX_RETRIES,
) -> str:
    """Call a Strands agent with Langfuse tracing and retry logic.

    Replaces the old ``call_with_retry(lambda: str(agent(prompt)), ...)`` pattern
    so that token metrics are captured before the result is stringified.

    Args:
        agent: Configured strands Agent instance.
        prompt: User prompt to pass to the agent.
        model_id: Model identifier for Langfuse cost tracking.
        label: Label for log messages.
        max_retries: Maximum retry attempts for transient errors.

    Returns:
        Raw response string.

    Raises:
        ContextOverflowError: On HTTP 400 (context window exceeded).
        Exception: After max_retries exhausted for other errors.
    """
    import openai

    for attempt in range(max_retries + 1):
        try:
            return _traced_agent_call(agent, prompt, model_id)
        except openai.BadRequestError as e:
            logger.warning("%s: context overflow (400) on attempt %d: %s",
                           label, attempt + 1, e)
            raise ContextOverflowError(str(e)) from e
        except openai.RateLimitError as e:
            wait = (2 ** attempt) + random.uniform(0, 1)
            logger.warning("%s: rate limited (429), waiting %.1fs (attempt %d/%d)",
                           label, wait, attempt + 1, max_retries)
            if attempt < max_retries:
                time.sleep(wait)
            else:
                raise
        except Exception as e:
            wait = (2 ** attempt) + random.uniform(0, 1)
            logger.warning("%s: %s on attempt %d/%d, retrying in %.1fs",
                           label, type(e).__name__, attempt + 1, max_retries, wait)
            if attempt < max_retries:
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"{label}: exhausted {max_retries} retries")  # unreachable


def make_langfuse_client() -> Langfuse:
    """Create an initialized Langfuse client for tracing.

    Returns:
        Configured Langfuse client.

    Raises:
        ValueError: If Langfuse credentials are missing from environment.
    """
    if not all([LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY]):
        raise ValueError("Missing Langfuse credentials in .env")
    return Langfuse(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        host=LANGFUSE_HOST,
    )
