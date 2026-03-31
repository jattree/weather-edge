"""Retry with exponential backoff for external API calls.

Every external call in a live trading system must handle transient failures.
A single 502 should not silently skip an entire cycle.
"""
from __future__ import annotations

import asyncio
import logging
import random
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Default: 3 attempts, 1s base delay, 2x backoff, ±25% jitter
DEFAULT_ATTEMPTS = 3
DEFAULT_BASE_DELAY = 1.0
DEFAULT_BACKOFF = 2.0
DEFAULT_JITTER = 0.25


async def retry_async(
    fn,
    *args,
    attempts: int = DEFAULT_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY,
    backoff: float = DEFAULT_BACKOFF,
    jitter: float = DEFAULT_JITTER,
    label: str = "",
    **kwargs,
):
    """Call an async function with exponential backoff retry.

    Args:
        fn: Async callable to retry.
        attempts: Max number of attempts (default 3).
        base_delay: Initial delay in seconds (default 1.0).
        backoff: Multiplier per retry (default 2.0).
        jitter: Random ± fraction of delay (default 0.25).
        label: Human-readable name for logging.

    Returns:
        The return value of fn.

    Raises:
        The last exception if all attempts fail.
    """
    last_exc = None
    tag = label or getattr(fn, "__name__", "call")

    for attempt in range(1, attempts + 1):
        try:
            return await fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt == attempts:
                logger.error(
                    "RETRY EXHAUSTED [%s]: %d/%d attempts failed, %s",
                    tag, attempt, attempts, exc,
                )
                raise
            delay = base_delay * (backoff ** (attempt - 1))
            delay *= 1 + random.uniform(-jitter, jitter)
            logger.warning(
                "RETRY [%s]: attempt %d/%d failed (%s), retrying in %.1fs",
                tag, attempt, attempts, exc, delay,
            )
            await asyncio.sleep(delay)

    raise last_exc  # Unreachable, but satisfies type checker


def retry_sync(
    fn,
    *args,
    attempts: int = DEFAULT_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY,
    backoff: float = DEFAULT_BACKOFF,
    jitter: float = DEFAULT_JITTER,
    label: str = "",
    **kwargs,
):
    """Call a sync function with exponential backoff retry.

    For DB writes and other synchronous operations.
    """
    last_exc = None
    tag = label or getattr(fn, "__name__", "call")

    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt == attempts:
                logger.error(
                    "RETRY EXHAUSTED [%s]: %d/%d attempts failed, %s",
                    tag, attempt, attempts, exc,
                )
                raise
            delay = base_delay * (backoff ** (attempt - 1))
            delay *= 1 + random.uniform(-jitter, jitter)
            logger.warning(
                "RETRY [%s]: attempt %d/%d failed (%s), retrying in %.1fs",
                tag, attempt, attempts, exc, delay,
            )
            import time
            time.sleep(delay)

    raise last_exc
