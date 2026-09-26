"""Retry with exponential backoff for external API calls.

Every external call in a live trading system must handle transient failures.
A single 502 should not silently skip an entire cycle.

Only *transient* failures are retried: transport errors, timeouts, HTTP 429 and
5xx responses, and SQLite lock contention. A 4xx (bad request, bad key) or a
programming error (KeyError, TypeError, ...) is raised on the first attempt,
retrying those only burns time and API quota.

Exception text is passed through :func:`redact` before it is logged, because
``httpx.HTTPStatusError`` messages embed the full request URL, query string
included, and some APIs take credentials there.
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import sqlite3
from collections.abc import Callable
from typing import TypeVar

import httpx

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Default: 3 attempts, 1s base delay, 2x backoff, ±25% jitter
DEFAULT_ATTEMPTS = 3
DEFAULT_BASE_DELAY = 1.0
DEFAULT_BACKOFF = 2.0
DEFAULT_JITTER = 0.25

# ---------------------------------------------------------------------------
# Transient-error classification
# ---------------------------------------------------------------------------

_RETRYABLE_STATUS = frozenset({408, 425, 429})


def _status_is_transient(status: int | None) -> bool:
    if status is None:
        return False
    return status in _RETRYABLE_STATUS or 500 <= status < 600


def is_transient(exc: BaseException) -> bool:
    """Return True if ``exc`` looks like a failure worth retrying."""
    if isinstance(exc, httpx.HTTPStatusError):
        return _status_is_transient(exc.response.status_code)
    if isinstance(exc, (httpx.TransportError, httpx.TimeoutException)):
        return True
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True
    if isinstance(exc, sqlite3.OperationalError):
        # "database is locked" / "database is busy", contention, not a bug
        msg = str(exc).lower()
        return "locked" in msg or "busy" in msg
    try:  # requests is an optional dependency (py-clob-client uses it)
        import requests

        if isinstance(exc, requests.HTTPError):
            resp = getattr(exc, "response", None)
            return _status_is_transient(getattr(resp, "status_code", None))
        if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
            return True
    except ImportError:  # pragma: no cover - requests normally installed
        pass
    # Third-party API wrappers (e.g. py-clob-client) carry a status code attr
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return _status_is_transient(status)
    if type(exc).__name__ == "PolyApiException" and status is None:
        # py-clob-client raises this with no status for request/transport failures
        return True
    return False


# ---------------------------------------------------------------------------
# Secret redaction for log output
# ---------------------------------------------------------------------------

_URL_QUERY_RE = re.compile(r"(https?://[^\s?#'\"<>]+)\?[^\s'\"<>]*")
_KV_SECRET_RE = re.compile(
    r"(?i)\b((?:api[_-]?)?key|token|secret|signature|passphrase|password)=([^&\s'\"]+)",
)
_SECRET_NAME_HINTS = ("key", "secret", "token", "passphrase", "password")


def _known_secrets() -> list[str]:
    """Secret values from the environment and settings, longest first."""
    values: set[str] = set()
    for name, val in os.environ.items():
        if val and len(val) >= 8 and any(h in name.lower() for h in _SECRET_NAME_HINTS):
            values.add(val)
    try:
        from weather_edge.config import settings

        dump = settings.model_dump() if hasattr(settings, "model_dump") else {}
        for name, val in dump.items():
            if (
                isinstance(val, str)
                and len(val) >= 8
                and any(h in name.lower() for h in _SECRET_NAME_HINTS)
            ):
                values.add(val)
    except Exception:
        pass
    return sorted(values, key=len, reverse=True)


def redact(text: object) -> str:
    """Strip URL query strings, ``key=value`` credentials and known secrets."""
    s = str(text)
    s = _URL_QUERY_RE.sub(r"\1?<redacted>", s)
    s = _KV_SECRET_RE.sub(r"\1=<redacted>", s)
    for secret in _known_secrets():
        if secret in s:
            s = s.replace(secret, "<redacted>")
    return s


class RedactingFilter(logging.Filter):
    """Redact a log record's text and traceback (attach to output handlers).

    httpx exception text embeds the request URL, and some APIs (Open-Meteo)
    only take their key as a query parameter, so any ``logger.x("%s", e)``
    or ``exc_info=True`` anywhere could print a key. The record is shared by
    every handler, so it is only ever made safer: the message is rendered
    and redacted, and a redacted traceback is pre-rendered into exc_text
    (formatters use exc_text when it is set); exc_info is left in place.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg, record.args = redact(record.getMessage()), None
        if record.exc_info and not record.exc_text:
            record.exc_text = redact(logging.Formatter().formatException(record.exc_info))
        return True


# ---------------------------------------------------------------------------
# Retry loops
# ---------------------------------------------------------------------------


def _next_delay(attempt: int, base_delay: float, backoff: float, jitter: float) -> float:
    delay = base_delay * (backoff ** (attempt - 1))
    return delay * (1 + random.uniform(-jitter, jitter))


async def retry_async(
    fn,
    *args,
    attempts: int = DEFAULT_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY,
    backoff: float = DEFAULT_BACKOFF,
    jitter: float = DEFAULT_JITTER,
    label: str = "",
    retry_on: Callable[[BaseException], bool] = is_transient,
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
        retry_on: Predicate deciding whether an exception is retryable
            (default :func:`is_transient`). Non-retryable errors raise at once.

    Returns:
        The return value of fn.

    Raises:
        The last exception if all attempts fail, or the first non-retryable one.
    """
    tag = label or getattr(fn, "__name__", "call")

    for attempt in range(1, attempts + 1):
        try:
            return await fn(*args, **kwargs)
        except Exception as exc:
            if not retry_on(exc):
                logger.warning(
                    "RETRY ABORT [%s]: non-transient %s on attempt %d, %s",
                    tag, type(exc).__name__, attempt, redact(exc),
                )
                raise
            if attempt == attempts:
                logger.error(
                    "RETRY EXHAUSTED [%s]: %d/%d attempts failed, %s",
                    tag, attempt, attempts, redact(exc),
                )
                raise
            delay = _next_delay(attempt, base_delay, backoff, jitter)
            logger.warning(
                "RETRY [%s]: attempt %d/%d failed (%s), retrying in %.1fs",
                tag, attempt, attempts, redact(exc), delay,
            )
            await asyncio.sleep(delay)

    raise RuntimeError("unreachable")  # pragma: no cover


def retry_sync(
    fn,
    *args,
    attempts: int = DEFAULT_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY,
    backoff: float = DEFAULT_BACKOFF,
    jitter: float = DEFAULT_JITTER,
    label: str = "",
    retry_on: Callable[[BaseException], bool] = is_transient,
    **kwargs,
):
    """Call a sync function with exponential backoff retry.

    For DB writes and other synchronous operations. Same transient-only
    semantics as :func:`retry_async`.
    """
    import time

    tag = label or getattr(fn, "__name__", "call")

    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if not retry_on(exc):
                logger.warning(
                    "RETRY ABORT [%s]: non-transient %s on attempt %d, %s",
                    tag, type(exc).__name__, attempt, redact(exc),
                )
                raise
            if attempt == attempts:
                logger.error(
                    "RETRY EXHAUSTED [%s]: %d/%d attempts failed, %s",
                    tag, attempt, attempts, redact(exc),
                )
                raise
            delay = _next_delay(attempt, base_delay, backoff, jitter)
            logger.warning(
                "RETRY [%s]: attempt %d/%d failed (%s), retrying in %.1fs",
                tag, attempt, attempts, redact(exc), delay,
            )
            time.sleep(delay)

    raise RuntimeError("unreachable")  # pragma: no cover
