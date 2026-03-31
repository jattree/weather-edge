"""Redis-backed live state for hot-path data.

Hot (Redis, <10ms reads):
- Heartbeat IDs for Polymarket CLOB (60s TTL)
- Live order book snapshots (5min TTL)
- Active signal state (prevents double-trading)
- Cycle lock (prevents concurrent execution)
- Latest dashboard state (WebSocket broadcast cache)

Cold (SQLite, permanent):
- Trade history
- AI decision logs
- Session data
- P&L records

Falls back gracefully to in-memory dicts if Redis is unavailable.
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_redis_client = None
_fallback_cache: dict[str, Any] = {}  # In-memory fallback if Redis is down


def _get_redis():
    """Lazy Redis connection, only connects when first used."""
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    try:
        import redis

        from weather_edge.config import settings
        _redis_client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        _redis_client.ping()
        logger.info(
            "Redis connected (%s:%d/%d)",
            settings.redis_host, settings.redis_port, settings.redis_db,
        )
        # Record Redis health (avoid circular import, write directly to in-memory store)
        try:
            from weather_edge.analysis.service_health import record_service_call
            record_service_call("redis", True)
        except Exception:
            logger.debug("Failed to record Redis health", exc_info=True)
        return _redis_client
    except Exception as e:
        logger.warning("Redis unavailable (%s), using in-memory fallback", e)
        _redis_client = None
        try:
            from weather_edge.analysis.service_health import record_service_call
            record_service_call("redis", False)
        except Exception:
            logger.debug("Failed to record Redis failure", exc_info=True)
        return None


def set_value(key: str, value: str, ttl: int | None = None) -> None:
    """Set a string value in Redis (or fallback cache)."""
    r = _get_redis()
    if r:
        try:
            r.set(key, value, ex=ttl)
            return
        except Exception:
            logger.debug("Redis set failed for %s, using fallback", key)
    _fallback_cache[key] = value


def get_value(key: str) -> str | None:
    """Get a string value from Redis (or fallback cache)."""
    r = _get_redis()
    if r:
        try:
            return r.get(key)
        except Exception:
            logger.debug("Redis get failed for %s, using fallback", key)
    return _fallback_cache.get(key)


def set_json(key: str, data: dict | list, ttl: int | None = None) -> None:
    """Set a JSON-serializable value."""
    set_value(key, json.dumps(data, default=str), ttl)


def get_json(key: str) -> dict | list | None:
    """Get a JSON value."""
    raw = get_value(key)
    if raw:
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            pass
    return None


def set_lock(key: str, ttl: int = 180) -> bool:
    """Try to acquire a lock. Returns True if acquired, False if already held."""
    r = _get_redis()
    if r:
        try:
            return bool(r.set(key, "locked", nx=True, ex=ttl))
        except Exception:
            logger.debug("Redis lock failed for %s, using fallback", key)
    # Fallback: simple in-memory check
    if key in _fallback_cache:
        return False
    _fallback_cache[key] = "locked"
    return True


def release_lock(key: str) -> None:
    """Release a lock."""
    r = _get_redis()
    if r:
        try:
            r.delete(key)
            return
        except Exception:
            logger.debug("Redis delete failed for %s, using fallback", key)
    _fallback_cache.pop(key, None)


# ---- Domain-specific helpers ----

def set_heartbeat(market_id: str, heartbeat_id: str) -> None:
    """Store Polymarket heartbeat ID with 55s TTL (must be fresh for orders)."""
    set_value(f"hb:{market_id}", heartbeat_id, ttl=55)


def get_heartbeat(market_id: str) -> str | None:
    """Get latest heartbeat ID for a market."""
    return get_value(f"hb:{market_id}")


def update_book(market_id: str, book_data: dict) -> None:
    """Cache order book snapshot for dashboard (5min TTL)."""
    set_json(f"book:{market_id}", book_data, ttl=300)


def get_book(market_id: str) -> dict | None:
    """Get cached order book for a market."""
    return get_json(f"book:{market_id}")


def set_signal_lock(city: str) -> bool:
    """Prevent double-trading on same city within a cycle (3min TTL)."""
    return set_lock(f"lock:trade:{city}", ttl=180)


def release_signal_lock(city: str) -> None:
    """Release city trade lock after cycle."""
    release_lock(f"lock:trade:{city}")


def cache_dashboard_state(state: dict) -> None:
    """Cache latest dashboard state for instant WebSocket delivery (60s TTL)."""
    set_json("dashboard:state", state, ttl=60)


def get_cached_dashboard_state() -> dict | None:
    """Get cached dashboard state (faster than recomputing)."""
    return get_json("dashboard:state")
