"""Service health tracking for external API dependencies.

Records each external service call's success/failure and response time.
Stores in Redis with TTLs so stale status auto-expires to "unknown".
Falls back to in-memory dict if Redis is unavailable.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# In-memory fallback when Redis is unavailable
_health_store: dict[str, dict] = {}

# Service definitions with endpoints and metadata keys
SERVICE_DEFS = {
    "openmeteo": {
        "name": "Open-Meteo",
        "endpoint": "customer-api.open-meteo.com",
        "description": "Forecast API",
        "has_key": True,
        "key_env": "OPENMETEO_API_KEY",
    },
    "polymarket_gamma": {
        "name": "Polymarket Gamma",
        "endpoint": "gamma-api.polymarket.com",
        "description": "Market discovery",
        "has_key": False,
        "key_env": None,
    },
    "polymarket_clob": {
        "name": "Polymarket CLOB",
        "endpoint": "clob.polymarket.com",
        "description": "Order book / execution",
        "has_key": False,
        "key_env": None,
    },
    "claude": {
        "name": "Claude API",
        "endpoint": "api.anthropic.com",
        "description": "Trade reasoning",
        "has_key": True,
        "key_env": "ANTHROPIC_API_KEY",
    },
    "gemini": {
        "name": "Gemini API",
        "endpoint": "generativelanguage.googleapis.com",
        "description": "Red team",
        "has_key": True,
        "key_env": "GEMINI_API_KEY",
    },
    "gribstream": {
        "name": "GribStream",
        "endpoint": "gribstream.com",
        "description": "GraphCast AI model",
        "has_key": True,
        "key_env": "GRIBSTREAM_API_KEY",
    },
    "redis": {
        "name": "Redis",
        "endpoint": "localhost:6379",
        "description": "Live state cache",
        "has_key": False,
        "key_env": None,
    },
    "nws": {
        "name": "NWS API",
        "endpoint": "api.weather.gov",
        "description": "US weather alerts",
        "has_key": False,
        "key_env": None,
    },
    "openmeteo_archive": {
        "name": "Open-Meteo Archive",
        "endpoint": "archive-api.open-meteo.com",
        "description": "Trade resolution",
        "has_key": False,
        "key_env": None,
    },
    "noaa_cpc": {
        "name": "NOAA CPC",
        "endpoint": "cpc.ncep.noaa.gov",
        "description": "ENSO state",
        "has_key": False,
        "key_env": None,
    },
}

# TTLs for Redis keys (seconds)
_STATUS_TTL = 1800  # 30 minutes, status expires to "unknown" if not refreshed
_METRIC_TTL = 86400  # 24 hours for daily counters


def record_service_call(
    service_name: str,
    success: bool,
    response_time_ms: float | None = None,
    extra: dict | None = None,
) -> None:
    """Record an external service call result.

    Args:
        service_name: Key from SERVICE_DEFS (e.g. "openmeteo", "claude").
        success: Whether the call succeeded.
        response_time_ms: Response time in milliseconds (optional).
        extra: Additional metadata to store (e.g. {"markets_discovered": 42}).
    """
    now = datetime.now(timezone.utc).isoformat()
    data = {
        "last_call": now,
        "last_success": now if success else _health_store.get(service_name, {}).get("last_success"),
        "last_status": "ok" if success else "error",
        "last_response_ms": response_time_ms,
        "call_count": _health_store.get(service_name, {}).get("call_count", 0) + 1,
        "error_count": _health_store.get(service_name, {}).get("error_count", 0) + (0 if success else 1),
    }
    if extra:
        data.update(extra)

    # Store in Redis if available
    try:
        from weather_edge.live_state import set_json, get_json
        existing = get_json(f"svc:{service_name}") or {}
        # Preserve counters from Redis
        if "call_count" in existing:
            data["call_count"] = existing.get("call_count", 0) + 1
        if not success and "error_count" in existing:
            data["error_count"] = existing.get("error_count", 0) + 1
        if success:
            data["last_success"] = now
        elif existing.get("last_success"):
            data["last_success"] = existing["last_success"]
        # Merge extra from previous calls (preserve metrics like markets_discovered)
        for k, v in existing.items():
            if k not in data:
                data[k] = v
        set_json(f"svc:{service_name}", data, ttl=_STATUS_TTL)
    except Exception:
        pass

    # Always update in-memory fallback
    _health_store[service_name] = data


def _check_key_present(key_env: str | None) -> bool:
    """Check if an API key is configured."""
    if not key_env:
        return False
    val = os.environ.get(key_env, "")
    if val:
        return True
    # Also check pydantic settings
    try:
        from weather_edge.config import settings
        attr_name = key_env.lower()
        # Map env var names to settings attributes
        key_map = {
            "OPENMETEO_API_KEY": "openmeteo_api_key",
            "ANTHROPIC_API_KEY": "anthropic_api_key",
            "GEMINI_API_KEY": "gemini_api_key",
            "GRIBSTREAM_API_KEY": "gribstream_api_key",
        }
        attr = key_map.get(key_env, attr_name)
        return bool(getattr(settings, attr, ""))
    except Exception:
        return False


def _get_redis_info() -> dict:
    """Get Redis server info for the status tab."""
    try:
        from weather_edge.live_state import _get_redis
        r = _get_redis()
        if r is None:
            return {"status": "down", "keys_count": 0, "memory_used": "N/A"}
        info = r.info(section="memory")
        keys_count = r.dbsize()
        memory = info.get("used_memory_human", "N/A")
        return {
            "status": "ok",
            "keys_count": keys_count,
            "memory_used": memory,
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "keys_count": 0, "memory_used": "N/A"}


def _get_enso_info() -> dict:
    """Get cached ENSO state if available."""
    try:
        from weather_edge.analysis.enso_regime import _cached_enso
        if _cached_enso:
            return {
                "phase": _cached_enso.phase,
                "oni_value": _cached_enso.oni_value,
                "last_update": _cached_enso.fetched_at.isoformat(),
            }
    except Exception:
        pass
    return {"phase": "unknown", "oni_value": None, "last_update": None}


def get_service_status() -> dict:
    """Return health status for all monitored services.

    Returns a dict with service statuses, key presence, and metrics.
    Each service has: name, endpoint, status, last_call, last_success, key_present, metrics.
    """
    now = datetime.now(timezone.utc)
    services = []

    for svc_id, svc_def in SERVICE_DEFS.items():
        # Try Redis first, fall back to in-memory
        health_data = {}
        try:
            from weather_edge.live_state import get_json
            health_data = get_json(f"svc:{svc_id}") or {}
        except Exception:
            pass
        if not health_data:
            health_data = _health_store.get(svc_id, {})

        last_call = health_data.get("last_call")
        last_success = health_data.get("last_success")
        last_status = health_data.get("last_status")
        call_count = health_data.get("call_count", 0)
        error_count = health_data.get("error_count", 0)
        last_response_ms = health_data.get("last_response_ms")

        # Determine status color
        # green = OK in last 5min, yellow = OK in last 30min, red = failed or no data recently, grey = disabled
        key_present = None
        if svc_def["has_key"]:
            key_present = _check_key_present(svc_def["key_env"])
            if not key_present:
                status = "disabled"
            elif last_status == "ok" and last_success:
                status = _compute_freshness(last_success, now)
            elif last_status == "error":
                status = "error"
            else:
                status = "unknown"
        else:
            if last_status == "ok" and last_success:
                status = _compute_freshness(last_success, now)
            elif last_status == "error":
                status = "error"
            else:
                status = "unknown"

        entry = {
            "id": svc_id,
            "name": svc_def["name"],
            "endpoint": svc_def["endpoint"],
            "description": svc_def["description"],
            "status": status,
            "last_call": last_call,
            "last_success": last_success,
            "key_present": key_present,
            "call_count": call_count,
            "error_count": error_count,
            "last_response_ms": last_response_ms,
        }

        # Service-specific metrics
        if svc_id == "polymarket_gamma":
            entry["markets_discovered"] = health_data.get("markets_discovered", 0)
        elif svc_id == "claude":
            entry["decisions_today"] = health_data.get("decisions_today", 0)
        elif svc_id == "gemini":
            entry["dissents_today"] = health_data.get("dissents_today", 0)
        elif svc_id == "gribstream":
            entry["credits_remaining"] = health_data.get("credits_remaining")
        elif svc_id == "openmeteo":
            try:
                from weather_edge.config import settings
                entry["paid_tier"] = settings.openmeteo_paid_tier or bool(settings.openmeteo_api_key)
            except Exception:
                entry["paid_tier"] = False

        services.append(entry)

    # Special handling: Redis health check (live probe)
    redis_entry = next((s for s in services if s["id"] == "redis"), None)
    if redis_entry:
        redis_info = _get_redis_info()
        redis_entry["status"] = "green" if redis_info["status"] == "ok" else "error"
        redis_entry["keys_count"] = redis_info["keys_count"]
        redis_entry["memory_used"] = redis_info["memory_used"]

    # Special handling: ENSO state
    enso_entry = next((s for s in services if s["id"] == "noaa_cpc"), None)
    if enso_entry:
        enso_info = _get_enso_info()
        enso_entry["enso_phase"] = enso_info["phase"]
        enso_entry["oni_value"] = enso_info["oni_value"]
        if enso_info["last_update"]:
            enso_entry["last_success"] = enso_info["last_update"]
            enso_entry["last_call"] = enso_info["last_update"]
            enso_entry["status"] = _compute_freshness(enso_info["last_update"], now)

    # Summary
    healthy_count = sum(1 for s in services if s["status"] in ("green", "yellow"))
    total_active = sum(1 for s in services if s["status"] != "disabled")

    return {
        "services": services,
        "summary": {
            "healthy": healthy_count,
            "total": total_active,
            "total_services": len(services),
        },
        "timestamp": now.isoformat(),
    }


def _compute_freshness(last_success_iso: str, now: datetime) -> str:
    """Determine status color based on time since last success."""
    try:
        last = datetime.fromisoformat(last_success_iso)
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        delta = (now - last).total_seconds()
        if delta < 300:  # 5 minutes
            return "green"
        elif delta < 1800:  # 30 minutes
            return "yellow"
        else:
            return "red"
    except (ValueError, TypeError):
        return "unknown"
