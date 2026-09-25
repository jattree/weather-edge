"""Claude reasoning layer, the Meteorologist.

Claude is MARKET-BLIND. It assesses the atmosphere, not the trade.
Its job is to answer: "Is this forecast trustworthy?"

Claude provides:
1. Physical plausibility assessment of model consensus
2. Outlier diagnosis (real mesoscale feature vs model error)
3. Forecast confidence based on meteorological factors
4. Risk factors: specific physical bust mechanisms
5. Weather insight for the dashboard

Gemini (the Quant) handles market skepticism, sizing, and risk management.
The two AIs must have UNCORRELATED errors to maximize signal quality.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import date

import httpx

from weather_edge.analysis.edge import Signal

logger = logging.getLogger(__name__)

# ---- Decision history for dashboard AI Decisions tab ----
_decision_history: list[dict] = []
MAX_DECISION_HISTORY = 200


def record_decision(reasoning: "TradeReasoning") -> None:
    """Record a Claude trade decision for the dashboard."""
    from datetime import datetime, timezone

    entry = {
        "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
        "city": (
            reasoning.signal.city_id.upper()
            if isinstance(reasoning.signal.city_id, str)
            else reasoning.signal.city_id
        ),
        "decision": "TRADE" if reasoning.should_trade else "SKIP",
        "signal": reasoning.signal.description[:60] if reasoning.signal.description else "",
        "adjustment": reasoning.confidence_adjustment,
        "rationale": reasoning.rationale,
        "risk_factors": reasoning.risk_factors,
    }
    _decision_history.insert(0, entry)
    # Trim to max size
    while len(_decision_history) > MAX_DECISION_HISTORY:
        _decision_history.pop()


def get_decisions() -> list[dict]:
    """Return a copy of the decision history (most recent first)."""
    return list(_decision_history)


def clear_decisions() -> None:
    """Clear the decision history (e.g. on new session)."""
    _decision_history.clear()


def _get_api_key() -> str:
    """Get API key from env or .env file via pydantic-settings."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        try:
            from weather_edge.config import settings
            key = getattr(settings, "anthropic_api_key", "")
        except Exception:
            pass
    return key

ANTHROPIC_API_KEY = _get_api_key()

DEFAULT_CLAUDE_MODEL = "claude-sonnet-5"
# Enough room for the JSON verdict plus any adaptive thinking the model does;
# a truncated reply is treated as a failed review (fail closed), so do not
# lowball this.
CLAUDE_MAX_TOKENS = 4096


def _get_model() -> str:
    """Claude model id, overridable via settings.claude_model / CLAUDE_MODEL."""
    model = os.environ.get("CLAUDE_MODEL", "")
    if not model:
        try:
            from weather_edge.config import settings
            model = getattr(settings, "claude_model", "") or ""
        except Exception:
            model = ""
    return model or DEFAULT_CLAUDE_MODEL


CLAUDE_MODEL = _get_model()

# Contract: warn at module load if AI keys are missing
def _check_ai_keys_at_load() -> None:
    from weather_edge.analysis.contracts import validate_ai_keys_present
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        try:
            from weather_edge.config import settings as _s
            gemini_key = getattr(_s, "gemini_api_key", "")
        except Exception:
            pass
    result = validate_ai_keys_present(ANTHROPIC_API_KEY, gemini_key)
    if not result.valid:
        logger.warning(
            "CONTRACT [%s]: %s, AI reasoning will be degraded",
            result.code, result.error,
        )

_check_ai_keys_at_load()


@dataclass
class TradeReasoning:
    """Claude's analysis of a potential trade."""
    signal: Signal
    should_trade: bool
    confidence_adjustment: float  # Multiplier: 0.5-1.5
    rationale: str  # Human-readable explanation
    risk_factors: list[str]
    weather_insight: str  # What Claude sees in the pattern
    # False when the reply could not be parsed / was truncated / refused. Such
    # a review always has should_trade=False (fail closed) but is NOT a
    # meteorological veto, so it is not remembered across cycles.
    review_ok: bool = True


SYSTEM_PROMPT = (
    "You are a forensic meteorologist. Your ONLY job is to assess "
    "what the atmosphere will do. You are MARKET-BLIND, you do not "
    "care about prices, edges, or whether a trade \"looks too good.\" "
    "That is someone else's job.\n\n"
    "You receive model forecasts from 6-8 weather models for a "
    "specific city/date. Assess:\n\n"
    "1. PHYSICAL PLAUSIBILITY: Do the models agree for the right "
    "physical reasons? Or are they clustering on a shared bias "
    "(e.g., all using the same SST boundary condition)?\n"
    "2. MODEL TRUST: Which models should be weighted more for THIS "
    "city? (HRRR for US short-range, UKV for UK, ECMWF for 3+ day "
    "global)\n"
    "3. OUTLIER DIAGNOSIS: If one model disagrees, is it seeing a "
    "real mesoscale feature (frontal boundary, sea breeze, orographic "
    "effect) or is it just wrong?\n"
    "4. FORECAST CONFIDENCE: How confident are you in the consensus "
    "temperature? Consider: time of year, city microclimate, "
    "synoptic pattern.\n"
    "5. RISKS: What specific physical mechanisms could bust this "
    "forecast? (fronts, inversions, lake effect, marine layer, "
    "convective initiation)\n\n"
    "CRITICAL RULES:\n"
    "- NEVER say \"the edge looks too large\" or \"massive edge "
    "suggests market error.\" You don't know market prices.\n"
    "- NEVER skip a trade because \"the market knows something.\" "
    "You are the weather expert.\n"
    "- Default to should_trade=true unless you identify a specific "
    "physical mechanism the models are missing or mishandling.\n"
    "- Only set should_trade=false if you have a SPECIFIC "
    "meteorological reason (not market skepticism).\n"
    "- confidence_adjustment reflects YOUR forecast confidence, "
    "not market confidence.\n"
    "- Include your estimated probability that the actual temperature "
    "falls in the target range "
    "(e.g., \"I estimate 65% chance of 25-26°C\").\n\n"
    "Respond in JSON:\n"
    "{\n"
    "  \"should_trade\": true/false,\n"
    "  \"confidence_adjustment\": 0.5-1.5,\n"
    "  \"rationale\": \"one sentence about the WEATHER, not the "
    "market\",\n"
    "  \"risk_factors\": [\"specific physical mechanism 1\", "
    "\"mechanism 2\"],\n"
    "  \"weather_insight\": \"what the atmosphere is doing\",\n"
    "  \"estimated_probability\": 0.0-1.0\n"
    "}"
)


def _coerce_bool(value: object) -> bool | None:
    """Strict boolean parse: True/False or "true"/"false" only, else None."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v == "true":
            return True
        if v == "false":
            return False
    return None


def _extract_json_object(text: str) -> dict | None:
    """Pull the first JSON object out of a model reply (tolerates code fences)."""
    cleaned = re.sub(r"```(?:json)?", "", text)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def parse_claude_response(signal: Signal, data: dict) -> TradeReasoning:
    """Turn a Messages API response into a TradeReasoning, failing closed.

    Anything short of a complete, well-formed verdict (truncated reply,
    refusal, no text block, bad JSON, non-boolean should_trade) yields
    ``should_trade=False, review_ok=False``.
    """
    def _failed(why: str, raw: str = "") -> TradeReasoning:
        logger.warning(
            "CLAUDE REVIEW FAILED (%s) for %s, failing closed: %s",
            why, signal.city_id, raw[:200],
        )
        return TradeReasoning(
            signal=signal,
            should_trade=False,
            confidence_adjustment=1.0,
            rationale=f"Review failed ({why})",
            risk_factors=[],
            weather_insight="",
            review_ok=False,
        )

    if not isinstance(data, dict):
        return _failed("non-object response")
    stop_reason = data.get("stop_reason")
    text = "".join(
        b.get("text", "")
        for b in (data.get("content") or [])
        if isinstance(b, dict) and b.get("type") == "text"
    )
    if stop_reason in ("max_tokens", "refusal"):
        return _failed(f"stop_reason={stop_reason}", text)
    if not text.strip():
        return _failed("no text content")

    result = _extract_json_object(text)
    if result is None:
        return _failed("unparseable JSON", text)

    should_trade = _coerce_bool(result.get("should_trade"))
    if should_trade is None:
        return _failed(f"invalid should_trade={result.get('should_trade')!r}", text)

    try:
        adj = float(result.get("confidence_adjustment", 1.0))
    except (TypeError, ValueError):
        adj = 1.0
    if not math.isfinite(adj):
        adj = 1.0
    adj = max(0.5, min(1.5, adj))

    risks = result.get("risk_factors", [])
    if not isinstance(risks, list):
        risks = [str(risks)] if risks else []

    return TradeReasoning(
        signal=signal,
        should_trade=should_trade,
        confidence_adjustment=adj,
        rationale=str(result.get("rationale", "") or ""),
        risk_factors=[str(r) for r in risks],
        weather_insight=str(result.get("weather_insight", "") or ""),
    )


async def analyze_trade(
    signal: Signal,
    model_values: dict[str, float],
    consensus_mean: float,
    consensus_std: float,
    variable: str = "temp_max_c",
) -> TradeReasoning | None:
    """Ask Claude to analyze a potential trade before execution.

    Returns None when no review could be obtained (no key, no model data, API
    failure); callers must treat None as "not approved" outside hail-mary
    mode. A reply that cannot be parsed returns ``should_trade=False`` with
    ``review_ok=False`` (fail closed).
    """
    from weather_edge.retry import redact, retry_async

    if not ANTHROPIC_API_KEY:
        logger.debug("No ANTHROPIC_API_KEY set, skipping Claude reasoning")
        return None
    if not model_values:
        logger.warning(
            "CLAUDE: no model forecasts for %s %s, cannot review",
            signal.city_id, signal.target_date,
        )
        return None

    model_summary = "\n".join(
        f"  {name}: {val:.1f}°C" for name, val in sorted(model_values.items())
    )
    desc = signal.description[:120] if signal.description else ""
    spread = max(model_values.values()) - min(model_values.values())

    user_prompt = f"""Assess this weather forecast:

CITY: {signal.city_id.upper()}
VARIABLE: {variable}
DATE: {signal.target_date or "unknown"}
QUESTION: {desc}

MODEL FORECASTS:
{model_summary}

CONSENSUS: mean={consensus_mean:.1f}°C, std={consensus_std:.1f}°C
MODEL SPREAD: {spread:.1f}°C range across {len(model_values)} models

Is this consensus trustworthy? What physical mechanisms could bust it?"""

    try:
        async def _call_claude():
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": ANTHROPIC_API_KEY,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": CLAUDE_MODEL,
                        "max_tokens": CLAUDE_MAX_TOKENS,
                        "system": SYSTEM_PROMPT,
                        "messages": [{"role": "user", "content": user_prompt}],
                    },
                    timeout=60.0,
                )
                resp.raise_for_status()
                return resp.json()

        data = await retry_async(
            _call_claude,
            attempts=3,
            base_delay=2.0,
            label=f"claude:{signal.city_id}",
        )

        reasoning = parse_claude_response(signal, data)

        logger.info(
            "CLAUDE: %s %s, %s (adj=%.1fx), %s",
            "TRADE" if reasoning.should_trade else "SKIP",
            signal.city_id.upper(),
            reasoning.rationale[:60],
            reasoning.confidence_adjustment,
            ", ".join(reasoning.risk_factors[:2]) or "no risks flagged",
        )

        try:
            from weather_edge.analysis.service_health import record_service_call
            existing = {}
            try:
                from weather_edge.live_state import get_json
                existing = get_json("svc:claude") or {}
            except Exception:
                logger.debug("Failed to read Claude service state", exc_info=True)
            decisions_today = existing.get("decisions_today", 0) + 1
            record_service_call(
                "claude", reasoning.review_ok,
                extra={"decisions_today": decisions_today},
            )
        except Exception:
            logger.debug("Failed to record Claude health", exc_info=True)

        return reasoning

    except Exception as e:
        logger.warning("Claude reasoning failed: %s", redact(e))
        try:
            from weather_edge.analysis.service_health import record_service_call
            record_service_call("claude", False)
        except Exception:
            logger.debug("Failed to record Claude failure", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# AI review memory
# ---------------------------------------------------------------------------
#
# A veto must outlive the cycle that produced it: sniper-triggered and
# cooldown-refresh cycles skip the (paid) AI step, and without memory a market
# Claude or Gemini rejected at 10:00 would be traded by the 10:03 sniper cycle.
#
# Vetoes are keyed by (market_id, target_date), any side, and last until the
# target date has passed. Approvals are keyed by (market_id, side, target_date),
# carry the size multiplier the AI layer applied, and expire after
# APPROVAL_TTL_SEC so a stale approval cannot carry a trade all day.

APPROVAL_TTL_SEC = 6 * 3600
_VETO_STATE_KEY = "ai_review:vetoes"
_VETO_STATE_TTL = 3 * 86400


@dataclass
class ReviewVerdict:
    approved: bool
    size_multiplier: float
    reason: str
    source: str
    recorded_at: float


def _side(signal: Signal) -> str:
    side = getattr(signal, "recommended_side", None)
    return str(getattr(side, "value", side))


class AIReviewMemory:
    """Per-day memory of AI approvals and vetoes (in-memory + live_state)."""

    def __init__(self, persist: bool = True):
        self._persist = persist
        self._loaded = not persist
        self._cutoff = ""  # target dates before this were evicted; never re-merge them
        self._vetoes: dict[tuple[str, str], ReviewVerdict] = {}
        self._approvals: dict[tuple[str, str, str], ReviewVerdict] = {}

    # Persistence covers vetoes only; approvals are re-earned after a restart.
    @staticmethod
    def _read_persisted() -> dict | None:
        """Persisted vetoes, {} if none, or None if the store could not be read.

        A failed read must not look like "no vetoes": that would let a vetoed
        market be approved again and let the next save overwrite the snapshot.
        """
        try:
            from weather_edge.live_state import get_value_strict
            raw = get_value_strict(_VETO_STATE_KEY)
        except Exception:
            logger.warning("AI review memory unavailable, will retry", exc_info=True)
            return None
        if not raw:
            return {}
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("AI review memory snapshot unreadable, ignoring it")
            return {}
        return data if isinstance(data, dict) else {}

    def _merge(self, raw: dict) -> None:
        for k, v in raw.items():
            market_id, _, target_date = k.partition("|")
            if self._cutoff and target_date and target_date < self._cutoff:
                continue
            try:
                verdict = ReviewVerdict(
                    approved=False,
                    size_multiplier=0.0,
                    reason=str(v.get("reason", "")),
                    source=str(v.get("source", "")),
                    recorded_at=float(v.get("recorded_at", 0.0)),
                )
            except (AttributeError, TypeError, ValueError):
                continue
            self._vetoes.setdefault((market_id, target_date), verdict)

    def _load(self) -> None:
        """Load persisted vetoes; stays unloaded (retried next call) on failure."""
        if self._loaded:
            return
        raw = self._read_persisted()
        if raw is None:
            return
        self._merge(raw)
        self._loaded = True

    def _save(self, merge: bool = True) -> None:
        """Write vetoes back, merging in any persisted ones first.

        If the store can't be read the save is skipped (vetoes stay in memory
        and are written on a later save) rather than overwriting a snapshot
        this process never saw.
        """
        if not self._persist:
            return
        if merge:
            raw = self._read_persisted()
            if raw is None:
                return
            self._merge(raw)
            self._loaded = True
        try:
            from weather_edge.live_state import set_json
            set_json(_VETO_STATE_KEY, {
                f"{m}|{d}": {
                    "reason": v.reason, "source": v.source, "recorded_at": v.recorded_at,
                }
                for (m, d), v in self._vetoes.items()
            }, ttl=_VETO_STATE_TTL)
        except Exception:
            logger.warning("AI review memory save failed", exc_info=True)

    def record_veto(self, signal: Signal, reason: str, source: str) -> None:
        self._load()
        key = (signal.market_id, str(signal.target_date))
        self._vetoes[key] = ReviewVerdict(False, 0.0, str(reason)[:200], source, time.time())
        for akey in [k for k in self._approvals if (k[0], k[2]) == key]:
            del self._approvals[akey]
        self._save()

    def record_approval(
        self, signal: Signal, size_multiplier: float, reason: str = "",
    ) -> None:
        self._load()
        if (signal.market_id, str(signal.target_date)) in self._vetoes:
            return  # a veto for the day wins over a later approval
        key = (signal.market_id, _side(signal), str(signal.target_date))
        self._approvals[key] = ReviewVerdict(
            True, float(size_multiplier), str(reason)[:200], "ai", time.time(),
        )

    def veto_for(self, signal: Signal) -> ReviewVerdict | None:
        self._load()
        return self._vetoes.get((signal.market_id, str(signal.target_date)))

    def approval_for(self, signal: Signal) -> ReviewVerdict | None:
        self._load()
        key = (signal.market_id, _side(signal), str(signal.target_date))
        verdict = self._approvals.get(key)
        if verdict and time.time() - verdict.recorded_at > APPROVAL_TTL_SEC:
            del self._approvals[key]
            return None
        return verdict

    def evict_before(self, today: date) -> int:
        """Drop entries whose target date is before ``today``."""
        self._load()
        cutoff = today.isoformat()
        self._cutoff = max(self._cutoff, cutoff)
        stale_v = [k for k in self._vetoes if k[1] and k[1] < cutoff]
        stale_a = [k for k in self._approvals if k[2] and k[2] < cutoff]
        for k in stale_v:
            del self._vetoes[k]
        for k in stale_a:
            del self._approvals[k]
        if stale_v:
            self._save()
        return len(stale_v) + len(stale_a)

    def clear(self) -> None:
        self._vetoes.clear()
        self._approvals.clear()
        self._save(merge=False)


_review_memory: AIReviewMemory | None = None


def get_review_memory() -> AIReviewMemory:
    """Process-wide review memory shared by every cycle type."""
    global _review_memory
    if _review_memory is None:
        _review_memory = AIReviewMemory()
    return _review_memory
