"""Regret & Adherence Tracker, measures whether AI reasoning is adding value.

Correlates AI decisions (Claude approve/skip, Gemini agree/dissent) with
actual trade outcomes to answer:
- Did we win more when Claude approved vs when it didn't see the trade?
- Did Gemini's dissent correctly predict losers?
- What's the opportunity cost of following Gemini's dissent on winners?

Run after trade resolution to build the adherence report.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class AdherenceReport:
    """Summary of AI reasoning accuracy."""
    total_resolved: int = 0
    # Claude
    claude_approved_wins: int = 0
    claude_approved_losses: int = 0
    claude_skipped_would_have_won: int = 0
    claude_skipped_would_have_lost: int = 0
    # Gemini
    gemini_dissent_correct: int = 0  # Dissented and trade lost (or was reduced and lost less)
    gemini_dissent_wrong: int = 0  # Dissented but trade won (opportunity cost)
    gemini_agree_correct: int = 0  # Agreed and trade won
    gemini_agree_wrong: int = 0  # Agreed but trade lost
    # P&L attribution
    pnl_claude_approved: float = 0.0
    pnl_gemini_reduced: float = 0.0  # P&L on trades where Gemini halved the size
    opportunity_cost: float = 0.0  # Profit missed from Gemini dissent on winners

    @property
    def claude_accuracy(self) -> float:
        """Win rate on Claude-approved trades."""
        total = self.claude_approved_wins + self.claude_approved_losses
        return self.claude_approved_wins / total if total > 0 else 0.0

    @property
    def gemini_dissent_accuracy(self) -> float:
        """How often Gemini was right to dissent (trade would have lost)."""
        total = self.gemini_dissent_correct + self.gemini_dissent_wrong
        return self.gemini_dissent_correct / total if total > 0 else 0.0

    @property
    def ai_alpha(self) -> float:
        """Net value added by AI reasoning (P&L saved minus opportunity cost)."""
        return self.pnl_claude_approved - self.opportunity_cost

    def to_dict(self) -> dict:
        return {
            "total_resolved": self.total_resolved,
            "claude_accuracy": round(self.claude_accuracy * 100, 1),
            "claude_approved_wins": self.claude_approved_wins,
            "claude_approved_losses": self.claude_approved_losses,
            "gemini_dissent_accuracy": round(self.gemini_dissent_accuracy * 100, 1),
            "gemini_dissent_correct": self.gemini_dissent_correct,
            "gemini_dissent_wrong": self.gemini_dissent_wrong,
            "pnl_claude_approved": round(self.pnl_claude_approved, 2),
            "pnl_gemini_reduced": round(self.pnl_gemini_reduced, 2),
            "opportunity_cost": round(self.opportunity_cost, 2),
            "ai_alpha": round(self.ai_alpha, 2),
        }


def compute_adherence(
    resolved_trades: list,
    ai_decisions: list[dict],
) -> AdherenceReport:
    """Compute adherence report from resolved trades and AI decision history.

    Args:
        resolved_trades: List of PaperTrade objects with status won/lost
        ai_decisions: List of decision dicts from claude_reasoning._decision_history
    """
    report = AdherenceReport()

    # Build lookup: city -> list of AI decisions
    decisions_by_city: dict[str, list[dict]] = {}
    for d in ai_decisions:
        city = (d.get("city") or "").upper()
        if city:
            decisions_by_city.setdefault(city, []).append(d)

    for trade in resolved_trades:
        if trade.pnl is None:
            continue

        report.total_resolved += 1
        city = (trade.city_id or "").upper()
        won = trade.pnl > 0

        # Find matching AI decisions for this trade's city
        city_decisions = decisions_by_city.get(city, [])

        # Find Claude decision (source != "gemini" and != "exit_monitor")
        claude_decision = None
        gemini_decision = None
        for d in city_decisions:
            source = d.get("source", "")
            if source == "gemini":
                gemini_decision = d
            elif source == "exit_monitor":
                continue
            else:
                claude_decision = d

        # Claude tracking
        if claude_decision:
            if claude_decision.get("decision") == "TRADE":
                report.pnl_claude_approved += trade.pnl
                if won:
                    report.claude_approved_wins += 1
                else:
                    report.claude_approved_losses += 1
            elif claude_decision.get("decision") == "SKIP":
                if won:
                    report.claude_skipped_would_have_won += 1
                else:
                    report.claude_skipped_would_have_lost += 1

        # Gemini tracking
        if gemini_decision:
            is_dissent = gemini_decision.get("decision") == "DISSENT"
            if is_dissent:
                if not won:
                    report.gemini_dissent_correct += 1
                else:
                    report.gemini_dissent_wrong += 1
                    report.opportunity_cost += abs(trade.pnl) * 0.5  # We halved the size
            else:
                if won:
                    report.gemini_agree_correct += 1
                else:
                    report.gemini_agree_wrong += 1

        # Track P&L on Gemini-reduced trades
        if "[SPREAD]" not in (trade.description or "") and gemini_decision:
            if gemini_decision.get("decision") == "DISSENT":
                report.pnl_gemini_reduced += trade.pnl

    return report
