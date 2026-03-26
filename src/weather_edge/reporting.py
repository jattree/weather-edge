"""CLI reporting with Rich tables."""
from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from weather_edge.analysis.edge import Signal
from weather_edge.config import CITIES
from weather_edge.fetchers.openmeteo import ForecastResult, c_to_f
from weather_edge.models.enums import City, SignalTier
from weather_edge.trading.paper import PaperTrader

console = Console()


def print_forecasts(all_forecasts: dict[City, list[ForecastResult]]) -> None:
    """Print forecast summary table."""
    table = Table(title="Weather Forecasts", show_header=True, header_style="bold cyan")
    table.add_column("City", style="white")
    table.add_column("ICAO", style="dim")
    table.add_column("Models", justify="right")
    table.add_column("High °C", justify="right", style="red")
    table.add_column("High °F", justify="right", style="red")
    table.add_column("Low °C", justify="right", style="blue")
    table.add_column("Precip mm", justify="right", style="cyan")
    table.add_column("Snow cm", justify="right", style="white bold")

    for city_id, forecasts in all_forecasts.items():
        city = CITIES[city_id]
        if not forecasts:
            table.add_row(city.name, city.icao, "0", "-", "-", "-", "-", "-")
            continue

        # Average across models
        highs = [f.temp_max_c for f in forecasts if f.temp_max_c is not None]
        lows = [f.temp_min_c for f in forecasts if f.temp_min_c is not None]
        precips = [f.precip_sum_mm for f in forecasts if f.precip_sum_mm is not None]
        snows = [f.snow_sum_cm for f in forecasts if f.snow_sum_cm is not None]

        avg_high = sum(highs) / len(highs) if highs else None
        avg_low = sum(lows) / len(lows) if lows else None
        avg_precip = sum(precips) / len(precips) if precips else None
        avg_snow = sum(snows) / len(snows) if snows else None

        table.add_row(
            city.name,
            city.icao,
            str(len(forecasts)),
            f"{avg_high:.1f}" if avg_high is not None else "-",
            f"{c_to_f(avg_high):.0f}" if avg_high is not None else "-",
            f"{avg_low:.1f}" if avg_low is not None else "-",
            f"{avg_precip:.1f}" if avg_precip is not None else "-",
            f"{avg_snow:.1f}" if avg_snow is not None else "-",
        )

    console.print(table)


def print_signals(signals: list[Signal]) -> None:
    """Print signals table, sorted by edge magnitude."""
    signals_sorted = sorted(signals, key=lambda s: abs(s.edge), reverse=True)

    table = Table(title="Trading Signals", show_header=True, header_style="bold green")
    table.add_column("City", style="white")
    table.add_column("Side", justify="center")
    table.add_column("Model P", justify="right")
    table.add_column("Market P", justify="right")
    table.add_column("Edge", justify="right")
    table.add_column("Edge %", justify="right")
    table.add_column("Confidence", justify="right")
    table.add_column("Kelly", justify="right")
    table.add_column("Size $", justify="right")
    table.add_column("Tier", justify="center")

    for s in signals_sorted:
        # Color coding for tier
        if s.confidence_tier == SignalTier.HIGH:
            tier_style = "[bold green]HIGH[/]"
            row_style = "green"
        elif s.confidence_tier == SignalTier.MEDIUM:
            tier_style = "[bold yellow]MED[/]"
            row_style = "yellow"
        else:
            tier_style = "[dim]LOW[/]"
            row_style = "dim"

        side_str = f"[green]{s.recommended_side.value}[/]" if s.recommended_side.value == "YES" else f"[red]{s.recommended_side.value}[/]"

        table.add_row(
            s.city_id.upper(),
            side_str,
            f"{s.model_prob:.3f}",
            f"{s.market_prob:.3f}",
            f"{s.edge:+.3f}",
            f"{s.edge_pct:+.1%}",
            f"{s.model_confidence:.0%}",
            f"{s.half_kelly:.3f}",
            f"${s.recommended_size:.0f}",
            tier_style,
            style=row_style,
        )

    console.print(table)


def print_paper_trades(trader: PaperTrader) -> None:
    """Print paper trading summary."""
    summary = trader.summary()

    # Summary panel
    pnl_color = "green" if summary["total_pnl"] >= 0 else "red"
    summary_text = (
        f"Total Trades: {summary['total_trades']}  |  "
        f"Open: {summary['open']}  |  "
        f"Won: {summary['wins']}  |  "
        f"Lost: {summary['losses']}  |  "
        f"Win Rate: {summary['win_rate']}%  |  "
        f"P&L: [{pnl_color}]${summary['total_pnl']:+.2f}[/]"
    )
    console.print(Panel(summary_text, title="Paper Trading", border_style="blue"))

    # Recent trades table
    recent = sorted(trader.trades, key=lambda t: t.placed_at, reverse=True)[:20]
    if not recent:
        return

    table = Table(title="Recent Paper Trades", show_header=True, header_style="bold blue")
    table.add_column("Time", style="dim")
    table.add_column("City")
    table.add_column("Side", justify="center")
    table.add_column("Size", justify="right")
    table.add_column("Entry", justify="right")
    table.add_column("Status", justify="center")
    table.add_column("P&L", justify="right")
    table.add_column("Description", max_width=40)

    for t in recent:
        side_str = f"[green]YES[/]" if t.side == "YES" else f"[red]NO[/]"
        status_str = {
            "open": "[yellow]OPEN[/]",
            "won": "[green]WON[/]",
            "lost": "[red]LOST[/]",
        }.get(t.status.value, t.status.value)
        pnl_str = f"${t.pnl:+.2f}" if t.pnl is not None else "-"
        pnl_color = "green" if t.pnl and t.pnl > 0 else "red" if t.pnl and t.pnl < 0 else ""

        table.add_row(
            t.placed_at.strftime("%H:%M:%S"),
            t.city_id.upper(),
            side_str,
            f"${t.size_usd:.0f}",
            f"{t.entry_price:.3f}",
            status_str,
            f"[{pnl_color}]{pnl_str}[/]" if pnl_color else pnl_str,
            (t.description or "")[:40],
        )

    console.print(table)
