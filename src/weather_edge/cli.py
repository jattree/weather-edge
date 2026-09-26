"""CLI entry point for weather-edge."""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import date, timedelta

import click
from rich.console import Console
from rich.logging import RichHandler

console = Console()


def _redacting(handler: logging.Handler) -> logging.Handler:
    """Keys in request URLs never reach the terminal (see retry.RedactingFilter)."""
    from weather_edge.retry import RedactingFilter
    handler.addFilter(RedactingFilter())
    return handler


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        # Plain tracebacks: Rich's renderer reads the live exception, which
        # would bypass the redacted text the filter prepares.
        handlers=[_redacting(RichHandler(rich_tracebacks=False, show_path=False))],
    )
    # Quiet noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)



@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")
def cli(verbose: bool) -> None:
    """Weather Edge: Polymarket weather trading system."""
    setup_logging(verbose)


DEFAULT_DAYS = 4  # today .. today+3, same window as the scheduler default
DAYS_HELP = (
    "Scan today .. today+N-1. Dates resolving inside the entry horizon "
    "(36h, 0h in hail-mary mode) are skipped, so N=2 scans nothing after 12:00 UTC."
)


def _print_scan_plan(target_dates: list[date]) -> list[date]:
    """Print which dates will be scanned and why others are not. Returns kept."""
    from weather_edge.analysis import claude_reasoning
    from weather_edge.config import settings
    from weather_edge.scheduler import filter_dates_by_horizon

    kept, blocked, horizon_h = filter_dates_by_horizon(target_dates)
    console.print(
        f"\n[bold cyan]Weather Edge[/], scanning {', '.join(map(str, kept)) or 'nothing'}"
    )
    if blocked:
        console.print(
            f"[dim]Skipping {', '.join(map(str, blocked))}: resolves in < {horizon_h}h "
            "(entry horizon filter)[/]"
        )
    if not kept:
        console.print(
            f"[yellow]Every requested date resolves inside the {horizon_h}h horizon, "
            f"nothing to scan. Use --days {DEFAULT_DAYS} (the default).[/]"
        )
    if getattr(settings, "require_ai_review", True) and not claude_reasoning.ANTHROPIC_API_KEY:
        console.print(
            "[yellow]No ANTHROPIC_API_KEY: signals are shown but not paper-traded "
            "(set REQUIRE_AI_REVIEW=false to paper-trade unreviewed signals).[/]"
        )
    console.print()
    return kept


@cli.command()
@click.option("--days", default=DEFAULT_DAYS, show_default=True, type=click.IntRange(1),
              help=DAYS_HELP)
def run(days: int) -> None:
    """Run one fetch → analyze → signal cycle (paper trading, in memory)."""
    from weather_edge.config import settings
    from weather_edge.reporting import print_paper_trades, print_signals
    from weather_edge.scheduler import default_target_dates, run_cycle
    from weather_edge.trading.paper import PaperTrader

    async def _run():
        trader = PaperTrader(bankroll=settings.bankroll)
        target_dates = default_target_dates(date.today(), days)
        if not _print_scan_plan(target_dates):
            return

        signals, _forecasts, _volume = await run_cycle(trader, target_dates)

        if signals:
            print_signals(signals)
        else:
            console.print("[yellow]No signals generated for the scanned dates.[/]")

        print_paper_trades(trader)

    asyncio.run(_run())


@cli.command()
@click.option("--days", default=DEFAULT_DAYS, show_default=True, type=click.IntRange(1),
              help=DAYS_HELP)
def watch(days: int) -> None:
    """Run continuously, fetching every FETCH_INTERVAL_MINUTES."""
    from weather_edge.config import settings
    from weather_edge.scheduler import default_target_dates, run_loop
    from weather_edge.trading.paper import PaperTrader

    trader = PaperTrader(bankroll=settings.bankroll)
    console.print("[bold cyan]Weather Edge[/], Continuous mode")
    _print_scan_plan(default_target_dates(date.today(), days))
    console.print(
        f"Fetching every {settings.fetch_interval_minutes} minutes. Press Ctrl+C to stop.\n"
    )

    try:
        asyncio.run(run_loop(trader, days=days))
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped.[/]")
        from weather_edge.reporting import print_paper_trades
        print_paper_trades(trader)


@cli.command()
def forecast() -> None:
    """Fetch and display forecasts for all cities (no trading logic)."""
    from weather_edge.config import CITIES
    from weather_edge.fetchers.openmeteo import fetch_city_forecasts
    from weather_edge.models.enums import City
    from weather_edge.reporting import print_forecasts

    async def _forecast():
        today = date.today()
        tomorrow = today + timedelta(days=1)
        all_forecasts = {}

        for city_id in City:
            city = CITIES[city_id]
            console.print(f"Fetching forecasts for [cyan]{city.name}[/] ({city.icao})...")
            forecasts = await fetch_city_forecasts(city_id, tomorrow)
            all_forecasts[city_id] = forecasts

        print_forecasts(all_forecasts)

    asyncio.run(_forecast())


@cli.command()
def markets() -> None:
    """List active Polymarket weather markets."""
    from rich.table import Table

    from weather_edge.fetchers.polymarket import discover_weather_markets

    async def _markets():
        mkts = await discover_weather_markets()

        table = Table(title="Active Weather Markets", show_header=True, header_style="bold magenta")
        table.add_column("City", style="cyan")
        table.add_column("Type")
        table.add_column("Threshold", justify="right")
        table.add_column("Date")
        table.add_column("Description", max_width=60)
        table.add_column("Token", style="dim", max_width=20)

        for m in mkts:
            table.add_row(
                m.city_id.value.upper() if m.city_id else "???",
                m.market_type.value,
                f"{m.threshold_value:.1f}°C",
                str(m.target_date),
                m.description[:60],
                (m.token_id_yes or "")[:20],
            )

        console.print(table)
        console.print(f"\n[green]{len(mkts)} markets found[/]")

    asyncio.run(_markets())


@cli.command()
def dashboard() -> None:
    """Launch the web dashboard."""
    try:
        import uvicorn
    except ImportError:
        console.print(
            "[red]Dashboard requires extra deps: pip install 'weather-edge[dashboard]'[/]"
        )
        sys.exit(1)

    # Bind to localhost by default, the dashboard has NO authentication, so it
    # must not be exposed on all interfaces. Override with WE_DASHBOARD_HOST only
    # if you have put your own auth/proxy in front of it.
    host = os.environ.get("WE_DASHBOARD_HOST", "127.0.0.1")
    console.print(f"[bold cyan]Weather Edge Dashboard[/], Starting on http://{host}:8000")
    uvicorn.run(
        "weather_edge.dashboard.app:app",
        host=host,
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    cli()
