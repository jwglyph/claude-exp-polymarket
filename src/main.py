"""Main entry point — runs the arbitrage bot with a live dashboard.

Architecture:
  1. Price feed aggregator polls BTC price from 5 exchanges every 0.5s
  2. Polymarket monitor tracks BTC bracket markets every 1s
  3. Divergence detector compares prices and finds opportunities
  4. Executor places trades (dry-run by default)
  5. Risk manager enforces limits throughout
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
import time

import aiohttp
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text

from .config import load_settings, Settings, TradingConfig
from .price_feeds import PriceFeedAggregator
from .polymarket import PolymarketMonitor
from .detector import DivergenceDetector, Signal
from .executor import TradeExecutor
from .risk import RiskManager

console = Console()


def build_dashboard(
    btc_price: float | None,
    price_sources: dict,
    brackets: list,
    opportunities: list,
    risk_stats: dict,
    is_dry_run: bool,
    cycle_time_ms: float,
    cycle_count: int,
) -> Layout:
    """Build the rich dashboard layout."""
    layout = Layout()

    # ── Header ──
    mode = "[yellow]DRY RUN[/yellow]" if is_dry_run else "[red bold]LIVE[/red bold]"
    header = Text.from_markup(
        f" POLYMARKET BTC ARB BOT  |  {mode}  |  "
        f"Cycle #{cycle_count}  |  {cycle_time_ms:.0f}ms  |  "
        f"BTC: ${btc_price:,.2f}" if btc_price else " POLYMARKET BTC ARB BOT  |  Waiting for data..."
    )

    # ── Price feeds table ──
    price_table = Table(title="Price Feeds", expand=True)
    price_table.add_column("Source", style="cyan")
    price_table.add_column("Price", justify="right", style="green")
    price_table.add_column("Age", justify="right", style="dim")

    for src, point in sorted(price_sources.items()):
        price_table.add_row(
            src,
            f"${point.price:,.2f}",
            f"{point.age_ms:.0f}ms",
        )
    if btc_price:
        price_table.add_row(
            "[bold]MEDIAN[/bold]",
            f"[bold]${btc_price:,.2f}[/bold]",
            "",
            style="bold green",
        )

    # ── Markets table ──
    market_table = Table(title="BTC Brackets", expand=True)
    market_table.add_column("Threshold", justify="right", style="cyan")
    market_table.add_column("YES", justify="right")
    market_table.add_column("NO", justify="right")
    market_table.add_column("Fair", justify="right", style="yellow")
    market_table.add_column("Edge", justify="right")
    market_table.add_column("Signal")

    for opp in opportunities[:10]:
        edge_color = "green" if opp.edge > 0 else "red"
        signal_style = {
            Signal.BUY_YES: "[bold green]BUY YES[/bold green]",
            Signal.BUY_NO: "[bold red]BUY NO[/bold red]",
            Signal.HOLD: "[dim]HOLD[/dim]",
        }
        market_table.add_row(
            f"${opp.bracket.threshold_price:,.0f}",
            f"{opp.market_price:.3f}",
            f"{opp.bracket.no_price:.3f}",
            f"{opp.fair_value:.3f}",
            f"[{edge_color}]{opp.edge:+.3f}[/{edge_color}]",
            signal_style[opp.signal],
        )

    # ── Risk panel ──
    risk_text = (
        f"Daily P&L: ${risk_stats.get('daily_pnl', 0):.2f}  |  "
        f"Trades: {risk_stats.get('daily_trades', 0)}  |  "
        f"Volume: ${risk_stats.get('daily_volume', 0):.2f}  |  "
        f"Exposure: ${risk_stats.get('open_exposure', 0):.2f}  |  "
        f"Risk: {risk_stats.get('risk_utilization', 0):.1%}"
    )

    # ── Compose layout ──
    layout.split_column(
        Layout(Panel(header, style="bold blue"), size=3),
        Layout(name="body"),
        Layout(Panel(Text.from_markup(risk_text), title="Risk"), size=3),
    )
    layout["body"].split_row(
        Layout(price_table, name="prices"),
        Layout(market_table, name="markets", ratio=2),
    )

    return layout


async def run_bot(settings: Settings, dry_run: bool = True, portfolio: float = 1000.0) -> None:
    """Main bot loop."""
    async with aiohttp.ClientSession() as session:
        feeds = PriceFeedAggregator(session)
        monitor = PolymarketMonitor(settings, session)
        detector = DivergenceDetector(settings.trading.divergence_threshold)
        risk_mgr = RiskManager(settings.trading, portfolio)
        executor = TradeExecutor(settings, risk_mgr, session, dry_run=dry_run)

        console.print("[bold blue]Starting Polymarket BTC Arbitrage Bot...[/bold blue]")

        # Check credentials before doing anything
        if not dry_run:
            if not settings.credentials.is_configured:
                console.print("[red]ERROR: API credentials not configured. Set .env file.[/red]")
                console.print("[red]Need: POLYMARKET_API_KEY, API_SECRET, API_PASSPHRASE, PRIVATE_KEY[/red]")
                return
            console.print("[red bold]LIVE TRADING MODE — Real money at risk![/red bold]")
        else:
            console.print("[yellow]Running in DRY RUN mode (no real trades)[/yellow]")

        # Initial market discovery
        console.print("[dim]Discovering BTC markets on Polymarket...[/dim]")
        markets = await monitor.discover_btc_markets()
        if not markets:
            console.print("[yellow]WARNING: No BTC markets found. Bot will retry periodically.[/yellow]")
        else:
            console.print(f"[green]Found {len(markets)} BTC markets[/green]")
        brackets = monitor.parse_btc_brackets()
        if markets and not brackets:
            console.print("[yellow]WARNING: Markets found but no price brackets parsed.[/yellow]")
        else:
            console.print(f"[green]Parsed {len(brackets)} price brackets[/green]")

        cycle_count = 0
        btc_price: float | None = None
        opportunities: list = []
        no_price_warned = False

        with Live(console=console, refresh_per_second=2) as live:
            while True:
                cycle_start = time.time()
                cycle_count += 1

                try:
                    # 1. Fetch real BTC prices
                    await feeds.fetch_all()
                    btc_price = feeds.get_median_price()

                    if btc_price is None and not no_price_warned:
                        console.print("[yellow]WARNING: No BTC price data from exchanges yet.[/yellow]")
                        no_price_warned = True
                    elif btc_price is not None:
                        no_price_warned = False

                    # 2. Refresh Polymarket prices (every 5 cycles)
                    if cycle_count % 5 == 0:
                        await monitor.refresh_prices()
                        brackets = monitor.parse_btc_brackets()

                    # 3. Re-discover markets periodically (every 60 cycles)
                    if cycle_count % 60 == 0:
                        await monitor.discover_btc_markets()
                        brackets = monitor.parse_btc_brackets()

                    # 4. Detect divergences
                    if btc_price and brackets:
                        opportunities = detector.scan_brackets(brackets, btc_price)
                        actionable = detector.get_actionable(opportunities)

                        # 5. Execute trades
                        for opp in actionable:
                            await executor.execute(opp)

                    # 6. Update dashboard
                    cycle_ms = (time.time() - cycle_start) * 1000
                    dashboard = build_dashboard(
                        btc_price=btc_price,
                        price_sources=feeds._latest,
                        brackets=brackets,
                        opportunities=opportunities,
                        risk_stats=risk_mgr.stats,
                        is_dry_run=dry_run,
                        cycle_time_ms=cycle_ms,
                        cycle_count=cycle_count,
                    )
                    live.update(dashboard)

                except Exception as e:
                    import traceback
                    console.print(f"[red]Error in cycle {cycle_count}: {e}[/red]")
                    console.print(f"[dim]{traceback.format_exc()}[/dim]")

                # Sleep for next cycle
                elapsed = time.time() - cycle_start
                sleep_time = max(0, settings.trading.price_poll_interval - elapsed)
                await asyncio.sleep(sleep_time)


def main() -> None:
    parser = argparse.ArgumentParser(description="Polymarket BTC Arbitrage Bot")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live trading (default: dry run)",
    )
    parser.add_argument(
        "--portfolio",
        type=float,
        default=1000.0,
        help="Portfolio value in USDC (default: 1000)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override divergence threshold (e.g., 0.003 for 0.3%%)",
    )
    args = parser.parse_args()

    settings = load_settings()
    if args.threshold is not None:
        # Override threshold via CLI
        settings = Settings(
            credentials=settings.credentials,
            trading=TradingConfig(
                divergence_threshold=args.threshold,
                max_risk_per_trade=settings.trading.max_risk_per_trade,
                daily_risk_cap=settings.trading.daily_risk_cap,
                order_size_usdc=settings.trading.order_size_usdc,
            ),
        )

    # Handle Ctrl+C gracefully
    def handle_sigint(sig, frame):
        console.print("\n[yellow]Shutting down...[/yellow]")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    asyncio.run(run_bot(settings, dry_run=not args.live, portfolio=args.portfolio))


if __name__ == "__main__":
    main()
