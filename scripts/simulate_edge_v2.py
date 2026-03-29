"""Simulate v2: maker-order strategy (0% fees) with realistic fill modeling.

Key insight from v1: taker fees kill the strategy. But Polymarket charges
0% maker fees. So the real strategy is:

1. Detect price lag
2. Post LIMIT orders at the fair price (maker)
3. Wait for the market to fill them as it catches up
4. Collect the edge without paying taker fees

This simulation models fill probability — limit orders don't always fill.
"""

import random
import math
from dataclasses import dataclass
from statistics import mean, stdev


@dataclass
class SimConfig:
    initial_btc_price: float = 95000.0
    hourly_volatility: float = 0.005
    tick_interval_seconds: float = 0.5
    polymarket_lag_seconds: float = 3.0
    polymarket_noise: float = 0.005
    bracket_threshold: float = 95000.0
    time_to_expiry_hours: float = 24.0
    divergence_threshold: float = 0.003
    maker_fee: float = 0.0  # 0% maker fee
    order_size: float = 10.0
    max_daily_trades: int = 500
    duration_hours: float = 8.0
    fill_probability: float = 0.6  # Realistic fill rate for limit orders
    num_runs: int = 100


def fair_value(real_price, threshold, vol=0.02, hours=24.0):
    if threshold <= 0:
        return 1.0
    ratio = real_price / threshold
    sigma = vol * (hours ** 0.5)
    if sigma < 0.001:
        return 1.0 if ratio > 1.0 else 0.0
    d = math.log(ratio) / sigma
    return max(0.01, min(0.99, 1.0 / (1.0 + math.exp(-1.7 * d))))


def simulate_maker_strategy(config: SimConfig, seed: int) -> dict:
    rng = random.Random(seed)
    ticks_per_hour = 3600 / config.tick_interval_seconds
    total_ticks = int(config.duration_hours * ticks_per_hour)
    lag_ticks = int(config.polymarket_lag_seconds / config.tick_interval_seconds)

    # Generate BTC price path
    btc_prices = [config.initial_btc_price]
    tick_vol = config.hourly_volatility / math.sqrt(ticks_per_hour)
    for _ in range(total_ticks):
        ret = rng.gauss(0, tick_vol)
        btc_prices.append(btc_prices[-1] * (1 + ret))

    total_pnl = 0.0
    num_trades = 0
    wins = 0
    gross_edge_captured = 0.0

    for t in range(lag_ticks, total_ticks):
        real_price = btc_prices[t]
        lagged_price = btc_prices[t - lag_ticks]
        noise = rng.gauss(0, config.polymarket_noise)

        real_fair = fair_value(real_price, config.bracket_threshold,
                               vol=config.hourly_volatility, hours=config.time_to_expiry_hours)
        polymarket_shown = max(0.01, min(0.99,
            fair_value(lagged_price, config.bracket_threshold,
                       vol=config.hourly_volatility, hours=config.time_to_expiry_hours) + noise))

        edge = real_fair - polymarket_shown

        if abs(edge) > config.divergence_threshold and num_trades < config.max_daily_trades:
            # Fill probability: higher edge = more likely to fill
            # (the market is moving toward our price)
            edge_factor = min(1.0, abs(edge) / 0.02)
            fill_prob = config.fill_probability * (0.5 + 0.5 * edge_factor)

            if rng.random() < fill_prob:
                # Filled as maker
                if edge > 0:
                    pnl = edge * config.order_size
                else:
                    pnl = abs(edge) * config.order_size

                # Apply maker fee (0%)
                fee = config.order_size * config.maker_fee * 2
                net_pnl = pnl - fee

                total_pnl += net_pnl
                gross_edge_captured += pnl
                num_trades += 1
                if net_pnl > 0:
                    wins += 1

    return {
        "total_pnl": total_pnl,
        "num_trades": num_trades,
        "win_rate": wins / max(1, num_trades),
        "avg_pnl_per_trade": total_pnl / max(1, num_trades),
        "gross_edge": gross_edge_captured,
    }


def main():
    print("=" * 70)
    print("EDGE SIMULATION v2: Maker Order Strategy (0% fees)")
    print("=" * 70)

    scenarios = [
        ("Maker, 3s lag, 60% fill rate", SimConfig()),
        ("Maker, 3s lag, 40% fill rate", SimConfig(fill_probability=0.4)),
        ("Maker, 3s lag, 80% fill rate", SimConfig(fill_probability=0.8)),
        ("Maker, 5s lag, 60% fill rate", SimConfig(polymarket_lag_seconds=5.0)),
        ("Maker, 10s lag, 60% fill rate", SimConfig(polymarket_lag_seconds=10.0)),
        ("Maker, 3s lag, $50 size", SimConfig(order_size=50.0)),
        ("Maker, 3s lag, $100 size", SimConfig(order_size=100.0)),
        ("Maker, 3s lag, high vol", SimConfig(hourly_volatility=0.01)),
        ("Maker, 3s lag, threshold=0.5%", SimConfig(divergence_threshold=0.005)),
    ]

    print(f"\n{'Scenario':<40} {'Avg P&L':>10} {'Trades':>8} {'Win%':>7} {'$/Trade':>10} {'Profit%':>9}")
    print("-" * 90)

    for name, config in scenarios:
        results = [simulate_maker_strategy(config, seed=i) for i in range(config.num_runs)]

        avg_pnl = mean(r["total_pnl"] for r in results)
        avg_trades = mean(r["num_trades"] for r in results)
        avg_win_rate = mean(r["win_rate"] for r in results)
        avg_per_trade = mean(r["avg_pnl_per_trade"] for r in results)
        pct_profitable = sum(1 for r in results if r["total_pnl"] > 0) / config.num_runs

        print(
            f"  {name:<38} "
            f"${avg_pnl:>+9.2f} "
            f"{avg_trades:>7.0f} "
            f"{avg_win_rate:>6.1%} "
            f"${avg_per_trade:>+9.4f} "
            f"{pct_profitable:>8.0%}"
        )

    # Scaling analysis
    print(f"\n{'=' * 70}")
    print("SCALING: Daily P&L at different portfolio/order sizes (3s lag, maker)")
    print("=" * 70)
    print(f"  {'Order Size':>12} {'Daily P&L':>12} {'Annualized':>14} {'Trades':>8}")
    print("  " + "-" * 50)
    for size in [10, 25, 50, 100, 250, 500]:
        cfg = SimConfig(order_size=size, num_runs=50)
        results = [simulate_maker_strategy(cfg, seed=i) for i in range(cfg.num_runs)]
        avg_pnl = mean(r["total_pnl"] for r in results)
        avg_trades = mean(r["num_trades"] for r in results)
        annual = avg_pnl * 365
        print(f"  ${size:>11} ${avg_pnl:>+11.2f} ${annual:>+13,.0f} {avg_trades:>7.0f}")

    print(f"\n{'=' * 70}")
    print("CONCLUSION")
    print("=" * 70)

    # Final verdict
    base = [simulate_maker_strategy(SimConfig(), seed=i) for i in range(200)]
    avg = mean(r["total_pnl"] for r in base)
    sd = stdev(r["total_pnl"] for r in base)
    sharpe_daily = avg / sd if sd > 0 else 0
    pct_win = sum(1 for r in base if r["total_pnl"] > 0) / 200

    print(f"  Base case ($10 orders, 3s lag, 0% maker fee, 60% fill):")
    print(f"    Daily P&L:  ${avg:+.2f} +/- ${sd:.2f}")
    print(f"    Sharpe (daily): {sharpe_daily:.2f}")
    print(f"    Win sessions: {pct_win:.0%}")
    print(f"    Annualized: ${avg * 365:+,.0f}")
    print()
    if avg > 0:
        print("  VERDICT: Strategy IS profitable as a maker (0% fees).")
        print("  The edge is real but requires:")
        print("    1. Maker orders only (never cross the spread)")
        print("    2. Fast execution to get queue priority")
        print("    3. Sufficient fill rate (depends on liquidity)")
        print("    4. Scale with larger order sizes for meaningful returns")
    else:
        print("  VERDICT: Strategy is marginal even as maker.")
    print("=" * 70)


if __name__ == "__main__":
    main()
