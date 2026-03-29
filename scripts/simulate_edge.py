"""Simulate the edge with realistic BTC price movements and Polymarket lag.

Models:
- Real BTC price: random walk with realistic volatility
- Polymarket price: follows real price with configurable lag (seconds)
- Bot: detects divergence and trades when edge > threshold
- Accounts for 2% taker fee

Answers the question: is this strategy actually profitable?
"""

import random
import math
from dataclasses import dataclass, field
from statistics import mean


@dataclass
class SimConfig:
    # BTC price model
    initial_btc_price: float = 95000.0
    hourly_volatility: float = 0.005  # 0.5% per hour (realistic for BTC)
    tick_interval_seconds: float = 0.5

    # Polymarket lag model
    polymarket_lag_seconds: float = 3.0  # How many seconds Polymarket lags
    polymarket_noise: float = 0.005  # Random noise in Polymarket pricing

    # Market setup
    bracket_threshold: float = 95000.0  # "Will BTC be above $95k?"
    time_to_expiry_hours: float = 24.0

    # Trading
    divergence_threshold: float = 0.003  # 0.3%
    taker_fee: float = 0.02  # 2% taker fee
    order_size: float = 10.0  # $10 per trade
    max_daily_trades: int = 500

    # Simulation
    duration_hours: float = 8.0
    num_runs: int = 50


def fair_value(real_price: float, threshold: float, vol: float = 0.02, hours: float = 24.0) -> float:
    if threshold <= 0:
        return 1.0
    ratio = real_price / threshold
    sigma = vol * (hours ** 0.5)
    if sigma < 0.001:
        return 1.0 if ratio > 1.0 else 0.0
    d = math.log(ratio) / sigma
    return max(0.01, min(0.99, 1.0 / (1.0 + math.exp(-1.7 * d))))


def simulate_run(config: SimConfig, seed: int) -> dict:
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

    # Polymarket lagged fair values (what Polymarket shows)
    trades = []
    total_pnl = 0.0
    total_volume = 0.0
    num_trades = 0
    wins = 0
    losses = 0

    for t in range(lag_ticks, total_ticks):
        # Real BTC price right now
        real_price = btc_prices[t]

        # Polymarket sees the price from `lag_ticks` ago + noise
        lagged_price = btc_prices[t - lag_ticks]
        noise = rng.gauss(0, config.polymarket_noise)

        # Fair values
        real_fair = fair_value(real_price, config.bracket_threshold,
                               vol=config.hourly_volatility, hours=config.time_to_expiry_hours)
        polymarket_fair = fair_value(lagged_price, config.bracket_threshold,
                                     vol=config.hourly_volatility, hours=config.time_to_expiry_hours)
        polymarket_shown = max(0.01, min(0.99, polymarket_fair + noise))

        # Edge
        edge = real_fair - polymarket_shown

        if abs(edge) > config.divergence_threshold and num_trades < config.max_daily_trades:
            # We'd buy YES if edge > 0 (underpriced), buy NO if edge < 0 (overpriced)
            if edge > 0:
                # Buy YES at polymarket_shown, it should converge to real_fair
                entry_price = polymarket_shown
                exit_price = real_fair  # Price after market catches up
                gross_pnl = (exit_price - entry_price) * config.order_size / entry_price
            else:
                # Buy NO at (1 - polymarket_shown)
                entry_price = 1 - polymarket_shown
                exit_price = 1 - real_fair
                gross_pnl = (exit_price - entry_price) * config.order_size / entry_price

            # Apply taker fee on both entry and exit
            fee = config.order_size * config.taker_fee * 2  # Round trip
            net_pnl = gross_pnl - fee

            total_pnl += net_pnl
            total_volume += config.order_size
            num_trades += 1
            if net_pnl > 0:
                wins += 1
            else:
                losses += 1

            trades.append({
                "tick": t,
                "edge": edge,
                "entry": entry_price,
                "exit": exit_price,
                "gross_pnl": gross_pnl,
                "fee": fee,
                "net_pnl": net_pnl,
            })

    return {
        "total_pnl": total_pnl,
        "num_trades": num_trades,
        "total_volume": total_volume,
        "win_rate": wins / max(1, num_trades),
        "avg_pnl_per_trade": total_pnl / max(1, num_trades),
        "max_btc_price": max(btc_prices),
        "min_btc_price": min(btc_prices),
        "final_btc_price": btc_prices[-1],
    }


def main():
    print("=" * 70)
    print("EDGE SIMULATION: Is the Polymarket price-lag strategy profitable?")
    print("=" * 70)

    configs = [
        ("Scenario 1: 3s lag, 0.3% threshold, 2% fee", SimConfig()),
        ("Scenario 2: 5s lag, 0.3% threshold, 2% fee", SimConfig(polymarket_lag_seconds=5.0)),
        ("Scenario 3: 3s lag, 0.5% threshold, 2% fee", SimConfig(divergence_threshold=0.005)),
        ("Scenario 4: 3s lag, 0.3% threshold, 0% fee (maker)", SimConfig(taker_fee=0.0)),
        ("Scenario 5: 10s lag, 0.3% threshold, 2% fee", SimConfig(polymarket_lag_seconds=10.0)),
        ("Scenario 6: 3s lag, 1% threshold, 2% fee", SimConfig(divergence_threshold=0.01)),
        ("Scenario 7: 1s lag, 0.3% threshold, 2% fee", SimConfig(polymarket_lag_seconds=1.0)),
        ("Scenario 8: 3s lag, high vol (1%/hr), 2% fee", SimConfig(hourly_volatility=0.01)),
    ]

    for name, config in configs:
        results = [simulate_run(config, seed=i) for i in range(config.num_runs)]

        avg_pnl = mean(r["total_pnl"] for r in results)
        avg_trades = mean(r["num_trades"] for r in results)
        avg_win_rate = mean(r["win_rate"] for r in results)
        avg_pnl_per_trade = mean(r["avg_pnl_per_trade"] for r in results)
        profitable_runs = sum(1 for r in results if r["total_pnl"] > 0)

        print(f"\n{name}")
        print(f"  Lag={config.polymarket_lag_seconds}s | Threshold={config.divergence_threshold:.1%} | Fee={config.taker_fee:.0%} | Vol={config.hourly_volatility:.1%}/hr")
        print(f"  Avg P&L over {config.duration_hours}h: ${avg_pnl:+.2f}")
        print(f"  Avg trades: {avg_trades:.0f}")
        print(f"  Avg P&L/trade: ${avg_pnl_per_trade:+.4f}")
        print(f"  Win rate: {avg_win_rate:.1%}")
        print(f"  Profitable runs: {profitable_runs}/{config.num_runs} ({profitable_runs/config.num_runs:.0%})")

    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Run the critical comparison: with fee vs without fee
    no_fee = [simulate_run(SimConfig(taker_fee=0.0), seed=i) for i in range(100)]
    with_fee = [simulate_run(SimConfig(taker_fee=0.02), seed=i) for i in range(100)]

    no_fee_pnl = mean(r["total_pnl"] for r in no_fee)
    with_fee_pnl = mean(r["total_pnl"] for r in with_fee)
    fee_drag = mean(r["num_trades"] for r in with_fee) * 10.0 * 0.02 * 2  # avg fees paid

    print(f"  Gross edge (no fees):     ${no_fee_pnl:+.2f}/day")
    print(f"  Net edge (2% taker fee):  ${with_fee_pnl:+.2f}/day")
    print(f"  Avg fee drag:             ${fee_drag:.2f}/day")
    print(f"  Fee-to-edge ratio:        {fee_drag / max(0.01, no_fee_pnl):.1%}")

    # Break-even analysis
    print(f"\n  Break-even fee analysis:")
    for fee_pct in [0, 0.5, 1.0, 1.5, 2.0, 3.0]:
        cfg = SimConfig(taker_fee=fee_pct / 100)
        runs = [simulate_run(cfg, seed=i) for i in range(50)]
        avg = mean(r["total_pnl"] for r in runs)
        pct_profitable = sum(1 for r in runs if r["total_pnl"] > 0) / 50
        print(f"    Fee={fee_pct:.1f}%: avg P&L=${avg:+.2f}, profitable={pct_profitable:.0%}")

    print(f"\n  Conclusion:")
    if with_fee_pnl > 0:
        print(f"  Strategy IS profitable after fees: ${with_fee_pnl:+.2f}/day on $10 trades")
    else:
        print(f"  Strategy is NOT profitable with 2% taker fees.")
        print(f"  The gross edge exists (${no_fee_pnl:+.2f}/day) but fees eat it.")
        print(f"  To be profitable, you need:")
        print(f"    - Maker orders (0% fee) instead of taker")
        print(f"    - Larger lag (>5s) or higher BTC volatility")
        print(f"    - Tighter execution with limit orders")
    print("=" * 70)


if __name__ == "__main__":
    main()
