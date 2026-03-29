# Polymarket BTC Price-Lag Arbitrage Bot

A Python bot that exploits the price lag between real BTC price feeds and Polymarket's BTC prediction market contracts.

## Strategy

Polymarket updates BTC contract prices slower than real exchange price feeds. This bot:

1. **Aggregates real BTC prices** from 5 exchanges (Binance, Coinbase, Kraken, Bybit, OKX) using the median for robustness
2. **Monitors Polymarket BTC bracket markets** (e.g., "Will BTC be above $95,000?")
3. **Computes fair values** using a log-normal probability model given the real BTC price
4. **Detects divergences** when Polymarket's implied probability lags the real price by >0.3%
5. **Executes trades** before the market catches up

## Architecture

```
src/
  config.py       - Settings, env loading, trading parameters
  price_feeds.py  - Multi-exchange BTC price aggregator
  polymarket.py   - Polymarket market discovery & price monitoring
  detector.py     - Divergence detection engine (core strategy)
  risk.py         - Risk manager (per-trade & daily limits)
  executor.py     - Trade execution via Polymarket CLOB API
  main.py         - Main loop with rich CLI dashboard
```

## Quick Start

```bash
# Install dependencies
pip install -e .

# Copy env template and add your credentials
cp .env.example .env

# Run in dry-run mode (paper trading)
python -m src.main

# Run with live trading (requires configured API keys)
python -m src.main --live --portfolio 1000
```

## Configuration

Set these in `.env` or pass via CLI:

| Variable | Default | Description |
|----------|---------|-------------|
| `DIVERGENCE_THRESHOLD` | `0.003` | Min price divergence to trigger trade (0.3%) |
| `MAX_RISK_PER_TRADE` | `0.005` | Max risk per trade as fraction of portfolio (0.5%) |
| `DAILY_RISK_CAP` | `0.02` | Max daily loss as fraction of portfolio (2%) |
| `ORDER_SIZE_USDC` | `10` | Max order size in USDC |

## Risk Management

- **Per-trade limit**: 0.5% of portfolio
- **Daily loss cap**: 2% of portfolio
- **Position sizing**: Half-Kelly criterion based on edge and confidence
- **Exposure cap**: Max 20% of portfolio in open positions
- **Dry-run default**: No real trades without explicit `--live` flag

## Tests

```bash
python -m pytest tests/ -v
```

## Disclaimer

This is an experimental project for educational purposes. Trading on prediction markets involves significant financial risk. Use at your own risk.
