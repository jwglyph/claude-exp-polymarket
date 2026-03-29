"""Live test: fetch real BTC prices and Polymarket data, check for divergence."""

import asyncio
import time
import json

import aiohttp


async def fetch_exchange_prices(session: aiohttp.ClientSession) -> dict[str, float]:
    """Fetch BTC price from multiple exchanges."""
    prices = {}

    async def binance():
        async with session.get(
            "https://api.binance.com/api/v3/ticker/price",
            params={"symbol": "BTCUSDT"},
        ) as resp:
            data = await resp.json()
            prices["binance"] = float(data["price"])

    async def coinbase():
        async with session.get(
            "https://api.coinbase.com/v2/prices/BTC-USD/spot",
        ) as resp:
            data = await resp.json()
            prices["coinbase"] = float(data["data"]["amount"])

    async def kraken():
        async with session.get(
            "https://api.kraken.com/0/public/Ticker",
            params={"pair": "XBTUSD"},
        ) as resp:
            data = await resp.json()
            prices["kraken"] = float(data["result"]["XXBTZUSD"]["c"][0])

    async def bybit():
        async with session.get(
            "https://api.bybit.com/v5/market/tickers",
            params={"category": "spot", "symbol": "BTCUSDT"},
        ) as resp:
            data = await resp.json()
            prices["bybit"] = float(data["result"]["list"][0]["lastPrice"])

    async def okx():
        async with session.get(
            "https://www.okx.com/api/v5/market/ticker",
            params={"instId": "BTC-USDT"},
        ) as resp:
            data = await resp.json()
            prices["okx"] = float(data["data"][0]["last"])

    tasks = [binance(), coinbase(), kraken(), bybit(), okx()]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            print(f"  [WARN] Exchange fetch {i} failed: {r}")
    return prices


async def fetch_polymarket_btc_markets(session: aiohttp.ClientSession) -> list[dict]:
    """Find BTC markets on Polymarket via Gamma API."""
    markets = []
    try:
        async with session.get(
            "https://gamma-api.polymarket.com/events",
            params={"tag": "crypto", "status": "active", "limit": 50},
        ) as resp:
            events = await resp.json()

        for event in events:
            title = (event.get("title") or "").lower()
            if "btc" not in title and "bitcoin" not in title:
                continue
            for m in event.get("markets", []):
                if not m.get("active"):
                    continue
                tokens = m.get("clobTokenIds", [])
                out_prices = m.get("outcomePrices", [])
                if len(tokens) >= 2 and len(out_prices) >= 2:
                    markets.append({
                        "question": m.get("question", ""),
                        "yes_price": float(out_prices[0]),
                        "no_price": float(out_prices[1]),
                        "token_yes": tokens[0],
                        "token_no": tokens[1],
                        "volume": m.get("volume", 0),
                        "conditionId": m.get("conditionId", ""),
                    })
    except Exception as e:
        print(f"  [ERROR] Gamma API: {e}")
    return markets


async def fetch_clob_midpoints(session: aiohttp.ClientSession, markets: list[dict]) -> None:
    """Fetch CLOB midpoints for comparison with Gamma prices."""
    for m in markets[:10]:  # Limit to avoid rate limits
        try:
            async with session.get(
                "https://clob.polymarket.com/midpoint",
                params={"token_id": m["token_yes"]},
            ) as resp:
                data = await resp.json()
                m["clob_mid_yes"] = float(data.get("mid", 0))
        except Exception:
            m["clob_mid_yes"] = None

        try:
            async with session.get(
                "https://clob.polymarket.com/book",
                params={"token_id": m["token_yes"]},
            ) as resp:
                book = await resp.json()
                bids = book.get("bids", [])
                asks = book.get("asks", [])
                m["best_bid"] = float(bids[0]["price"]) if bids else None
                m["best_ask"] = float(asks[0]["price"]) if asks else None
                m["spread"] = (m["best_ask"] - m["best_bid"]) if m["best_bid"] and m["best_ask"] else None
        except Exception:
            m["best_bid"] = m["best_ask"] = m["spread"] = None


async def main():
    print("=" * 70)
    print("LIVE EDGE CHECK: Exchange Prices vs Polymarket BTC Markets")
    print("=" * 70)

    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:

        # 1. Fetch exchange prices
        print("\n[1] Fetching real BTC prices from exchanges...")
        t0 = time.time()
        prices = await fetch_exchange_prices(session)
        fetch_ms = (time.time() - t0) * 1000
        print(f"    Fetched {len(prices)} prices in {fetch_ms:.0f}ms")

        for src, price in sorted(prices.items()):
            print(f"    {src:>10}: ${price:>12,.2f}")

        if len(prices) >= 2:
            from statistics import median
            med = median(prices.values())
            spread = max(prices.values()) - min(prices.values())
            print(f"    {'MEDIAN':>10}: ${med:>12,.2f}")
            print(f"    {'SPREAD':>10}: ${spread:>12,.2f} ({spread/med:.4%})")
        else:
            med = list(prices.values())[0] if prices else None
            print("    [WARN] Not enough exchange prices")

        # 2. Fetch Polymarket BTC markets
        print("\n[2] Fetching Polymarket BTC markets (Gamma API)...")
        t0 = time.time()
        markets = await fetch_polymarket_btc_markets(session)
        fetch_ms = (time.time() - t0) * 1000
        print(f"    Found {len(markets)} BTC markets in {fetch_ms:.0f}ms")

        if not markets:
            print("    [!] No BTC markets found. Cannot check edge.")
            return

        for m in markets[:15]:
            print(f"    {m['question'][:60]:<60} YES={m['yes_price']:.3f} NO={m['no_price']:.3f}")

        # 3. Fetch CLOB midpoints to compare with Gamma
        print("\n[3] Fetching CLOB order book data (live prices)...")
        t0 = time.time()
        await fetch_clob_midpoints(session, markets)
        fetch_ms = (time.time() - t0) * 1000
        print(f"    Fetched CLOB data in {fetch_ms:.0f}ms")

        # 4. Check for Gamma vs CLOB lag
        print("\n[4] Gamma API vs CLOB API price lag:")
        print(f"    {'Market':<45} {'Gamma':>8} {'CLOB':>8} {'Lag':>8} {'Spread':>8}")
        print("    " + "-" * 80)
        gamma_clob_lags = []
        for m in markets[:10]:
            clob_mid = m.get("clob_mid_yes")
            if clob_mid and clob_mid > 0:
                lag = abs(m["yes_price"] - clob_mid)
                gamma_clob_lags.append(lag)
                spread_str = f"{m['spread']:.4f}" if m.get("spread") else "N/A"
                print(
                    f"    {m['question'][:45]:<45} "
                    f"{m['yes_price']:>8.4f} {clob_mid:>8.4f} {lag:>8.4f} {spread_str:>8}"
                )

        # 5. Analyze the edge using our detector
        print("\n[5] Divergence analysis (real BTC price vs market fair values):")
        if med:
            import re
            price_pattern = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)\s*[kK]?")
            from math import exp, log

            def fair_value(real_price, threshold, vol=0.02, hours=24.0):
                if threshold <= 0:
                    return 1.0
                ratio = real_price / threshold
                sigma = vol * (hours ** 0.5)
                if sigma < 0.001:
                    return 1.0 if ratio > 1.0 else 0.0
                d = log(ratio) / sigma
                return max(0.01, min(0.99, 1.0 / (1.0 + exp(-1.7 * d))))

            print(f"\n    Real BTC (median): ${med:,.2f}")
            print(f"    {'Market':<40} {'Threshold':>10} {'YES':>6} {'Fair':>6} {'Edge':>8} {'Signal':<10}")
            print("    " + "-" * 85)

            edges_found = []
            for m in markets[:15]:
                match = price_pattern.search(m["question"])
                if not match:
                    continue
                raw = match.group(1).replace(",", "")
                threshold = float(raw)
                if threshold < 1000:
                    threshold *= 1000  # Handle "100k" style

                fv = fair_value(med, threshold)
                edge = fv - m["yes_price"]
                abs_edge = abs(edge)

                if edge > 0.003:
                    signal = "BUY YES"
                elif edge < -0.003:
                    signal = "BUY NO"
                else:
                    signal = "HOLD"

                edges_found.append((m["question"], threshold, m["yes_price"], fv, edge, signal))
                print(
                    f"    {m['question'][:40]:<40} "
                    f"${threshold:>9,.0f} "
                    f"{m['yes_price']:>6.3f} "
                    f"{fv:>6.3f} "
                    f"{edge:>+8.3f} "
                    f"{signal:<10}"
                )

            # Summary
            actionable = [e for e in edges_found if e[5] != "HOLD"]
            print(f"\n    Total markets analyzed: {len(edges_found)}")
            print(f"    Actionable signals (edge > 0.3%): {len(actionable)}")
            if actionable:
                max_edge = max(actionable, key=lambda x: abs(x[4]))
                print(f"    Biggest edge: {max_edge[4]:+.3f} on '{max_edge[0][:50]}'")
            else:
                print("    No actionable edges found at this moment.")

        # 6. Multi-sample timing test
        print("\n[6] Latency test (3 rapid samples)...")
        for i in range(3):
            t0 = time.time()
            p = await fetch_exchange_prices(session)
            ex_ms = (time.time() - t0) * 1000
            t0 = time.time()
            for m in markets[:3]:
                try:
                    async with session.get(
                        "https://clob.polymarket.com/midpoint",
                        params={"token_id": m["token_yes"]},
                    ) as resp:
                        await resp.json()
                except Exception:
                    pass
            pm_ms = (time.time() - t0) * 1000
            print(f"    Sample {i+1}: Exchanges={ex_ms:.0f}ms, Polymarket={pm_ms:.0f}ms")
            if i < 2:
                await asyncio.sleep(1)

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    if gamma_clob_lags:
        avg_lag = sum(gamma_clob_lags) / len(gamma_clob_lags)
        max_lag = max(gamma_clob_lags)
        print(f"  Gamma-CLOB price lag: avg={avg_lag:.4f}, max={max_lag:.4f}")
    print("  See signals above for current edge opportunities.")
    print("  NOTE: A real edge depends on timing, liquidity, and fees (~2% taker).")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
