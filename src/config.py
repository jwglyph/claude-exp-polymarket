"""Configuration and settings for the Polymarket BTC arbitrage bot."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def _load_env() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)


_load_env()


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_float(key: str, default: float) -> float:
    val = os.getenv(key)
    return float(val) if val else default


@dataclass(frozen=True)
class PolymarketCredentials:
    api_key: str = field(default_factory=lambda: _env("POLYMARKET_API_KEY"))
    api_secret: str = field(default_factory=lambda: _env("POLYMARKET_API_SECRET"))
    api_passphrase: str = field(default_factory=lambda: _env("POLYMARKET_API_PASSPHRASE"))
    wallet_address: str = field(default_factory=lambda: _env("POLYMARKET_WALLET_ADDRESS"))
    private_key: str = field(default_factory=lambda: _env("POLYMARKET_PRIVATE_KEY"))

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.private_key)


@dataclass(frozen=True)
class TradingConfig:
    # Divergence threshold — trade when Polymarket lags by more than this (0.3%)
    divergence_threshold: float = field(
        default_factory=lambda: _env_float("DIVERGENCE_THRESHOLD", 0.003)
    )

    # Risk limits
    max_risk_per_trade: float = field(
        default_factory=lambda: _env_float("MAX_RISK_PER_TRADE", 0.005)
    )  # 0.5%
    daily_risk_cap: float = field(
        default_factory=lambda: _env_float("DAILY_RISK_CAP", 0.02)
    )  # 2%

    # Order sizing (USDC)
    order_size_usdc: float = field(
        default_factory=lambda: _env_float("ORDER_SIZE_USDC", 10.0)
    )

    # Polling intervals (seconds)
    price_poll_interval: float = 0.5
    polymarket_poll_interval: float = 1.0

    # Execution
    max_slippage: float = 0.002  # 0.2% max slippage
    order_timeout_seconds: float = 5.0


@dataclass(frozen=True)
class Settings:
    credentials: PolymarketCredentials = field(default_factory=PolymarketCredentials)
    trading: TradingConfig = field(default_factory=TradingConfig)

    # Polymarket CLOB API
    clob_api_url: str = "https://clob.polymarket.com"
    gamma_api_url: str = "https://gamma-api.polymarket.com"

    # Chain
    chain_id: int = 137  # Polygon mainnet


def load_settings() -> Settings:
    return Settings()
