"""Configuration loaded from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")

BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET: str = os.getenv("BINANCE_SECRET", "")
SYMBOLS: list[str] = [
    s.strip().upper()
    for s in os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT").split(",")
    if s.strip()
]
TIMEFRAME: str = os.getenv("TIMEFRAME", "1h")
CAPITAL_USD: float = float(os.getenv("CAPITAL_USD", "10000"))
REGIME_MODEL: str = os.getenv("REGIME_MODEL", "xgboost").lower()
DATA_EXCHANGE: str = os.getenv("DATA_EXCHANGE", "binance").lower()
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LIVE_TRADING: bool = os.getenv("LIVE_TRADING", "false").lower() in ("true", "1", "yes")

# Testnet only when explicitly live trading with API keys
BINANCE_TESTNET: bool = LIVE_TRADING and bool(BINANCE_API_KEY)
CANDLE_BUFFER_SIZE: int = 500
SLIPPAGE_PCT: float = 0.001
FEE_PCT: float = 0.001
MAX_PORTFOLIO_HEAT: float = 0.20
KILL_SWITCH_DAILY_LOSS_PCT: float = 0.05
WALK_FORWARD_TRAIN_RATIO: float = 0.80

LOG_FILE: Path = _ROOT / "bot.log"
