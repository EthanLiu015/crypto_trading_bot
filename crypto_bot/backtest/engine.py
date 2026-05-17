"""Walk-forward backtester."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from crypto_bot import config
from crypto_bot.data import historical
from crypto_bot.features import engineer
from crypto_bot.regime.detector import RegimeDetector
from crypto_bot.strategy import ensemble

logger = logging.getLogger(__name__)

_BARS_PER_YEAR = {
    "1m": 525_600,
    "5m": 105_120,
    "15m": 35_040,
    "1h": 8_760,
    "4h": 2_190,
    "1d": 365,
}


def _trade_record(
    entry_time: str,
    exit_time: str,
    side: float,
    entry_price: float,
    exit_price: float,
    pnl_usd: float,
    equity_before: float,
    equity_after: float,
    note: str = "",
) -> dict:
    side_label = "long" if side > 0 else "short"
    pnl_pct = (pnl_usd / equity_before * 100) if equity_before > 0 else 0.0
    return {
        "entry_time": entry_time,
        "exit_time": exit_time,
        "side": side_label,
        "entry_price": round(entry_price, 2),
        "exit_price": round(exit_price, 2),
        "pnl_usd": round(pnl_usd, 2),
        "pnl_pct": round(pnl_pct, 2),
        "equity_after": round(equity_after, 2),
        "note": note,
    }


def _simulate_bar(
    equity: float,
    position: float,
    signal: float,
    open_price: float,
    next_open: float,
    slippage: float,
    fee: float,
) -> tuple[float, float, float | None]:
    """Returns (new_equity, new_position, trade_pnl or None)."""
    trade_pnl = None
    target_pos = float(np.clip(signal, -1, 1))
    exposure = 0.1

    if position != target_pos and target_pos != 0:
        trade_value = equity * abs(target_pos - position) * exposure
        equity -= trade_value * fee
        position = target_pos
    elif position != 0 and target_pos == 0:
        cost = equity * abs(position) * exposure * fee
        equity -= cost
        trade_pnl = position * (next_open - open_price) / open_price * equity * exposure
        equity += trade_pnl
        position = 0.0
    elif position != 0:
        ret = (next_open - open_price) / open_price
        equity += position * ret * equity * exposure

    return equity, position, trade_pnl


def run_backtest(
    symbol: str,
    timeframe: str = "1h",
    start_date: str | None = None,
    end_date: str | None = None,
    lookback_days: int = 365,
) -> dict:
    """
    Walk-forward backtest: 80% train for regime fit, 20% test for evaluation.
    No look-ahead on signals.
    """
    df = historical.fetch_ohlcv(symbol, timeframe, lookback_days)
    if df.empty:
        return _empty_result(symbol, timeframe)

    if "timestamp" in df.columns:
        df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        if start_date:
            df = df[df["dt"] >= pd.Timestamp(start_date, tz="UTC")]
        if end_date:
            df = df[df["dt"] <= pd.Timestamp(end_date, tz="UTC")]
        df = df.reset_index(drop=True)

    if len(df) < 150:
        logger.warning("Insufficient data for backtest: %d bars", len(df))
        return _empty_result(symbol, timeframe)

    featured = engineer.compute_features(df)
    split_idx = int(len(featured) * config.WALK_FORWARD_TRAIN_RATIO)
    train_df = featured.iloc[:split_idx].copy()
    test_df = featured.iloc[split_idx:].copy()

    if len(test_df) < 50:
        return _empty_result(symbol, timeframe)

    detector = RegimeDetector()
    detector.fit(train_df)

    slippage = config.SLIPPAGE_PCT
    fee = config.FEE_PCT
    equity = config.CAPITAL_USD
    equity_curve = [equity]
    equity_timestamps: list[int] = []
    position = 0.0
    trades: list[float] = []
    trade_log: list[dict] = []
    entry_price: float | None = None
    entry_side: float | None = None
    entry_time: str | None = None
    exposure = 0.1
    warmup = min(50, len(test_df) // 4)

    def _ts_ms(global_bar: int) -> int:
        raw = int(featured.iloc[global_bar]["timestamp"])
        return raw if raw > 1_000_000_000_000 else raw * 1000

    def _ts_seconds(global_bar: int) -> int:
        return _ts_ms(global_bar) // 1000

    equity_timestamps.append(_ts_seconds(split_idx + warmup))

    for i in range(warmup, len(test_df) - 1):
        global_idx = split_idx + i
        window = featured.iloc[: global_idx + 1]
        regime_probs = detector.latest_regime_dict(window)
        sig = ensemble.generate_signal(window, regime_probs)

        row = test_df.iloc[i]
        next_row = test_df.iloc[i + 1]
        bar_open = float(row["open"])
        fill_price = float(next_row["open"])
        exit_iso = pd.to_datetime(_ts_ms(split_idx + i + 1), unit="ms", utc=True).isoformat()
        prev_position = position
        equity_before = equity

        equity, position, trade_pnl = _simulate_bar(
            equity,
            position,
            sig,
            bar_open,
            fill_price,
            slippage,
            fee,
        )

        if trade_pnl is not None:
            trades.append(trade_pnl)
            if entry_price is not None and entry_side is not None and entry_time is not None:
                trade_log.append(
                    _trade_record(
                        entry_time=entry_time,
                        exit_time=exit_iso,
                        side=entry_side,
                        entry_price=entry_price,
                        exit_price=fill_price,
                        pnl_usd=trade_pnl,
                        equity_before=equity_before,
                        equity_after=equity,
                    )
                )
            entry_price = None
            entry_side = None
            entry_time = None

        if position != 0 and prev_position == 0:
            entry_price = fill_price
            entry_side = position
            entry_time = exit_iso

        equity_curve.append(equity)
        equity_timestamps.append(_ts_seconds(split_idx + i + 1))

    test_start_idx = split_idx + warmup
    test_end_idx = split_idx + len(test_df) - 2

    returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
    returns = returns[np.isfinite(returns)]
    sharpe = 0.0
    bars_per_year = _BARS_PER_YEAR.get(timeframe, 8760)
    if len(returns) > 1 and returns.std() > 0:
        sharpe = float(returns.mean() / returns.std() * np.sqrt(bars_per_year))

    eq = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / np.where(peak > 0, peak, 1)
    max_dd = float(dd.min()) if len(dd) else 0.0

    wins = [t for t in trades if t > 0]
    win_rate = len(wins) / len(trades) if trades else 0.0
    total_return = (equity - config.CAPITAL_USD) / config.CAPITAL_USD

    test_start = pd.to_datetime(_ts_ms(test_start_idx), unit="ms", utc=True)
    test_end = pd.to_datetime(_ts_ms(test_end_idx), unit="ms", utc=True)
    data_start = pd.to_datetime(_ts_ms(0), unit="ms", utc=True)
    data_end = pd.to_datetime(_ts_ms(len(featured) - 1), unit="ms", utc=True)
    data_days = (data_end - data_start).total_seconds() / 86400

    result = {
        "symbol": symbol,
        "timeframe": timeframe,
        "lookback_days": lookback_days,
        "data_bars": len(featured),
        "data_days": round(data_days, 1),
        "data_start": data_start.isoformat(),
        "data_end": data_end.isoformat(),
        "train_ratio": config.WALK_FORWARD_TRAIN_RATIO,
        "test_bars": len(equity_curve),
        "total_return": float(total_return),
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "win_rate": float(win_rate),
        "num_trades": len(trades),
        "trades": trade_log,
        "equity_curve": equity_curve,
        "equity_timestamps": equity_timestamps,
        "test_start": test_start.isoformat(),
        "test_end": test_end.isoformat(),
        "final_equity": float(equity),
    }
    logger.info(
        "Backtest %s: return=%.2f%% sharpe=%.2f max_dd=%.2f%% trades=%d",
        symbol,
        total_return * 100,
        sharpe,
        max_dd * 100,
        len(trades),
    )
    return result


def _empty_result(symbol: str = "", timeframe: str = "") -> dict:
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "total_return": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0,
        "num_trades": 0,
        "trades": [],
        "lookback_days": 0,
        "data_bars": 0,
        "data_days": 0,
        "data_start": "",
        "data_end": "",
        "train_ratio": config.WALK_FORWARD_TRAIN_RATIO,
        "test_bars": 0,
        "equity_curve": [config.CAPITAL_USD],
        "equity_timestamps": [],
        "test_start": "",
        "test_end": "",
        "final_equity": config.CAPITAL_USD,
    }
