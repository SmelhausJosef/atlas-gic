from __future__ import annotations

import math

import pandas as pd

from atlas.common.models import TradeRecord


def calculate_metrics(
    equity_curve: pd.DataFrame,
    trades: list[TradeRecord],
    periods_per_year: int,
    expected_bar_seconds: int,
) -> dict[str, float]:
    returns = equity_curve["equity"].pct_change().fillna(0.0)
    net_return = float(equity_curve["equity"].iloc[-1] / equity_curve["equity"].iloc[0] - 1)
    sharpe = _annualized_sharpe(returns, periods_per_year)
    sortino = _annualized_sortino(returns, periods_per_year)
    max_drawdown = _max_drawdown(equity_curve["equity"])
    win_rate = float(sum(1 for trade in trades if trade.pnl > 0) / len(trades)) if trades else 0.0
    gross_profit = sum(max(trade.pnl, 0.0) for trade in trades)
    gross_loss = abs(sum(min(trade.pnl, 0.0) for trade in trades))
    profit_factor = float(gross_profit / gross_loss) if gross_loss else float(gross_profit > 0)
    average_equity = float(equity_curve["equity"].mean()) if not equity_curve.empty else 0.0
    turnover = (
        float(sum(abs(trade.entry_price * trade.quantity) for trade in trades) / average_equity)
        if average_equity
        else 0.0
    )
    exposure = float(equity_curve["gross_exposure"].mean()) if "gross_exposure" in equity_curve else 0.0
    trade_count = float(len(trades))
    average_holding_seconds = (
        float(sum(trade.holding_bars for trade in trades) / len(trades) * expected_bar_seconds)
        if trades
        else 0.0
    )
    bars_in_market = float((equity_curve["position"] != 0).mean()) if "position" in equity_curve else 0.0

    return {
        "net_return": net_return,
        "annualized_sharpe": sharpe,
        "annualized_sortino": sortino,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "turnover": turnover,
        "exposure": exposure,
        "trade_count": trade_count,
        "average_holding_seconds": average_holding_seconds,
        "bars_in_market": bars_in_market,
    }


def _annualized_sharpe(returns: pd.Series, periods_per_year: int) -> float:
    std = float(returns.std(ddof=0))
    if std == 0:
        return 0.0
    return float((returns.mean() / std) * math.sqrt(periods_per_year))


def _annualized_sortino(returns: pd.Series, periods_per_year: int) -> float:
    downside = returns[returns < 0]
    std = float(downside.std(ddof=0)) if not downside.empty else 0.0
    if std == 0:
        return 0.0
    return float((returns.mean() / std) * math.sqrt(periods_per_year))


def _max_drawdown(equity: pd.Series) -> float:
    peaks = equity.cummax()
    drawdowns = equity / peaks - 1.0
    return float(abs(drawdowns.min()))

