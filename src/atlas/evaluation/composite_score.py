from __future__ import annotations


def composite_score(metrics: dict[str, float]) -> float:
    trade_penalty = 0.0
    if metrics["trade_count"] < 10:
        trade_penalty = (10 - metrics["trade_count"]) * 0.05

    return (
        metrics["net_return"] * 8.0
        + metrics["annualized_sharpe"] * 1.5
        + metrics["annualized_sortino"] * 1.0
        - metrics["max_drawdown"] * 6.0
        - metrics["turnover"] * 0.05
        - trade_penalty
    )

