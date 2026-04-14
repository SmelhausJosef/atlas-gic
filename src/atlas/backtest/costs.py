from __future__ import annotations


def bps_to_multiplier(bps: float, side: int) -> float:
    bump = bps / 10_000
    return 1 + bump if side > 0 else 1 - bump


def apply_execution_price(raw_price: float, side: int, total_bps: float) -> float:
    return raw_price * bps_to_multiplier(total_bps, side)

