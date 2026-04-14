from __future__ import annotations

from collections.abc import Callable

from atlas.strategies.base import BaseStrategy
from atlas.strategies.btc_mean_reversion_tight_v1 import BTCMeanReversionTightV1
from atlas.strategies.btc_mean_reversion_v1 import BTCMeanReversionV1


_REGISTRY: dict[str, type[BaseStrategy]] = {
    "btc_mean_reversion_v1": BTCMeanReversionV1,
    "btc_mean_reversion_tight_v1": BTCMeanReversionTightV1,
}


def get_strategy_class(name: str) -> type[BaseStrategy]:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown strategy: {name}") from exc


def register_strategy(name: str, strategy_cls: type[BaseStrategy]) -> None:
    _REGISTRY[name] = strategy_cls

