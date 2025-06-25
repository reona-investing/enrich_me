from __future__ import annotations

import pandas as pd

from ..evaluation_metric import AggregateMetric


class SharpeRatio(AggregateMetric):
    """シャープレシオを計算"""

    def __init__(self) -> None:
        super().__init__("シャープレシオ")

    def calculate(self, returns: pd.Series, **kwargs) -> None:
        mean = returns.mean()
        std = returns.std(ddof=0)
        if std == 0:
            self._value = float("nan")
        else:
            self._value = mean / std
