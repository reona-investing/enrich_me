from __future__ import annotations

import pandas as pd

from ...annualizer import Annualizer
from ..base.evaluation_metric import AggregateMetric


class AnnualizedStandardDeviation(AggregateMetric):
    """標準偏差を年率換算して返すクラス"""

    def __init__(self, trading_days_per_year: int = 252) -> None:
        super().__init__("年率標準偏差")
        self.annualizer = Annualizer(trading_days_per_year)

    def calculate(self, returns: pd.Series, **kwargs) -> None:
        std = returns.std(ddof=0)
        self._value = self.annualizer.annualize_volatility(std)
