from __future__ import annotations

import pandas as pd

from ..annualizer import Annualizer
from ..evaluation_metric import AggregateMetric


class AnnualizedReturn(AggregateMetric):
    """日次リターンを年率換算して返すクラス"""

    def __init__(self, trading_days_per_year: int = 252) -> None:
        super().__init__("年率リターン")
        self.annualizer = Annualizer(trading_days_per_year)

    def calculate(self, returns: pd.Series, **kwargs) -> None:
        mean_daily = returns.mean()
        self._value = self.annualizer.annualize_return(mean_daily)
