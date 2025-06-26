from __future__ import annotations

import pandas as pd

from ...annualizer import Annualizer
from ..evaluation_metric import AggregateMetric


class AnnualizedSharpeRatio(AggregateMetric):
    """シャープレシオを年率換算して返すクラス"""

    def __init__(self, trading_days_per_year: int = 252) -> None:
        super().__init__("年率シャープレシオ")
        self.annualizer = Annualizer(trading_days_per_year)

    def calculate(self, returns: pd.Series, **kwargs) -> None:
        mean_daily = returns.mean()
        std_daily = returns.std(ddof=0)
        if std_daily == 0:
            self._value = float("nan")
        else:
            ann_ret = self.annualizer.annualize_return(mean_daily)
            ann_std = self.annualizer.annualize_volatility(std_daily)
            self._value = ann_ret / ann_std
