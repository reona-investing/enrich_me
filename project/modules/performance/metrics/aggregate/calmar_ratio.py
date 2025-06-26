from __future__ import annotations

import pandas as pd

from ...annualizer import Annualizer
from ..base.evaluation_metric import AggregateMetric


class CalmarRatio(AggregateMetric):
    """カルマーレシオを計算するクラス"""

    def __init__(self, trading_days_per_year: int = 252) -> None:
        super().__init__("カルマーレシオ")
        self.annualizer = Annualizer(trading_days_per_year)

    def calculate(self, returns: pd.Series, **kwargs) -> None:
        ann_ret = self.annualizer.annualize_return(returns.mean())
        cumulative = (1 + returns).cumprod()
        dd = 1 - cumulative / cumulative.cummax()
        mdd = dd.max()
        if mdd == 0:
            self._value = float("nan")
        else:
            self._value = ann_ret / mdd
