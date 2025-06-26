from __future__ import annotations

import pandas as pd

from ..base.evaluation_metric import AggregateMetric


class StandardDeviationOfReturn(AggregateMetric):
    """標準偏差を計算"""

    def __init__(self) -> None:
        super().__init__("標準偏差")

    def calculate(self, returns: pd.Series, **kwargs) -> None:
        self._value = returns.std(ddof=0)
