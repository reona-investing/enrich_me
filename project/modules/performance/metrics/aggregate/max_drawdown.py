from __future__ import annotations

import pandas as pd

from ..base.evaluation_metric import AggregateMetric


class MaxDrawdown(AggregateMetric):
    """最大ドローダウン（実績）を計算"""

    def __init__(self) -> None:
        super().__init__("最大ドローダウン")

    def calculate(self, returns: pd.Series, **kwargs) -> None:
        cumulative = (1 + returns).cumprod()
        dd = 1 - cumulative / cumulative.cummax()
        self._value = dd.max()
