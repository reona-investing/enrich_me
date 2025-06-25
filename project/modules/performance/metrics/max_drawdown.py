from __future__ import annotations

import pandas as pd

from .evaluation_metric import AggregateMetric


class MaxDrawdown(AggregateMetric):
    """最大ドローダウン（実績）を計算"""

    def __init__(self) -> None:
        super().__init__("最大ドローダウン")

    def calculate(self, returns: pd.Series, **kwargs) -> float:
        cumulative = (1 + returns).cumprod()
        dd = 1 - cumulative / cumulative.cummax()
        return dd.max()
