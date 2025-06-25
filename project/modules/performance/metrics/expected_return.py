from __future__ import annotations

import pandas as pd

from .evaluation_metric import AggregateMetric


class ExpectedReturn(AggregateMetric):
    """期待リターン（平均）を計算"""

    def __init__(self) -> None:
        super().__init__("期待リターン")

    def calculate(self, returns: pd.Series, **kwargs) -> float:
        return returns.mean()
