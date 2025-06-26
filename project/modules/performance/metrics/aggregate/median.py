from __future__ import annotations

import pandas as pd

from ..base.evaluation_metric import AggregateMetric


class Median(AggregateMetric):
    """中央値を計算するクラス"""

    def __init__(self) -> None:
        super().__init__("中央値")

    def calculate(self, returns: pd.Series, **kwargs) -> None:
        self._value = returns.median()
