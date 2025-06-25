from __future__ import annotations

import pandas as pd

from ..evaluation_metric import AggregateMetric


class HitRate(AggregateMetric):
    """リターンが正の日の割合を計算するクラス"""

    def __init__(self) -> None:
        super().__init__("勝率")

    def calculate(self, returns: pd.Series, **kwargs) -> None:
        self._value = float((returns > 0).mean())
