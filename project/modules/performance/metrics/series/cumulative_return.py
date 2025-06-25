from __future__ import annotations

import pandas as pd

from ..evaluation_metric import SeriesMetric


class CumulativeReturn(SeriesMetric):
    """累積リターンを計算するクラス"""

    def __init__(self) -> None:
        super().__init__("累積リターン")

    def calculate(self, returns: pd.Series, **kwargs) -> None:
        cumulative = (1 + returns).cumprod() - 1
        df = cumulative.to_frame("CumulativeReturn")
        df.index.name = returns.index.name or "Date"
        self._value = df
