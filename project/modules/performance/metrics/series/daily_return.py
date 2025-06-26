from __future__ import annotations

import pandas as pd

from ..base.evaluation_metric import SeriesMetric


class DailyReturn(SeriesMetric):
    """日次リターンをそのまま返す"""

    def __init__(self) -> None:
        super().__init__("日次リターン")

    def calculate(self, returns: pd.Series, **kwargs) -> None:
        df = returns.to_frame("DailyReturn")
        df.index.name = returns.index.name or "Date"
        self._value = df
