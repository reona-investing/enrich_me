from __future__ import annotations

import pandas as pd

from ..evaluation_metric import SeriesMetric


class DailyReturn(SeriesMetric):
    """日次リターンをそのまま返す"""

    def __init__(self) -> None:
        super().__init__("日次リターン")

    def calculate(self, returns: pd.Series, **kwargs) -> pd.DataFrame:
        df = returns.to_frame("DailyReturn")
        df.index.name = returns.index.name or "Date"
        self.value = df
        return self.value
