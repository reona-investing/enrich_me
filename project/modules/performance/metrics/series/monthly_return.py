from __future__ import annotations

import pandas as pd

from ..base.evaluation_metric import SeriesMetric


class MonthlyReturn(SeriesMetric):
    """月次リターンを計算して返す"""

    def __init__(self) -> None:
        super().__init__("月次リターン")

    def calculate(self, returns: pd.Series, **kwargs) -> None:
        monthly = (1 + returns).resample("ME").prod() - 1
        df = monthly.to_frame("MonthlyReturn")
        df.index.name = "Date"
        self._value = df
