from __future__ import annotations

import pandas as pd

from ..base.evaluation_metric import SeriesMetric


class MonthlyCumulativeReturn(SeriesMetric):
    """月次累積リターンを計算して返す"""

    def __init__(self) -> None:
        super().__init__("月次累積リターン")

    def calculate(self, returns: pd.Series, **kwargs) -> None:
        monthly = (1 + returns).resample("ME").prod()
        cumulative = monthly.cumprod() - 1
        df = cumulative.to_frame("MonthlyCumulativeReturn")
        df.index.name = "Date"
        self._value = df
