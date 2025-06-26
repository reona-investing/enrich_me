from __future__ import annotations

import pandas as pd

from ..evaluation_metric import SeriesMetric


class AnnualCumulativeReturn(SeriesMetric):
    """年次累積リターンを計算して返す"""

    def __init__(self) -> None:
        super().__init__("年次累積リターン")

    def calculate(self, returns: pd.Series, **kwargs) -> None:
        annual = (1 + returns).resample("YE").prod()
        cumulative = annual.cumprod() - 1
        df = cumulative.to_frame("AnnualCumulativeReturn")
        df.index.name = "Date"
        self._value = df
