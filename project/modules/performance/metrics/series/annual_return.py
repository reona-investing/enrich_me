from __future__ import annotations

import pandas as pd

from ..evaluation_metric import SeriesMetric


class AnnualReturn(SeriesMetric):
    """年次リターンを計算して返す"""

    def __init__(self) -> None:
        super().__init__("年次リターン")

    def calculate(self, returns: pd.Series, **kwargs) -> None:
        annual = (1 + returns).resample("YE").prod() - 1
        df = annual.to_frame("AnnualReturn")
        df.index.name = "Date"
        self._value = df
