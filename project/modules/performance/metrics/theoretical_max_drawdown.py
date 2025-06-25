from __future__ import annotations

import pandas as pd

from .evaluation_metric import AggregateMetric


class TheoreticalMaxDrawdown(AggregateMetric):
    """理論上の最大ドローダウンを計算"""

    def __init__(self) -> None:
        super().__init__("理論上最大ドローダウン")

    def calculate(self, returns: pd.Series, **kwargs) -> float:
        mean = kwargs.get("expected_return")
        std = kwargs.get("std_return")
        if mean is None:
            mean = returns.mean()
        if std is None:
            std = returns.std(ddof=0)
        if mean == 0:
            return float("nan")
        return (std ** 2) / mean * 9 / 4
