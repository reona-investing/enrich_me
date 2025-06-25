from __future__ import annotations

import pandas as pd

from .evaluation_metric import AggregateMetric


class LongestDrawdownPeriod(AggregateMetric):
    """ドローダウンが続いた最長期間（日数）を計算するクラス"""

    def __init__(self) -> None:
        super().__init__("最長ドローダウン期間")

    def calculate(self, returns: pd.Series, **kwargs) -> float:
        cumulative = (1 + returns).cumprod()
        peaks = cumulative.cummax()
        drawdown = cumulative < peaks
        longest = 0
        current = 0
        for flag in drawdown:
            if flag:
                current += 1
            else:
                longest = max(longest, current)
                current = 0
        longest = max(longest, current)
        return float(longest)
