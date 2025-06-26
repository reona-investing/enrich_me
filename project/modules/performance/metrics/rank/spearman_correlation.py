from __future__ import annotations

import pandas as pd
from scipy.stats import spearmanr

from ..base.evaluation_metric import RankMetric


class SpearmanCorrelation(RankMetric):
    """Spearman順位相関を計算するクラス。"""

    def __init__(self) -> None:
        super().__init__("スピアマン順位相関")

    def calculate(self, series1: pd.Series, **kwargs) -> None:
        """予測と実績の順位相関を日次で計算し統計量を返す。"""
        series2: pd.Series = kwargs.get("series2")
        if series2 is None:
            raise ValueError("series2 が必要です")

        df = pd.concat([series1, series2], axis=1, keys=["pred", "actual"]).dropna()

        if isinstance(df.index, pd.MultiIndex) and "Date" in df.index.names:
            df["pred_rank"] = df.groupby("Date")["pred"].rank()
            df["actual_rank"] = df.groupby("Date")["actual"].rank()
            daily_corr = df.groupby("Date").apply(
                lambda x: spearmanr(x["pred_rank"], x["actual_rank"])[0]
            )
            self._value = daily_corr.to_frame("SpearmanCorr").describe().T
            return

        corr, _ = spearmanr(df["pred"], df["actual"])
        self._value = pd.DataFrame({"SpearmanCorr": [corr]}).describe().T
