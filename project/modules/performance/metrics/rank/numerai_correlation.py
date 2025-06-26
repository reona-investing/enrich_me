from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..base.evaluation_metric import RankMetric


class NumeraiCorrelation(RankMetric):
    """Numerai相関を計算するクラス。"""

    def __init__(self) -> None:
        super().__init__("Numerai相関")

    def _calc_daily_numerai_corr(self, target_rank: pd.Series, pred_rank: pd.Series) -> float:
        """1日分の Numerai 相関を計算する。"""
        pred_rank = np.array(pred_rank)
        scaled_pred = (pred_rank - 0.5) / len(pred_rank)
        gauss_pred = norm.ppf(scaled_pred)
        pred_p15 = np.sign(gauss_pred) * np.abs(gauss_pred) ** 1.5

        target_rank = np.array(target_rank)
        centered_target = target_rank - target_rank.mean()
        target_p15 = np.sign(centered_target) * np.abs(centered_target) ** 1.5

        return float(np.corrcoef(pred_p15, target_p15)[0, 1])

    def _calc_daily_numerai_rank_corr(self, target_rank: pd.Series, pred_rank: pd.Series) -> float:
        """Targetをランク化した Numerai 相関を計算する。"""
        processed = []
        for arr in (target_rank, pred_rank):
            arr = np.array(arr)
            scaled = (arr - 0.5) / len(arr)
            gauss = norm.ppf(scaled)
            p15 = np.sign(gauss) * np.abs(gauss) ** 1.5
            processed.append(p15)
        return float(np.corrcoef(processed[0], processed[1])[0, 1])

    def calculate(self, series1: pd.Series, **kwargs) -> None:
        """予測順位と実際順位から Numerai 相関を計算し統計量を返す。"""
        series2: pd.Series = kwargs.get("series2")
        if series2 is None:
            raise ValueError("series2 が必要です")

        df = pd.concat([series1, series2], axis=1, keys=["pred", "actual"]).dropna()

        if isinstance(df.index, pd.MultiIndex) and "Date" in df.index.names:
            df["pred_rank"] = df.groupby("Date")["pred"].rank()
            df["actual_rank"] = df.groupby("Date")["actual"].rank()
            daily_corr = df.groupby("Date").apply(
                lambda x: self._calc_daily_numerai_corr(x["actual_rank"], x["pred_rank"])
            )
            daily_rank_corr = df.groupby("Date").apply(
                lambda x: self._calc_daily_numerai_rank_corr(x["actual_rank"], x["pred_rank"])
            )
            result = pd.concat([daily_corr, daily_rank_corr], axis=1)
            result.columns = ["NumeraiCorr", "Rank_NumeraiCorr"]
            self._value = result.describe().T
            return

        pred_rank = series1.rank()
        actual_rank = series2.rank()
        corr = self._calc_daily_numerai_corr(actual_rank, pred_rank)
        rank_corr = self._calc_daily_numerai_rank_corr(actual_rank, pred_rank)
        df_out = pd.DataFrame({
            "NumeraiCorr": [corr],
            "Rank_NumeraiCorr": [rank_corr],
        })
        self._value = df_out.describe().T

    def get_name(self) -> str:
        return "NumeraiCorrelation"
