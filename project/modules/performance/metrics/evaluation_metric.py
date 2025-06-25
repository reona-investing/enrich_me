from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd


class EvaluationMetric(ABC):
    """評価指標計算クラスの共通基底クラス"""

    def __init__(self, metric_name: str) -> None:
        self._metric_name = metric_name
        self.value = None

    @abstractmethod
    def calculate(self, returns: pd.Series, **kwargs):
        """指標値を計算する"""
        pass

    def get_name(self) -> str:
        return self._metric_name


class AggregateMetric(EvaluationMetric):
    """時系列全体を集計し単一値を返す指標の基底クラス"""

    def __init__(self, metric_name: str) -> None:
        super().__init__(metric_name)


class SeriesMetric(EvaluationMetric):
    """時系列形式の値を返す指標の基底クラス"""

    def __init__(self, metric_name: str) -> None:
        super().__init__(metric_name)


class RankMetric(EvaluationMetric):
    """順位比較を行う指標の基底クラス"""

    def __init__(self, metric_name: str) -> None:
        super().__init__(metric_name)
