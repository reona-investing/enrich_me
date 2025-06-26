from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

import pandas as pd

T = TypeVar("T")


class EvaluationMetric(ABC, Generic[T]):
    """評価指標計算クラスの共通基底クラス"""

    def __init__(self, metric_name: str) -> None:
        self._metric_name = metric_name
        self._value: Optional[T] = None

    @abstractmethod
    def calculate(self, returns: pd.Series, **kwargs) -> None:
        """指標値を計算する"""
        pass

    @property
    def value(self) -> T:
        if self._value is None:
            raise ValueError("Metric value has not been calculated yet")
        return self._value

    def get_name(self) -> str:
        return self._metric_name


class AggregateMetric(EvaluationMetric[float]):
    """時系列全体を集計し単一値を返す指標の基底クラス"""

    def __init__(self, metric_name: str) -> None:
        super().__init__(metric_name)


class SeriesMetric(EvaluationMetric[pd.DataFrame]):
    """時系列形式の値を返す指標の基底クラス"""

    def __init__(self, metric_name: str) -> None:
        super().__init__(metric_name)


class RankMetric(EvaluationMetric[pd.DataFrame]):
    """順位比較を行う指標の基底クラス"""

    def __init__(self, metric_name: str) -> None:
        super().__init__(metric_name)
