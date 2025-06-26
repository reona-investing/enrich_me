from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from .evaluation_metric import EvaluationMetric
from .aggregate.theoretical_max_drawdown import TheoreticalMaxDrawdown
from ..annualizer import Annualizer


class EvaluationMetricsManager:
    """複数の指標を一括管理するマネージャ"""

    def __init__(self, annualizer: 'Annualizer') -> None:
        self.annualizer = annualizer
        self.metrics: list[EvaluationMetric] = []

    def add_metric(self, metric_instance: EvaluationMetric) -> None:
        self.metrics.append(metric_instance)

    def evaluate_all(self, returns: pd.Series, **kwargs) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        extra = {}
        for metric in self.metrics:
            if isinstance(metric, TheoreticalMaxDrawdown):
                # 期待リターンと標準偏差を渡す
                extra = {
                    "expected_return": returns.mean(),
                    "std_return": returns.std(ddof=0),
                }
            metric.calculate(returns, **extra)
            results[metric.get_name()] = metric.value
        return results
