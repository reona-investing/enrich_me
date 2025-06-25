from __future__ import annotations

from typing import Optional, Dict
import pandas as pd

from .. import (
    Annualizer,
    ExpectedReturn,
    StandardDeviationOfReturn,
    SharpeRatio,
    MaxDrawdown,
    TheoreticalMaxDrawdown,
    EvaluationMetricsManager,
    SpearmanCorrelation,
    Median,
)

from .daily_return_generator import DailyReturnGenerator


class ReturnMetricsRunner:
    """Calculate metrics from raw return data."""

    def __init__(
        self,
        date_series: pd.Series,
        return_series: pd.Series,
        sector_series: Optional[pd.Series] = None,
        correction_label_series: Optional[pd.Series] = None,
    ) -> None:
        self.date_series = date_series
        self.return_series = return_series
        self.label_series = correction_label_series
        self.sector_series = sector_series
        self._setup_aggregate_metrics_manager()

    def _setup_aggregate_metrics_manager(self) -> None:
        self.aggregate_metrics_manager = EvaluationMetricsManager(Annualizer())
        self.aggregate_metrics_manager.add_metric(ExpectedReturn())
        self.aggregate_metrics_manager.add_metric(StandardDeviationOfReturn())
        self.aggregate_metrics_manager.add_metric(Median())
        self.aggregate_metrics_manager.add_metric(SharpeRatio())
        self.aggregate_metrics_manager.add_metric(MaxDrawdown())
        self.aggregate_metrics_manager.add_metric(TheoreticalMaxDrawdown())

    def calculate(self) -> Dict[str, float]:
        results = self.aggregate_metrics_manager.evaluate_all(self.return_series)
        #TODO Spearman相関やNumerai相関などのRank特徴量を
        return results
