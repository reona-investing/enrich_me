from __future__ import annotations

from typing import Optional, Dict
import pandas as pd

from modules.performance import (
    Annualizer,
    ExpectedReturn,
    StandardDeviationOfReturn,
    SharpeRatio,
    MaxDrawdown,
    TheoreticalMaxDrawdown,
    EvaluationMetricsManager,
    SpearmanCorrelation,
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
        self.generator = DailyReturnGenerator(date_series, return_series, sector_series)
        self.label_series = correction_label_series
        self.sector_series = sector_series
        self._setup_manager()

    def _setup_manager(self) -> None:
        self.manager = EvaluationMetricsManager(Annualizer())
        self.manager.add_metric(ExpectedReturn())
        self.manager.add_metric(StandardDeviationOfReturn())
        self.manager.add_metric(SharpeRatio())
        self.manager.add_metric(MaxDrawdown())
        self.manager.add_metric(TheoreticalMaxDrawdown())

    def calculate(self) -> Dict[str, float]:
        daily_returns = self.generator.generate()
        results = self.manager.evaluate_all(daily_returns)
        if self.sector_series is not None and self.label_series is not None:
            label_gen = DailyReturnGenerator(self.generator.date_series, self.label_series, self.sector_series)
            daily_labels = label_gen.generate_label_series(self.label_series)
            corr_metric = SpearmanCorrelation()
            corr_df = corr_metric.calculate(daily_returns, series2=daily_labels)
            if "SpearmanCorr" in corr_df.index:
                results[corr_metric.get_name()] = corr_df.loc["SpearmanCorr", "mean"]
            else:
                results[corr_metric.get_name()] = float("nan")
        return results
