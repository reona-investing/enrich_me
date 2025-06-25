from __future__ import annotations

from typing import Dict
import pandas as pd

from .. import (
    Annualizer,
    ExpectedReturn,
    StandardDeviationOfReturn,
    SharpeRatio,
    MaxDrawdown,
    TheoreticalMaxDrawdown,
    CumulativeReturn,
    HitRate,
    AnnualizedReturn,
    AnnualizedStandardDeviation,
    LongestDrawdownPeriod,
    AnnualizedSharpeRatio,
    CalmarRatio,
    EvaluationMetricsManager,
    Median,
    DailyReturn,
    MonthlyReturn,
    AnnualReturn,
    TaxRate,
    Leverage,
)



class ReturnMetricsRunner:
    """Calculate metrics from raw return data."""

    def __init__(
        self,
        date_series: pd.Series,
        return_series: pd.Series,
        is_tax_excluded: bool = True,
        is_leverage_applied: bool = False,
        tax_rate: float = 0.20315,
        leverage_ratio: float = 3.1,
    ) -> None:
        self.date_series = date_series
        self.return_series = return_series
        self.is_tax_excluded = is_tax_excluded
        self.is_leverage_applied = is_leverage_applied
        self.tax_rate_obj = TaxRate(tax_rate)
        self.leverage_obj = Leverage(leverage_ratio)
        self._setup_aggregate_metrics_manager()
        self._setup_series_metrics_manager()

    def _setup_aggregate_metrics_manager(self) -> None:
        self.aggregate_metrics_manager = EvaluationMetricsManager(Annualizer())
        self.aggregate_metrics_manager.add_metric(ExpectedReturn())
        self.aggregate_metrics_manager.add_metric(StandardDeviationOfReturn())
        self.aggregate_metrics_manager.add_metric(Median())
        self.aggregate_metrics_manager.add_metric(SharpeRatio())
        self.aggregate_metrics_manager.add_metric(MaxDrawdown())
        self.aggregate_metrics_manager.add_metric(TheoreticalMaxDrawdown())
        self.aggregate_metrics_manager.add_metric(HitRate())
        self.aggregate_metrics_manager.add_metric(AnnualizedReturn())
        self.aggregate_metrics_manager.add_metric(AnnualizedStandardDeviation())
        self.aggregate_metrics_manager.add_metric(LongestDrawdownPeriod())
        self.aggregate_metrics_manager.add_metric(AnnualizedSharpeRatio())
        self.aggregate_metrics_manager.add_metric(CalmarRatio())

    def _setup_series_metrics_manager(self) -> None:
        self.series_metrics_manager = EvaluationMetricsManager(Annualizer())
        self.series_metrics_manager.add_metric(DailyReturn())
        self.series_metrics_manager.add_metric(MonthlyReturn())
        self.series_metrics_manager.add_metric(AnnualReturn())
        self.series_metrics_manager.add_metric(CumulativeReturn())

    def _get_base_returns(self) -> pd.Series:
        """入力リターンを税引前・レバレッジなしの状態に変換する"""
        base = self.return_series.copy()
        if self.is_leverage_applied:
            base = self.leverage_obj.remove_leverage(base)
        if not self.is_tax_excluded:
            base = self.tax_rate_obj.remove_tax(base)
        return base

    def calculate(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """3パターンのリターンに対する指標を階層的な辞書で返す"""
        base = self._get_base_returns()

        patterns = {
            "税引前・レバレッジ無": base,
            "税引前・レバレッジ有": self.leverage_obj.apply_leverage(base),
        }
        patterns["税引後・レバレッジ有"] = self.tax_rate_obj.apply_tax(patterns["税引前・レバレッジ有"])

        results: Dict[str, Dict[str, pd.DataFrame]] = {name: {} for name in patterns.keys()}

        for name, series in patterns.items():
            agg = self.aggregate_metrics_manager.evaluate_all(series)
            df = pd.DataFrame({"指標": agg.keys(), "値": agg.values()}).set_index("指標", drop=True)
            results[name]["集計"] = df

        for name, series in patterns.items():
            series_metrics = self.series_metrics_manager.evaluate_all(series)
            for metric_name, df in series_metrics.items():
                results[name][metric_name] = df

        return results
