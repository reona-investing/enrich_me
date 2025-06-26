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
    MonthlyCumulativeReturn,
    AnnualCumulativeReturn,
    HitRate,
    AnnualizedReturn,
    AnnualizedStandardDeviation,
    LongestDrawdownPeriod,
    AnnualizedSharpeRatio,
    CalmarRatio,
    MetricsCollection,
    Median,
    DailyReturn,
    MonthlyReturn,
    AnnualReturn,
    Leverage,
    TaxRate,
)



class ReturnMetricsRunner:
    """Calculate metrics from raw return data."""

    def __init__(
        self,
        date_series: pd.Series,
        return_series: pd.Series,
        is_tax_excluded: bool = True,
        is_leverage_applied: bool = False,
        leverage_ratio: float = 3.1,
    ) -> None:
        self.date_series = date_series
        self.return_series = return_series
        self.is_tax_excluded = is_tax_excluded
        self.is_leverage_applied = is_leverage_applied
        self.tax_rate_obj = TaxRate()
        self.leverage_obj = Leverage(leverage_ratio)
        self._setup_aggregate_metrics_manager()
        self._setup_series_metrics_manager()

    def _setup_aggregate_metrics_manager(self) -> None:
        self.aggregate_metrics_manager = MetricsCollection(Annualizer())
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
        self.series_metrics_manager = MetricsCollection(Annualizer())
        self.series_metrics_manager.add_metric(DailyReturn())
        self.series_metrics_manager.add_metric(MonthlyReturn())
        self.series_metrics_manager.add_metric(AnnualReturn())
        self.series_metrics_manager.add_metric(CumulativeReturn())
        self.series_metrics_manager.add_metric(MonthlyCumulativeReturn())
        self.series_metrics_manager.add_metric(AnnualCumulativeReturn())

    def _remove_leverage(self, returns: pd.Series) -> pd.Series:
        """入力リターンからレバレッジの影響を除去する"""
        base = returns.copy()
        if self.is_leverage_applied:
            base = self.leverage_obj.remove_leverage(base)
        return base

    def _convert_tax_status(self, returns: pd.Series) -> tuple[pd.Series, pd.Series]:
        """入力リターンから税引前/税引後の系列を取得する"""
        base = returns.copy()
        if self.is_tax_excluded:
            taxfree = base
            taxed = self.tax_rate_obj.apply_tax(base)
        else:
            taxed = base
            taxfree = self.tax_rate_obj.remove_tax(base)
        return taxfree, taxed

    def calculate(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """3パターンのリターンに対する指標を階層的な辞書で返す"""
        base_returns = self._remove_leverage(self.return_series)
        base_taxfree, base_taxed = self._convert_tax_status(base_returns)

        patterns = {
            "税引前・レバレッジ無": base_taxfree,
            "税引前・レバレッジ有": self.leverage_obj.apply_leverage(base_taxfree),
            "税引後・レバレッジ有": self.leverage_obj.apply_leverage(base_taxed),
        }

        total_profit = (1 + base_taxed).prod() - 1

        results: Dict[str, Dict[str, pd.DataFrame]] = {name: {} for name in patterns.keys()}

        for name, series in patterns.items():
            agg = self.aggregate_metrics_manager.evaluate_all(series)
            df = pd.DataFrame({"指標": agg.keys(), "値": agg.values()}).set_index("指標", drop=True)
            df.loc["通算損益"] = total_profit
            results[name]["集計"] = df

        for name, series in patterns.items():
            series_metrics = self.series_metrics_manager.evaluate_all(series)
            for metric_name, df in series_metrics.items():
                results[name][metric_name] = df

        return results
