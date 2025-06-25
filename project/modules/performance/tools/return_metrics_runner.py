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
    TaxRate,
    Leverage,
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
        is_tax_excluded: bool = True,
        is_leverage_applied: bool = False,
        tax_rate: float = 0.20315,
        leverage_ratio: float = 3.1,
    ) -> None:
        self.date_series = date_series
        self.return_series = return_series
        self.label_series = correction_label_series
        self.sector_series = sector_series
        self.is_tax_excluded = is_tax_excluded
        self.is_leverage_applied = is_leverage_applied
        self.tax_rate_obj = TaxRate(tax_rate)
        self.leverage_obj = Leverage(leverage_ratio)
        self._setup_aggregate_metrics_manager()

    def _setup_aggregate_metrics_manager(self) -> None:
        self.aggregate_metrics_manager = EvaluationMetricsManager(Annualizer())
        self.aggregate_metrics_manager.add_metric(ExpectedReturn())
        self.aggregate_metrics_manager.add_metric(StandardDeviationOfReturn())
        self.aggregate_metrics_manager.add_metric(Median())
        self.aggregate_metrics_manager.add_metric(SharpeRatio())
        self.aggregate_metrics_manager.add_metric(MaxDrawdown())
        self.aggregate_metrics_manager.add_metric(TheoreticalMaxDrawdown())

    def _get_base_returns(self) -> pd.Series:
        """入力リターンを税引前・レバレッジなしの状態に変換する"""
        base = self.return_series.copy()
        if self.is_leverage_applied:
            base = self.leverage_obj.remove_leverage(base)
        if not self.is_tax_excluded:
            base = self.tax_rate_obj.remove_tax(base)
        return base

    def calculate(self) -> Dict[str, pd.DataFrame]:
        """3パターンのリターンに対する指標を計算して返す"""
        base = self._get_base_returns()

        patterns = {
            "PreTax_PreLeverage": base,
            "PreTax_PostLeverage": self.leverage_obj.apply_leverage(base),
        }
        patterns["PostTax_PostLeverage"] = self.tax_rate_obj.apply_tax(patterns["PreTax_PostLeverage"])

        results: Dict[str, pd.DataFrame] = {}
        for name, series in patterns.items():
            agg = self.aggregate_metrics_manager.evaluate_all(series)
            df = pd.DataFrame({"指標": agg.keys(), "値": agg.values()}).set_index("指標", drop=True)
            results[name] = df
        # TODO: Spearman相関やNumerai相関などのRank特徴量を
        return results
