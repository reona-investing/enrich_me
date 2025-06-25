from __future__ import annotations

from typing import Optional, Dict, Union
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
    DailyReturn,
    MonthlyReturn,
    AnnualReturn,
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
        trade_sector_numbers: int = 1,
        top_slope: float = 1.0,
        is_tax_excluded: bool = True,
        is_leverage_applied: bool = False,
        tax_rate: float = 0.20315,
        leverage_ratio: float = 3.1,
    ) -> None:
        self.date_series = date_series
        self.return_series = return_series
        self.label_series = correction_label_series
        self.sector_series = sector_series
        self.trade_sector_numbers = trade_sector_numbers
        self.top_slope = top_slope
        self.is_tax_excluded = is_tax_excluded
        self.is_leverage_applied = is_leverage_applied
        self.tax_rate_obj = TaxRate(tax_rate)
        self.leverage_obj = Leverage(leverage_ratio)
        self._setup_aggregate_metrics_manager()
        self._setup_series_metrics_manager()

    def _generate_corrected_returns(self) -> Optional[pd.Series]:
        """correction_label_series が与えられている場合に LS リターンを計算する"""
        if self.label_series is None or self.sector_series is None:
            return None

        df = pd.DataFrame({
            "Date": self.date_series,
            "Target": self.return_series,
            "Pred": self.label_series,
            "Sector": self.sector_series,
        }).dropna()

        df["TargetRank"] = df.groupby("Date")["Target"].rank(ascending=False)
        df["PredRank"] = df.groupby("Date")["Pred"].rank(ascending=False)

        sector_numbers = df["Sector"].nunique()
        long_df = df[df["PredRank"] <= self.trade_sector_numbers]
        short_df = df[df["PredRank"] > sector_numbers - self.trade_sector_numbers]

        if self.trade_sector_numbers > 1:
            long_df.loc[long_df["PredRank"] == 1, "Target"] *= self.top_slope
            long_df.loc[long_df["PredRank"] != 1, "Target"] *= 1 - (self.top_slope - 1) / (
                self.trade_sector_numbers - 1
            )
            short_df.loc[short_df["PredRank"] == sector_numbers, "Target"] *= self.top_slope
            short_df.loc[short_df["PredRank"] != sector_numbers, "Target"] *= 1 - (
                self.top_slope - 1
            ) / (self.trade_sector_numbers - 1)

        long_return = long_df.groupby("Date")[["Target"]].mean()
        short_return = -short_df.groupby("Date")[["Target"]].mean()
        ls_return = (long_return + short_return) / 2
        ls_return = ls_return["Target"]

        return ls_return

    def _setup_aggregate_metrics_manager(self) -> None:
        self.aggregate_metrics_manager = EvaluationMetricsManager(Annualizer())
        self.aggregate_metrics_manager.add_metric(ExpectedReturn())
        self.aggregate_metrics_manager.add_metric(StandardDeviationOfReturn())
        self.aggregate_metrics_manager.add_metric(Median())
        self.aggregate_metrics_manager.add_metric(SharpeRatio())
        self.aggregate_metrics_manager.add_metric(MaxDrawdown())
        self.aggregate_metrics_manager.add_metric(TheoreticalMaxDrawdown())

    def _setup_series_metrics_manager(self) -> None:
        self.series_metrics_manager = EvaluationMetricsManager(Annualizer())
        self.series_metrics_manager.add_metric(DailyReturn())
        self.series_metrics_manager.add_metric(MonthlyReturn())
        self.series_metrics_manager.add_metric(AnnualReturn())

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
        corrected = self._generate_corrected_returns()
        if corrected is not None:
            base = corrected.copy()
            if self.is_leverage_applied:
                base = self.leverage_obj.remove_leverage(base)
            if not self.is_tax_excluded:
                base = self.tax_rate_obj.remove_tax(base)
        else:
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
