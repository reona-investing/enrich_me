from __future__ import annotations

import pandas as pd

from .transformation import TaxRate, Leverage
from .annualizer import Annualizer
from .metrics import (
    ExpectedReturn,
    StandardDeviationOfReturn,
    SharpeRatio,
    MaxDrawdown,
    TheoreticalMaxDrawdown,
    MetricsCollection,
)
from .analyzers import ReturnSeriesTransformer, PredictionReturnAnalyzer


class PredictionReturnExecutor:
    """PredictionReturnAnalyzer を簡易に実行するクラス"""

    def __init__(
        self,
        predicted_returns: pd.Series,
        actual_returns: pd.Series,
        tax_rate: float = 0.20315,
        leverage_ratio: float = 3.1,
        trading_days_per_year: int = 252,
    ) -> None:
        if not predicted_returns.index.equals(actual_returns.index):
            raise ValueError("インデックスが一致していません")
        self.predicted_returns = predicted_returns
        self.actual_returns = actual_returns
        self.tax_rate_obj = TaxRate(tax_rate)
        self.leverage_obj = Leverage(leverage_ratio)
        self.annualizer = Annualizer(trading_days_per_year)
        self.manager = MetricsCollection(self.annualizer)
        self.manager.add_metric(ExpectedReturn())
        self.manager.add_metric(StandardDeviationOfReturn())
        self.manager.add_metric(SharpeRatio())
        self.manager.add_metric(MaxDrawdown())
        self.manager.add_metric(TheoreticalMaxDrawdown())
        self.transformer = ReturnSeriesTransformer(self.tax_rate_obj, self.leverage_obj)

    def execute(self) -> pd.DataFrame:
        analyzer = PredictionReturnAnalyzer(
            self.predicted_returns,
            self.actual_returns,
            self.transformer,
            self.manager,
        )
        return analyzer.run_analysis()
