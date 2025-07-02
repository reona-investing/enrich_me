from __future__ import annotations

import pandas as pd
from ..transformation import TaxRate, Leverage

class TimeseriesReturn:
    """税引前/後・レバレッジ有無のリターン系列を保持するクラス"""

    def __init__(
        self,
        return_series: pd.Series,
        is_tax_excluded: bool = False,
        is_leverage_applied: bool = False,
        tax_rate: float = 0.20315,
        leverage_ratio: float = 3.1,
    ) -> None:
        self.return_series = return_series
        self.is_tax_excluded = is_tax_excluded
        self.is_leverage_applied = is_leverage_applied
        self.tax_rate_obj = TaxRate(tax_rate)
        self.leverage_obj = Leverage(leverage_ratio)
        self._setup_series()

    def _setup_series(self) -> None:
        series = self.return_series.copy()

        if self.is_tax_excluded and self.is_leverage_applied:
            self.pre_tax_pre_leverage = self.leverage_obj.remove_leverage(self.tax_rate_obj.remove_tax(series))
        elif self.is_tax_excluded and not self.is_leverage_applied:
            self.pre_tax_pre_leverage = self.tax_rate_obj.remove_tax(series)
        elif not self.is_tax_excluded and self.is_leverage_applied:
            self.pre_tax_pre_leverage = self.leverage_obj.remove_leverage(series)
        else:
            self.pre_tax_pre_leverage = series

        # 全ての組み合わせを計算
        self.pre_tax_post_leverage = self.leverage_obj.apply_leverage(self.pre_tax_pre_leverage)
        self.post_tax_pre_leverage = self.tax_rate_obj.apply_tax(self.pre_tax_pre_leverage)
        self.post_tax_post_leverage = self.tax_rate_obj.apply_tax(self.pre_tax_post_leverage)

    def get_returns(self, taxed: bool = False, leveraged: bool = False) -> pd.Series:
        """指定された条件のリターン系列を取得する"""
        if taxed and leveraged:
            return self.post_tax_post_leverage
        if taxed and not leveraged:
            return self.post_tax_pre_leverage
        if not taxed and leveraged:
            return self.pre_tax_post_leverage
        return self.pre_tax_pre_leverage
