from __future__ import annotations

import pandas as pd

from .transformation import TaxRate, Leverage
from dataclasses import dataclass


@dataclass
class _TimeseriesReturnTransformer:
    """リターン系列の変換を扱う補助クラス"""

    tax_rate_obj: TaxRate
    leverage_obj: Leverage

    def get_pre_tax_pre_leverage_returns(self, raw_returns: pd.Series) -> pd.Series:
        return raw_returns

    def get_pre_tax_post_leverage_returns(self, raw_returns: pd.Series) -> pd.Series:
        return self.leverage_obj.apply_leverage(raw_returns)

    def get_post_tax_post_leverage_returns(self, raw_returns: pd.Series) -> pd.Series:
        leveraged = self.leverage_obj.apply_leverage(raw_returns)
        return self.tax_rate_obj.apply_tax(leveraged)

    def get_pre_tax_pre_leverage_from_post_tax_post_leverage(
        self, post_tax_post_leverage_returns: pd.Series
    ) -> pd.Series:
        untaxed = self.tax_rate_obj.remove_tax(post_tax_post_leverage_returns)
        return self.leverage_obj.remove_leverage(untaxed)


class TimeseriesReturn:
    """税引前/後・レバレッジ有無のリターン系列を保持するクラス"""

    def __init__(
        self,
        return_series: pd.Series,
        init_tax: bool,
        init_leverage: bool,
        tax_rate: float = 0.20315,
        leverage_ratio: float = 3.1,
    ) -> None:
        self.return_series = return_series
        self.init_tax = init_tax
        self.init_leverage = init_leverage
        self.tax_rate_obj = TaxRate(tax_rate)
        self.leverage_obj = Leverage(leverage_ratio)
        self.transformer = _TimeseriesReturnTransformer(self.tax_rate_obj, self.leverage_obj)
        self._setup_series()

    def _setup_series(self) -> None:
        series = self.return_series.copy()

        if self.init_tax and self.init_leverage:
            self.post_tax_post_leverage = series
            self.pre_tax_pre_leverage = (
                self.transformer.get_pre_tax_pre_leverage_from_post_tax_post_leverage(series)
            )
        elif self.init_tax and not self.init_leverage:
            self.post_tax_pre_leverage = series
            self.pre_tax_pre_leverage = self.tax_rate_obj.remove_tax(series)
        elif not self.init_tax and self.init_leverage:
            self.pre_tax_post_leverage = series
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
