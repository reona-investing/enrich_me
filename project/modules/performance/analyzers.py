"""リターン評価アナライザー"""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
from .transformation import TaxRate, Leverage


@dataclass
class ReturnSeriesTransformer:
    """リターン系列の変換を扱うクラス"""

    tax_rate_obj: TaxRate
    leverage_obj: Leverage

    def get_pre_tax_pre_leverage_returns(self, raw_returns: pd.Series) -> pd.Series:
        return raw_returns

    def get_pre_tax_post_leverage_returns(self, raw_returns: pd.Series) -> pd.Series:
        return self.leverage_obj.apply_leverage(raw_returns)

    def get_post_tax_post_leverage_returns(self, raw_returns: pd.Series) -> pd.Series:
        leveraged = self.leverage_obj.apply_leverage(raw_returns)
        return self.tax_rate_obj.apply_tax(leveraged)

    def get_pre_tax_pre_leverage_from_post_tax_post_leverage(self, post_tax_post_leverage_returns: pd.Series) -> pd.Series:
        untaxed = self.tax_rate_obj.remove_tax(post_tax_post_leverage_returns)
        return self.leverage_obj.remove_leverage(untaxed)