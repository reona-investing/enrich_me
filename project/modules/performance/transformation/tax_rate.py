"""税金の計算と適用を管理するモジュール"""

from __future__ import annotations

import pandas as pd


class TaxRate:
    """税金の計算と適用を管理する。"""

    def __init__(self, tax_rate: float = 0.20315) -> None:
        self.tax_rate = tax_rate

    def apply_tax(self, returns: pd.Series) -> pd.Series:
        """税引後リターンを計算する。"""
        taxed = returns.copy()
        taxed = taxed * (1 - self.tax_rate)
        return taxed

    def remove_tax(self, returns: pd.Series) -> pd.Series:
        """税引後リターンから税引前リターンを推定する。"""
        untaxed = returns.copy()
        untaxed = untaxed / (1 - self.tax_rate)
        return untaxed
