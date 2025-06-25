"""税率・レバレッジ計算用クラス群"""

from __future__ import annotations

import pandas as pd


class TaxRate:
    """税金の計算と適用を管理する。"""

    def __init__(self, tax_rate: float = 0.20315) -> None:
        self.tax_rate = tax_rate

    def apply_tax(self, returns: pd.Series) -> pd.Series:
        """税引後リターンを計算する。"""
        taxed = returns.copy()
        taxed[taxed > 0] = taxed[taxed > 0] * (1 - self.tax_rate)
        return taxed

    def remove_tax(self, returns: pd.Series) -> pd.Series:
        """税引後リターンから税引前リターンを推定する。"""
        untaxed = returns.copy()
        untaxed[untaxed > 0] = untaxed[untaxed > 0] / (1 - self.tax_rate)
        return untaxed


class Leverage:
    """レバレッジの適用と解除を管理する。"""

    def __init__(self, leverage_ratio: float = 3.1) -> None:
        self.leverage_ratio = leverage_ratio

    def apply_leverage(self, returns: pd.Series) -> pd.Series:
        """レバレッジ適用後リターンを計算する。"""
        return returns * self.leverage_ratio

    def remove_leverage(self, leveraged_returns: pd.Series) -> pd.Series:
        """レバレッジ除去後のリターンを計算する。"""
        return leveraged_returns / self.leverage_ratio
