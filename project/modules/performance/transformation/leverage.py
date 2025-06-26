"""レバレッジ計算を扱うモジュール"""

from __future__ import annotations

import pandas as pd


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
