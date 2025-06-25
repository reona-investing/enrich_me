"""年次換算用ユーティリティ"""

from __future__ import annotations

import math


class Annualizer:
    """日次リターンやボラティリティを年次換算するクラス。"""

    def __init__(self, trading_days_per_year: int = 252) -> None:
        self.trading_days_per_year = trading_days_per_year

    def annualize_return(self, daily_return: float) -> float:
        """日次リターンを年率換算する。"""
        return (1 + daily_return) ** self.trading_days_per_year - 1

    def annualize_volatility(self, daily_volatility: float) -> float:
        """日次ボラティリティを年率換算する。"""
        return daily_volatility * math.sqrt(self.trading_days_per_year)
