from __future__ import annotations

from trading.sbi.orders.interface import IOrderExecutor
from trading.sbi.browser import SBIBrowserManager
from .components import (
    BaseExecutor,
    OrderPlacerMixin,
    OrderCancellerMixin,
    PositionSettlerMixin,
    OrderInfoFetcherMixin,
)

class SBIOrderExecutor(OrderPlacerMixin, OrderCancellerMixin, PositionSettlerMixin, OrderInfoFetcherMixin, IOrderExecutor):
    """SBI証券での注文実行を管理するクラス"""

    def __init__(self, browser_manager: SBIBrowserManager):
        BaseExecutor.__init__(self, browser_manager)
