from typing import List, Optional
import pandas as pd

from trading.sbi.orders.interface import IOrderExecutor, OrderRequest, OrderResult
from trading.sbi.browser import SBIBrowserManager

from .order_placer import SBIOrderPlacer
from .order_canceller import SBIOrderCanceller
from .order_inquiry import SBIOrderInquiry
from .position_manager import SBIPositionManager


class SBIOrderExecutor(IOrderExecutor):
    """SBI証券での注文実行責務を小クラスに委譲したファサード"""

    def __init__(self, browser_manager: SBIBrowserManager):
        self.browser_manager = browser_manager
        self.placer = SBIOrderPlacer(browser_manager)
        self.canceller = SBIOrderCanceller(browser_manager)
        self.inquiry = SBIOrderInquiry(browser_manager)
        self.position_manager = SBIPositionManager(browser_manager)

    async def place_order(self, order_request: OrderRequest) -> OrderResult:
        return await self.placer.place_order(order_request)

    async def cancel_all_orders(self) -> List[OrderResult]:
        return await self.canceller.cancel_all_orders()

    async def settle_position(self, symbol_code: str, unit: Optional[int] = None) -> OrderResult:
        return await self.position_manager.settle_position(symbol_code, unit)

    async def get_active_orders(self) -> pd.DataFrame:
        return await self.inquiry.get_active_orders()

    async def get_positions(self) -> pd.DataFrame:
        return await self.position_manager.get_positions()
