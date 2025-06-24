from __future__ import annotations

from typing import Literal

from utils.paths import Paths

from trading import TradingFacade


class TradeDataFacade:
    """取引データを取得するコード"""

    def __init__(self, mode: Literal['fetch', 'none'], trade_facade: TradingFacade, orders_csv: str = Paths.ORDERS_CSV) -> None:
        self.mode = mode
        self.trade_facade = trade_facade
        self.orders_csv = orders_csv

    async def execute(self) -> None:
        if self.mode == 'fetch':
            await self.trade_facade.fetch_invest_result(self.orders_csv)


if __name__ == '__main__':
    from utils.paths import Paths
    import asyncio

    async def main():
        tdf = TradeDataFacade(mode='fetch', trade_facade=TradingFacade())
        await tdf.execute()
    
    asyncio.get_event_loop().run_until_complete(main())

    


