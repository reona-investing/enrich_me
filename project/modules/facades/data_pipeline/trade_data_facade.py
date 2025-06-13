from __future__ import annotations

from typing import Literal

from trading import TradingFacade


class TradeDataFacade:
    """取引データを取得するコード"""

    def __init__(self, mode: Literal['fetch', 'none'], trade_facade: TradingFacade, sector_csv: str) -> None:
        self.mode = mode
        self.trade_facade = trade_facade
        self.sector_csv = sector_csv

    async def execute(self) -> None:
        if self.mode == 'fetch':
            await self.trade_facade.fetch_invest_result(self.sector_csv)


if __name__ == '__main__':
    from utils.paths import Paths
    import asyncio
    async def main():
        SECTOR_CSV = f'{Paths.SECTOR_REDEFINITIONS_FOLDER}/48sectors_2024-2025.csv'
        tdf = TradeDataFacade(mode='fetch', trade_facade=TradingFacade(), sector_csv=SECTOR_CSV)
        await tdf.execute()
    
    asyncio.get_event_loop().run_until_complete(main())

    


