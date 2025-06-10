from __future__ import annotations

from typing import Literal

from trading import TradingFacade


class TradeDataFacade:
    """Facade for fetching trading data."""

    def __init__(self, mode: Literal['fetch', 'none'], trade_facade: TradingFacade, sector_csv: str) -> None:
        self.mode = mode
        self.trade_facade = trade_facade
        self.sector_csv = sector_csv

    async def execute(self) -> None:
        if self.mode == 'fetch':
            await self.trade_facade.fetch_invest_result(self.sector_csv)

