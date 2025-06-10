from __future__ import annotations

from typing import Literal, Optional

from models.machine_learning.ml_dataset import SingleMLDataset
from trading import TradingFacade


class OrderExecutionFacade:
    """Facade for executing trading orders."""

    def __init__(
        self,
        mode: Literal['new', 'additional', 'settle', 'none'],
        trade_facade: TradingFacade,
        sector_csv: str,
        trading_sector_num: int,
        candidate_sector_num: int,
        top_slope: float,
    ) -> None:
        self.mode = mode
        self.trade_facade = trade_facade
        self.sector_csv = sector_csv
        self.trading_sector_num = trading_sector_num
        self.candidate_sector_num = candidate_sector_num
        self.top_slope = top_slope

    async def execute(self, ml_dataset: Optional[SingleMLDataset]) -> None:
        if self.mode == 'none' or ml_dataset is None:
            return
        if self.mode == 'new':
            materials = ml_dataset.post_processing_data.getter_stock_selection()
            await self.trade_facade.take_positions(
                order_price_df=materials.order_price_df,
                pred_result_df=materials.pred_result_df,
                SECTOR_REDEFINITIONS_CSV=self.sector_csv,
                num_sectors_to_trade=self.trading_sector_num,
                num_candidate_sectors=self.candidate_sector_num,
                top_slope=self.top_slope,
            )
        elif self.mode == 'additional':
            await self.trade_facade.take_additionals_until_completed()
        elif self.mode == 'settle':
            await self.trade_facade.settle_positions()

