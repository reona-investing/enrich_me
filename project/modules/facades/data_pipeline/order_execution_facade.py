from __future__ import annotations

from typing import Literal, Optional

from models.machine_learning.ml_dataset import MLDatasets
from trading import TradingFacade


class OrderExecutionFacade:
    """SBI証券でオーダーする"""

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

    async def execute(self, ml_datasets: Optional[MLDatasets] = None) -> None:
        if self.mode == 'none':
            return
        if self.mode == 'new' and ml_datasets is not None:
            materials = ml_datasets.post_processing_data.getter_stock_selection()
            await self.trade_facade.take_positions(
                order_price_df=ml_datasets.get_order_price(),
                pred_result_df=ml_datasets.get_pred_result(),
                SECTOR_REDEFINITIONS_CSV=self.sector_csv,
                num_sectors_to_trade=self.trading_sector_num,
                num_candidate_sectors=self.candidate_sector_num,
                top_slope=self.top_slope,
            )
        elif self.mode == 'additional':
            await self.trade_facade.take_additionals_until_completed()
        elif self.mode == 'settle':
            await self.trade_facade.settle_positions()
        else: 
            return None

