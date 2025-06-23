from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal
from trading import TradingFacade
from models.machine_learning.ml_dataset import MLDatasets

@dataclass
class ModelOrderConfig:
    ml_datasets: MLDatasets
    sector_csv: str
    trading_sector_num: int
    candidate_sector_num: int
    top_slope: float
    margin_weight: float = 0.5

class MultiModelOrderExecutionFacade:
    """複数モデルの予測結果を用いた発注を管理するファサード"""

    def __init__(
        self,
        mode: Literal["new", "none"],
        trade_facade: TradingFacade,
        configs: List[ModelOrderConfig],
    ):
        self.mode = mode
        self.trade_facade = trade_facade
        self.configs = configs
        self._normalize_margin_weights()

    def _normalize_margin_weights(self) -> None:
        """Ensure that margin_weight of configs sums to 1."""
        if not self.configs:
            return
        total = sum(cfg.margin_weight for cfg in self.configs)
        if total <= 0:
            equal_w = 1 / len(self.configs)
            for cfg in self.configs:
                cfg.margin_weight = equal_w
        elif total != 1:
            for cfg in self.configs:
                cfg.margin_weight = cfg.margin_weight / total

    async def execute(self) -> None:
        if self.mode == 'none':
            return
        if self.mode != 'new':
            raise NotImplementedError("現在のところ 'new' モードのみ対応しています。")

        await self.trade_facade.margin_provider.refresh()
        total_margin = await self.trade_facade.margin_provider.get_available_margin()

        for cfg in self.configs:
            ml_datasets = cfg.ml_datasets
            alloc_margin = total_margin * cfg.margin_weight
            await self.trade_facade.take_positions(
                order_price_df=ml_datasets.get_order_price(),
                pred_result_df=ml_datasets.get_pred_result(),
                SECTOR_REDEFINITIONS_CSV=cfg.sector_csv,
                num_sectors_to_trade=cfg.trading_sector_num,
                num_candidate_sectors=cfg.candidate_sector_num,
                top_slope=cfg.top_slope,
                margin_power=alloc_margin,
            )

