from __future__ import annotations

from typing import Iterable, Literal, Optional
import pandas as pd

from trading import TradingFacade
from .model_order_config import ModelOrderConfig, normalize_margin_weights


class OrderExecutionFacade:
    """SBI証券でオーダーするファサード"""

    def __init__(
        self,
        mode: Literal["new", "additional", "settle", "none"],
        trade_facade: TradingFacade,
    ) -> None:
        self.mode = mode
        self.trade_facade = trade_facade

    async def execute(
        self,
        configs: ModelOrderConfig | Iterable[ModelOrderConfig] | None = None,
    ) -> Optional[pd.DataFrame]:
        """
        モデル設定を基に発注を実行する。
        
        Args:
            configs (ModelOrderConfig | Iterable[ModelOrderConfig] | None):
                modeが"new"のときのみ必須。銘柄選択用の情報を格納したインスタンス
        """

        if self.mode == "none":
            return None

        if self.mode == "new" and configs is not None:
            if isinstance(configs, ModelOrderConfig):
                configs = [configs]

            configs = list(configs)
            normalize_margin_weights(configs)

            await self.trade_facade.margin_provider.refresh()
            total_margin = (
                await self.trade_facade.margin_provider.get_available_margin()
            )

            orders_list: list[pd.DataFrame] = []
            for cfg in configs:
                alloc_margin = total_margin * cfg.margin_weight
                df = await self.trade_facade.take_positions(
                    order_price_df=cfg.ml_datasets.get_order_price(),
                    pred_result_df=cfg.ml_datasets.get_pred_result(),
                    SECTOR_REDEFINITIONS_CSV=cfg.sector_csv,
                    num_sectors_to_trade=cfg.trading_sector_num,
                    num_candidate_sectors=cfg.candidate_sector_num,
                    top_slope=cfg.top_slope,
                    margin_power=alloc_margin,
                )
                if df is not None:
                    orders_list.append(df)

            if orders_list:
                return pd.concat(orders_list, ignore_index=True)
            return None

        if self.mode == "additional":
            await self.trade_facade.take_additionals_until_completed()
        elif self.mode == "settle":
            await self.trade_facade.settle_positions()

        return None
