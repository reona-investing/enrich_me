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
                    order_price_df=cfg.ml_dataset.order_price_df,
                    pred_result_df=cfg.ml_dataset.pred_result_df,
                    SECTOR_REDEFINITIONS_CSV=cfg.sector_csv,
                    num_sectors_to_trade=cfg.trading_sector_num,
                    num_candidate_sectors=cfg.candidate_sector_num,
                    top_slope=cfg.top_slope,
                    margin_power=alloc_margin,
                )
                if df is not None:
                    orders_list.append(df)

            if orders_list:
                df = pd.concat(orders_list, ignore_index=True)
                if 'Code' in df.columns:
                    df = self._merge_same_codes(df)
                return df
            return None

        if self.mode == "additional":
            await self.trade_facade.take_additionals_until_completed()
        elif self.mode == "settle":
            await self.trade_facade.settle_positions()

        return None

    def _merge_same_codes(self, df: pd.DataFrame) -> pd.DataFrame:
        """同一銘柄の注文を合算し1行にまとめる"""
        df = df.copy()
        duplicate_codes = df['Code'].duplicated(keep=False)
        if not duplicate_codes.any():
            df['CumCost_byLS'] = df['TotalCost'].cumsum()
            return df

        for code, group in df.groupby('Code'):
            if len(group) == 1:
                continue

            long_sum = group.loc[group['Direction'] == 'Long', 'Unit'].sum()
            short_sum = group.loc[group['Direction'] == 'Short', 'Unit'].sum()
            x = long_sum - short_sum

            keep_idx = group.index.min()
            long_rows = group[group['Direction'] == 'Long']
            base_row = long_rows.iloc[0] if not long_rows.empty else group.iloc[0]

            if x == 0:
                df = df.drop(group.index)
                continue

            df.loc[keep_idx, 'Direction'] = 'Long' if x > 0 else 'Short'
            unit = x if x > 0 else -x
            df.loc[keep_idx, 'Unit'] = unit
            df.loc[keep_idx, 'UpperLimitCost'] = base_row['UpperLimitCost']
            df.loc[keep_idx, 'isBorrowingStock'] = base_row.get('isBorrowingStock', False)
            df.loc[keep_idx, 'TotalCost'] = unit * df.loc[keep_idx, 'EstimatedCost']
            df.loc[keep_idx, 'UpperLimitTotal'] = unit * df.loc[keep_idx, 'UpperLimitCost']

            drop_idx = [i for i in group.index if i != keep_idx]
            df = df.drop(drop_idx)

        df = df.sort_index().reset_index(drop=True)
        df['CumCost_byLS'] = df['TotalCost'].cumsum()
        return df
