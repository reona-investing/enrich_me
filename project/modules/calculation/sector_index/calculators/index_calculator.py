import numpy as np
import pandas as pd
from typing import Callable


class SectorIndexCalculator:
    """セクター別の株価データからインデックス値を算出するユーティリティ。"""

    @staticmethod
    def aggregate(sector_price_data: pd.DataFrame, col_getter: Callable) -> pd.DataFrame:
        """セクターインデックスの基礎データを集約する。"""

        _, _, sector_col = col_getter()
        columns_to_sum = [
            sector_col['始値時価総額'], sector_col['終値時価総額'], sector_col['高値時価総額'],
            sector_col['安値時価総額'], sector_col['発行済み株式数'], sector_col['指数算出用の補正値']
        ]
        sector_index = sector_price_data.groupby([sector_col['日付'], sector_col['セクター']])[columns_to_sum].sum()
        sector_index[sector_col['1日リターン']] = sector_index[sector_col['終値時価総額']] / (
            sector_index.groupby(sector_col['セクター'])[sector_col['終値時価総額']].shift(1) +
            sector_index[sector_col['指数算出用の補正値']]
        ) - 1
        sector_index[sector_col['終値前日比']] = 1 + sector_index[sector_col['1日リターン']]
        sector_index[sector_col['終値']] = sector_index.groupby(sector_col['セクター'])[sector_col['終値前日比']].cumprod()
        return SectorIndexCalculator._calculate_ohlc(sector_index, col_getter)

    @staticmethod
    def _calculate_ohlc(sector_index: pd.DataFrame, col_getter: Callable) -> pd.DataFrame:
        """集約後のデータから OHLC を求める補助メソッド。"""

        _, _, sector_col = col_getter()
        sector_index[sector_col['始値']] = (
            sector_index[sector_col['終値']] *
            sector_index[sector_col['始値時価総額']] /
            sector_index[sector_col['終値時価総額']]
        ).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        sector_index[sector_col['高値']] = (
            sector_index[sector_col['終値']] *
            sector_index[sector_col['高値時価総額']] /
            sector_index[sector_col['終値時価総額']]
        ).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        sector_index[sector_col['安値']] = (
            sector_index[sector_col['終値']] *
            sector_index[sector_col['安値時価総額']] /
            sector_index[sector_col['終値時価総額']]
        ).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        return sector_index
