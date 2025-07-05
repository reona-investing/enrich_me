from typing import Callable, Dict
import pandas as pd

from .sector_index.calculators.marketcap_calculator import MarketCapCalculator
from .sector_index.preparers.sector_data_preparer import SectorDataPreparer


class OrderPriceCalculator:
    """発注用株価データを計算するクラス。"""

    def __init__(self) -> None:
        self._marketcap_calc = MarketCapCalculator()
        self._data_preparer = SectorDataPreparer()

    def calculate_order_price(
        self,
        stock_price: pd.DataFrame,
        stock_fin: pd.DataFrame,
        col_getter: Callable,
    ) -> pd.DataFrame:
        """発注処理用の株価データを計算して返す。"""
        stock_price_with_shares = self._marketcap_calc.merge_stock_price_and_shares(
            stock_price, stock_fin, col_getter
        )
        stock_price_cap = self._marketcap_calc.calc_adjustment_factor(
            stock_price_with_shares, stock_price, col_getter
        )
        stock_price_cap = self._marketcap_calc.adjust_shares(stock_price_cap, col_getter)
        stock_price_cap = self._marketcap_calc.calc_marketcap(stock_price_cap, col_getter)
        stock_price_cap = self._marketcap_calc.calc_correction_value(
            stock_price_cap, col_getter
        )

        _, price_col, sector_col = col_getter()
        return stock_price_cap[[
            sector_col['日付'],
            sector_col['銘柄コード'],
            sector_col['終値時価総額'],
            sector_col['終値'],
            price_col['取引高'],
        ]]

    def get_order_price_dict(
        self,
        order_price_df: pd.DataFrame,
        sector_redefinitions_csv: str,
        col_getter: Callable,
    ) -> Dict[str, pd.DataFrame]:
        """セクター別に分割した発注処理用株価データを返す。"""
        order_price_data = self._data_preparer.from_csv(
            order_price_df,
            sector_redefinitions_csv,
            col_getter,
        )
        _, _, sector_col = col_getter()
        date_name = sector_col['日付']
        sector_name = sector_col['セクター']
        code_name = sector_col['銘柄コード']

        order_dict: Dict[str, pd.DataFrame] = {}
        for sec in order_price_data[sector_name].unique():
            df = order_price_data[order_price_data[sector_name] == sec].copy()
            df = df.set_index([date_name, code_name])
            order_dict[sec] = df
        return order_dict
