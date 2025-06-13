import pandas as pd
from typing import Callable, Dict


class SectorDataPreparer:
    """株価データにセクター定義を付与するためのユーティリティ。"""

    @staticmethod
    def from_csv(stock_price_data: pd.DataFrame, csv_path: str, col_getter: Callable) -> pd.DataFrame:
        """CSV からセクター定義を読み込み株価データに結合する。"""

        _, _, sector_col = col_getter()
        new_sector_list = pd.read_csv(csv_path).dropna(how='any', axis=1)
        new_sector_list[sector_col['銘柄コード']] = new_sector_list[sector_col['銘柄コード']].astype(str)
        sector_price_data = pd.merge(new_sector_list, stock_price_data, how='right', on=sector_col['銘柄コード'])
        return sector_price_data

    @staticmethod
    def from_dict(stock_price_data: pd.DataFrame, sector_stock_dict: Dict[str, list], col_getter: Callable) -> pd.DataFrame:
        """辞書形式のセクター定義を株価データに適用する。"""

        _, _, sector_col = col_getter()
        sector_definitions = []
        for sector_name, stock_codes in sector_stock_dict.items():
            for code in stock_codes:
                sector_definitions.append({
                    sector_col['銘柄コード']: str(code),
                    sector_col['セクター']: sector_name
                })
        sector_df = pd.DataFrame(sector_definitions)
        sector_price_data = pd.merge(sector_df, stock_price_data, how='inner', on=sector_col['銘柄コード'])
        return sector_price_data
