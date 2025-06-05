#%% モジュールのインポート
from utils.paths import Paths
from utils.yaml_utils import ColumnConfigsGetter
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Tuple

from .marketcap_calculator import MarketCapCalculator
from .sector_data_preparer import SectorDataPreparer
from .index_calculator import SectorIndexCalculator


class SectorIndex:
    def __init__(self, stock_dfs_dict: dict | None = None, sector_redefinitions_csv: str | None = None, sector_index_parquet: str | None = None):
        self.stock_dfs_dict = stock_dfs_dict
        self.sector_redefinitions_csv = sector_redefinitions_csv
        self.sector_index_parquet = sector_index_parquet
        self.sector_index_df: pd.DataFrame | None = None
        self.stock_price_for_order: pd.DataFrame | None = None
        self.marketcap_df: pd.DataFrame | None = None
        self._marketcap_calc = MarketCapCalculator()
        self._data_preparer = SectorDataPreparer()
        self._index_calc = SectorIndexCalculator()

    _col_names = None
    
    @staticmethod
    def _get_column_names() -> Tuple[dict, dict, dict]:
        if SectorIndex._col_names is None:
            fin_col_configs = ColumnConfigsGetter(Paths.STOCK_FIN_COLUMNS_YAML)
            fin_cols = fin_col_configs.get_all_columns_name_asdict()
            price_col_configs = ColumnConfigsGetter(Paths.STOCK_PRICE_COLUMNS_YAML)
            price_cols = price_col_configs.get_all_columns_name_asdict()
            sector_col_configs = ColumnConfigsGetter(Paths.SECTOR_INDEX_COLUMNS_YAML)
            sector_cols = sector_col_configs.get_all_columns_name_asdict()
            
            SectorIndex._col_names = (fin_cols, price_cols, sector_cols)
        return SectorIndex._col_names

    # ========================================
    # パブリックメソッド（外部から呼び出し可能）
    # ========================================

    def calc_sector_index(
        self, stock_dfs_dict: dict, SECTOR_REDEFINITIONS_CSV: str, SECTOR_INDEX_PARQUET: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.stock_dfs_dict = stock_dfs_dict
        self.sector_redefinitions_csv = SECTOR_REDEFINITIONS_CSV
        self.sector_index_parquet = SECTOR_INDEX_PARQUET
        '''
        セクターインデックスを算出します。
        Args:
            stock_dfs_dict (dict): 'list', 'fin', 'price'が定義されたデータフレーム
            SECTOR_REDEFINITIONS_CSV (str): セクター定義の設定ファイルのパス
            SECTOR_INDEX_PARQUET (str): セクターインデックスを出力するparquetファイルのパス
        Returns:
            pd.DataFrame: セクターインデックスを格納
            pd.DataFrame: 発注用に、個別銘柄の終値と時価総額を格納
        '''
        _, price_col, sector_col = SectorIndex._get_column_names()
        stock_price = stock_dfs_dict['price']
        stock_fin = stock_dfs_dict['fin']

        # 価格情報に発行済み株式数の情報を結合
        stock_price_for_order = self.calc_marketcap(stock_price, stock_fin)

        # セクター定義を読み込み、株価データと結合
        sector_price_data = self._data_preparer.from_csv(stock_price_for_order, SECTOR_REDEFINITIONS_CSV, self._get_column_names)

        # 共通のインデックス計算処理
        new_sector_price = self._index_calc.aggregate(sector_price_data, self._get_column_names)
        
        # データフレームを保存して、インデックスを設定
        new_sector_price = new_sector_price.reset_index()
        new_sector_price.to_parquet(SECTOR_INDEX_PARQUET)
        new_sector_price = new_sector_price.set_index([sector_col['日付'], sector_col['セクター']])

        stock_price_for_order = stock_price_for_order[[
            sector_col['日付'], sector_col['銘柄コード'], sector_col['終値時価総額'], 
            sector_col['終値'], price_col['取引高']
        ]]
        self.sector_index_df = new_sector_price
        self.stock_price_for_order = stock_price_for_order
        print('セクターのインデックス値の算出が完了しました。')
        
        return new_sector_price, stock_price_for_order

    def calc_sector_index_by_dict(self, sector_stock_dict: dict, stock_price_data: pd.DataFrame) -> pd.DataFrame:
        """
        セクター名をキーとし、そのセクターに属する銘柄コードの配列を値とする辞書から
        セクターインデックスを算出します。同じ銘柄コードが複数のセクターに含まれる場合も対応します。
        
        Args:
            sector_stock_dict (dict): セクター名をキー、銘柄コード配列を値とする辞書
                                    例: {'JPY感応株': ['6758', '6501', '6702'], 
                                        'JPbond感応株': ['6751', '8306', '8316', '8411']}
            stock_price_data (pd.DataFrame): 株価データ（SectorIndex.calc_marketcapの出力と同じ構造）
            
        Returns:
            pd.DataFrame: セクターインデックスのデータフレーム
        """
        # セクター定義を辞書から作成し、株価データと結合
        sector_price_data = self._data_preparer.from_dict(stock_price_data, sector_stock_dict, self._get_column_names)

        # 共通のインデックス計算処理
        sector_index = self._index_calc.aggregate(sector_price_data, self._get_column_names)
        self.sector_index_df = sector_index
        
        return sector_index

    def calc_marketcap(self, stock_price: pd.DataFrame, stock_fin: pd.DataFrame) -> pd.DataFrame:
        '''
        各銘柄の日ごとの時価総額を算出する。
        Args:
            stock_price (pd.DataFrame): 価格情報
            stok_fin (pd.DataFrame): 財務情報
        Returns:
            pd.DataFrame: 価格情報に時価総額を付記
        '''
        # 価格情報に発行済み株式数の情報を照合
        stock_price_with_shares = self._marketcap_calc.merge_stock_price_and_shares(stock_price, stock_fin, self._get_column_names)
        # 発行済み株式数の補正係数を算出
        stock_price_cap = self._marketcap_calc.calc_adjustment_factor(stock_price_with_shares, stock_price, self._get_column_names)
        stock_price_cap = self._marketcap_calc.adjust_shares(stock_price_cap, self._get_column_names)
        # 時価総額と指数計算用の補正値を算出
        stock_price_cap = self._marketcap_calc.calc_marketcap(stock_price_cap, self._get_column_names)
        stock_price_cap = self._marketcap_calc.calc_correction_value(stock_price_cap, self._get_column_names)
        self.marketcap_df = stock_price_cap
        
        return stock_price_cap
if __name__ == '__main__':
    from acquisition.jquants_api_operations.facades import StockAcquisitionFacade
    acq = StockAcquisitionFacade(filter = "(Listing==1)&((ScaleCategory=='TOPIX Core30')|(ScaleCategory=='TOPIX Large70')|(ScaleCategory=='TOPIX Mid400')|(ScaleCategory=='TOPIX Small 1'))")
    stock_dfs = acq.get_stock_data_dict()

    sic = SectorIndex()
    sector_price_df, order_price_df = sic.calc_sector_index(stock_dfs, 
                                            f'{Paths.SECTOR_REDEFINITIONS_FOLDER}/topix1000.csv', 
                                            f'{Paths.SECTOR_REDEFINITIONS_FOLDER}/TOPIX1000_price.parquet')
    #print(sector_price_df.index.get_level_values('Sector').unique())
    print(sector_price_df)

    marketcap_df = sic.calc_marketcap(stock_dfs['price'], stock_dfs['fin'])
    sector_index = sic.calc_sector_index_by_dict(sector_stock_dict={'JPY+': ['2413', '3141', '4587', '1835', '4684'],
                                                                 'JPY-': ['7283', '7296', '5988', '8015', '7278']},
                                                                 stock_price_data=marketcap_df)
    
    print(sector_index)