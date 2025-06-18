import os
import pandas as pd
from datetime import datetime
from utils.paths import Paths
from utils.yaml_utils import ColumnConfigsGetter
from acquisition.jquants_api_operations.utils import FileHandler
from acquisition.jquants_api_operations.reader.reader_utils import filter_stocks

class Reader:
    def __init__(self,
                 filter: str | None = None, 
                 filtered_code_list: list[str] | None = None):
        """
        抽出条件を指定します。filterとfiltered_code_listを両方設定した場合、filterの条件が優先されます。
        Args:
            filter (str): 読み込み対象銘柄のフィルタリング条件（クエリ）
            filtered_code_list (list[str]): フィルタリングする銘柄コードのをリストで指定
        """
        self.filter = filter
        self.filtered_code_list = filtered_code_list

    def read_list(self, path: str = Paths.STOCK_LIST_PARQUET) -> pd.DataFrame:
        """
        銘柄一覧を読み込みます。filterとfiltered_code_listを両方設定した場合、filterの条件が優先されます。
        Args:
            path (str): 銘柄一覧のparquetファイルのパス
        Returns:
            pd.DataFrame: 銘柄一覧
        """
        df = FileHandler.read_parquet(path)
        return filter_stocks(df, df, self.filter, self.filtered_code_list, 'Code')

    def read_fin(self,
                 path: str = Paths.STOCK_FIN_PARQUET, 
                 list_path: str = Paths.STOCK_LIST_PARQUET,
                 cols_yaml_path: str = Paths.STOCK_FIN_COLUMNS_YAML,
                 end_date: datetime = datetime.today()) -> pd.DataFrame:
        """
        財務情報データを読み込みます。filterとfiltered_code_listを両方設定した場合、filterの条件が優先されます。
        Args:
            path (str): 財務情報のparquetファイルのパス
            list_path (str): 銘柄一覧のparquetファイルのパス（フィルタリング用）
            end_date (datetime): データの終了日
        Returns:
            pd.DataFrame: 財務情報
        """
        ccg = ColumnConfigsGetter(cols_yaml_path = cols_yaml_path)
        date_col = ccg.get_column_name('日付')
        code_col = ccg.get_column_name('銘柄コード')

        df = FileHandler.read_parquet(path)
        list_df = FileHandler.read_parquet(list_path)
        df = filter_stocks(df, list_df, self.filter, self.filtered_code_list, code_col)

        return df[df[date_col]<=end_date]

    def read_price(self,
                   price_path: str = f"H:\マイドライブ\enrich_me\project/rawdata_JQuantsAPI/stock_price.parquet",
                   list_path: str = Paths.STOCK_LIST_PARQUET,
                   cols_yaml_path: str = Paths.STOCK_PRICE_COLUMNS_YAML,
                   end_date: datetime = datetime.today()) ->pd.DataFrame: # 価格情報の読み込み
        """
        価格情報を読み込み、調整を行います。
        Args:
            price_path (str): 株価データのparquetファイルのパス
            list_path (str): 銘柄一覧のparquetファイルのパス（フィルタリング用）
            end_date (datetime): データの終了日
        Returns:
            pd.DataFrame: 価格情報
        """
        ccg = ColumnConfigsGetter(cols_yaml_path = cols_yaml_path)
        date_col = ccg.get_column_name('日付')
        code_col = ccg.get_column_name('銘柄コード')
        cols = ccg.get_column_names(['日付', '銘柄コード', '始値', '高値', '安値', '終値', '取引高', '調整係数', '取引代金'])
        if cols is None:
            cols = []
        cols.append('CumulativeAdjustmentFactor')
        
        list_df = FileHandler.read_parquet(list_path)
        df = pd.read_parquet(price_path)
        df = df[cols].copy()
        df = filter_stocks(df, list_df, filtered_code_list=self.filtered_code_list, filter=self.filter, code_col=code_col)
        return self._recalc_adjustment_factors(df, cols_yaml_path)
    
    
    def _recalc_adjustment_factors(self, df: pd.DataFrame, cols_yaml_path: str) -> pd.DataFrame:
        """全銘柄の最終日の累積調整係数が1となるように再計算します。"""
        ccg = ColumnConfigsGetter(cols_yaml_path = cols_yaml_path)
        code_col = ccg.get_column_name('銘柄コード')
        cols = ccg.get_column_names(['始値', '高値', '安値', '終値', '取引高'])

        adjustment_factors = df.groupby(code_col)['CumulativeAdjustmentFactor'].transform('last')
        df[cols] *= adjustment_factors.values[:, None]
        return df


if __name__ == '__main__':
    universe_filter = "(Listing==1)&((ScaleCategory=='TOPIX Core30')|(ScaleCategory=='TOPIX Large70')|(ScaleCategory=='TOPIX Mid400'))"
    reader = Reader(filter = universe_filter)
    stock_list = reader.read_list()
    print(stock_list)
    stock_fin = reader.read_fin()
    print(stock_fin)
    stock_price = reader.read_price()
    print(stock_price)