import os
import pandas as pd
from datetime import datetime
from utils import Paths
from acquisition.jquants_api_operations.utils import FileHandler
from acquisition.jquants_api_operations.reader.reader_utils import filter_stocks

class Reader:
    def __init__(self,
                 filter: str = None, 
                 filtered_code_list: list[str] = None):
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
        return filter_stocks(df, df, self.filter, self.filtered_code_list)

    def read_fin(self,
                 path: str = Paths.STOCK_FIN_PARQUET, 
                 list_path: str = Paths.STOCK_LIST_PARQUET, 
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
        df = FileHandler.read_parquet(path)
        list_df = FileHandler.read_parquet(list_path)
        df = filter_stocks(df, list_df, self.filter, self.filtered_code_list)
        return df[df['Date']<=end_date]

    def read_price(self,
                   basic_path: str = Paths.STOCK_PRICE_PARQUET, 
                   list_path: str = Paths.STOCK_LIST_PARQUET, 
                   filter: str = None, 
                   filtered_code_list: list[str] = None,
                   end_date: datetime = datetime.today()) ->pd.DataFrame: # 価格情報の読み込み
        """
        価格情報を読み込み、調整を行います。
        Args:
            basic_path (str): 株価データのparquetファイルのパス
            list_path (str): 銘柄一覧のparquetファイルのパス（フィルタリング用）
            end_date (datetime): データの終了日
        Returns:
            pd.DataFrame: 価格情報
        """
        df = self._generate_price_df(basic_path, list_path, filter, filtered_code_list, end_date)
        return self._recalc_adjustment_factors(df)

    def _recalc_adjustment_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """全銘柄の最終日の累積調整係数が1となるように再計算します。"""
        adjustment_factors = df.groupby('Code')['CumulativeAdjustmentFactor'].transform('last')
        df[['Open', 'High', 'Low', 'Close', 'Volume']] *= adjustment_factors.values[:, None]
        return df

    def _generate_price_df(self, basic_path: str, list_path: str, filter: str, filtered_code_list: list[str], end_date: datetime) -> pd.DataFrame:
        """年次の株価データから、通期の価格データフレームを生成します。"""
        list_df = FileHandler.read_parquet(list_path)
        dfs = []
        for my_year in range(2013, end_date.year + 1):
            if os.path.exists(basic_path.replace('0000', str(my_year))):
                df = pd.read_parquet(Paths.STOCK_PRICE_PARQUET.replace('0000', str(my_year)))
                df = df[['Date', 'Code', 'Open', 'High', 'Low', 'Close', 'Volume',
                        'AdjustmentFactor', 'CumulativeAdjustmentFactor', 'TurnoverValue']]
                df = filter_stocks(df, list_df, filtered_code_list=filtered_code_list, filter=filter)
                dfs.append(df)
        df =  pd.concat(dfs, axis=0).drop_duplicates(subset=['Date', 'Code'], keep='last')
        return df[df['Date']<=end_date]


if __name__ == '__main__':
    universe_filter = "(Listing==1)&((ScaleCategory=='TOPIX Core30')|(ScaleCategory=='TOPIX Large70')|(ScaleCategory=='TOPIX Mid400'))"
    reader = Reader(filter = universe_filter)
    stock_list = reader.read_list()
    print(stock_list)
    stock_fin = reader.read_fin()
    print(stock_fin)
    stock_price = reader.read_price()
    print(stock_price)