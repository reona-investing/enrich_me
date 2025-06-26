import pandas as pd
from utils.paths import Paths
from utils.jquants_api_utils import cli
from acquisition.jquants_api_operations.utils import FileHandler

class ListUpdater:
    def __init__(self,
                 path: str = Paths.RAW_STOCK_LIST_PARQUET
                 ) -> None:
        """
        銘柄一覧の更新を行い、指定されたパスにParquet形式で保存する。
        
        :param path: Parquetファイルの保存先パス
        """
        stock_list_df = self._fetch()

        if FileHandler.file_exists(path):
            existing_stock_list_df = FileHandler.read_parquet(path)
            stock_list_df = self._merge(stock_list_df, existing_stock_list_df)

        stock_list_df = self._format(stock_list_df)
        FileHandler.write_parquet(stock_list_df, path)


    def _fetch(self) -> pd.DataFrame:
        """JQuants APIからデータを取得する"""
        fetched_stock_list_df = cli.get_list()
        fetched_stock_list_df['Listing'] = 1
        return fetched_stock_list_df

    def _merge(self, fetched_stock_list_df: pd.DataFrame, existing_stock_list_df: pd.DataFrame) -> pd.DataFrame:
        """
        取得したデータと既存のデータをマージする。
        
        :param fetched_stock_list_df: 取得した銘柄一覧
        :param existing_stock_list_df: 既存の銘柄一覧
        :return: マージ後の銘柄一覧
        """
        abolished_stock_list_df = existing_stock_list_df.loc[
            ~existing_stock_list_df['Code'].isin(fetched_stock_list_df['Code'])
        ]
        abolished_stock_list_df['Listing'] = 0
        return pd.concat([fetched_stock_list_df, abolished_stock_list_df], axis=0)

    def _format(self, stock_list_df: pd.DataFrame) -> pd.DataFrame:
        """
        データフレームを指定された形式に整形する。
        
        :param stock_list_df: 整形対象のデータフレーム
        :return: 整形後のデータフレーム
        """
        return stock_list_df.astype({
            'Code': str, 
            'Sector17Code': str, 
            'Sector33Code': str, 
            'MarketCode': str, 
            'MarginCode': str
        }).reset_index(drop=True)


if __name__ == '__main__':
    ListUpdater()