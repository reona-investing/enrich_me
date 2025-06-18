from acquisition.jquants_api_operations.updater import ListUpdater, FinUpdater, PriceUpdater
from acquisition.jquants_api_operations.processor import ListProcessor, FinProcessor, PriceProcessor
from acquisition.jquants_api_operations.reader import Reader
from typing import Literal
import pandas as pd


class StockAcquisitionFacade:
    def __init__(self, update: bool = False, process: bool = False,
                 filter: str | None = None, filtered_code_list: list[str] | None = None):
        """
        インスタンス生成時に株式データの一括読み込みを行います。
        引数:
            update (bool): True の場合、データを更新します。
            process (bool): True の場合、データを加工します。
            filter (str): 読み込み対象銘柄のフィルタリング条件（クエリ）
            filtered_code_list (list[str]): フィルタリングする銘柄コードのをリストで指定
        """
        needs_processing = False
        if update:
            ListUpdater()
            FinUpdater()
            PriceUpdater()
            ListProcessor()
            FinProcessor()
            PriceProcessor()
        reader = Reader(filter=filter, filtered_code_list=filtered_code_list)
        self.list_df = reader.read_list()
        self.fin_df = reader.read_fin()
        self.price_df = reader.read_price()
    
    def get_stock_data(self, target: Literal['list', 'fin', 'price']) -> pd.DataFrame:
        '''
        targetに指定した文字列をもとに、適切なデータフレームを返します。
        Args:
            target (Literal['list', 'fin', 'price']): どのデータフレームを取得したいか
        Returns:
            pd.DataFrame: targetの文字列に合わせたデータフレーム
        '''
        if target == 'list':
            return self.list_df
        if target == 'fin':
            return self.fin_df
        if target == 'price':
            return self.price_df

    def get_stock_data_dict(self) -> dict[str, pd.DataFrame]:
        '''
        list, fin, priceを一つの辞書として返す。
        Returns:
            dict[pd.DataFrame]
        '''
        return {
            'list': self.list_df,
            'fin': self.fin_df,
            'price': self.price_df
            }

if __name__ == "__main__":
    # Example usage
    from datetime import datetime
    filter = "(Listing==1)&((ScaleCategory=='TOPIX Core30')|(ScaleCategory=='TOPIX Large70')|(ScaleCategory=='TOPIX Mid400'))"
    start = datetime.now()
    stock_df_dict = StockAcquisitionFacade(update=True, process=True, filter=filter).get_stock_data_dict()
    print(datetime.now() - start)