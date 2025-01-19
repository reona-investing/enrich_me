from acquisition.jquants_api_operations.updater import ListUpdater, FinUpdater, PriceUpdater
from acquisition.jquants_api_operations.processor import ListProcessor, FinProcessor, PriceProcessor
from acquisition.jquants_api_operations.reader import Reader
from typing import List, Literal
import pandas as pd


class StockAcquisitionFacade:
    def __init__(self, update: bool = False, process: bool = False,
                 filter: str = None, filtered_code_list: str = None):
        """
        インスタンス生成時に株式データの一括読み込みを行います。
        引数:
            update (bool): True の場合、データを更新します。
            process (bool): True の場合、データを加工します。
            filter (str): 読み込み対象銘柄のフィルタリング条件（クエリ）
            filtered_code_list (list[str]): フィルタリングする銘柄コードのをリストで指定
        """
        if update:
            ListUpdater()
            FinUpdater()
            PriceUpdater()
        if process:
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

    def get_stock_data_dict(self) -> dict[pd.DataFrame]:
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
    filter = "(Listing==1)&((ScaleCategory=='TOPIX Core30')|(ScaleCategory=='TOPIX Large70')|(ScaleCategory=='TOPIX Mid400'))"
    from utils.flag_manager import FlagManager, Flags
    flag_manager = FlagManager()
    flag_manager.set_flag(flag=Flags.PROCESS_STOCK_PRICE, value=True)
    stock_df_dict = StockAcquisitionFacade(update=True, process=True, filter=filter).get_stock_data_dict()
    
    '''
    from utils import Paths
    data = []
    for i in range(2013, 2026):
        path = Paths.RAW_STOCK_PRICE_PARQUET.replace('0000', str(i))
        df = pd.read_parquet(path)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Code'] = df['Code'].astype(str)
        df = df[df['Code'] == '91010']
        data.append(df[['Date', 'Code', 'Close', 'Volume', 'TurnoverValue', 'AdjustmentFactor']])
    df = pd.concat(data, axis=0).sort_values('Date')
    df.to_csv('test_processed.csv')
    '''
    