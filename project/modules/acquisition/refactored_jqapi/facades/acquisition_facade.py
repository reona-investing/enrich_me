import pandas as pd
from typing import Optional, List, Literal, Dict
from acquisition.refactored_jqapi.updaters import ListUpdater, FinUpdater, PriceUpdater
from acquisition.refactored_jqapi.processors import ListProcessor, FinProcessor, PriceProcessor
from acquisition.refactored_jqapi.readers import StockReader
from utils.paths import Paths

class StockAcquisitionFacade:
    """株式データ取得のファサードクラス"""
    
    def __init__(self,
                 update: bool = False,
                 process: bool = False,
                 filter_query: Optional[str] = None,
                 code_list: Optional[List[str]] = None):
        """
        Args:
            update: データ更新フラグ
            process: データ処理フラグ
            filter_query: 銘柄フィルタリングクエリ
            code_list: 対象銘柄コードのリスト
        """
        # データ更新
        if update:
            try:
                self._update_data()
            except Exception as e:
                print(f"データ更新エラー: {e}")
        
        # データ処理
        if process:
            try:
                self._process_data()
            except Exception as e:
                print(f"データ処理エラー: {e}")
        
        # データ読み込み
        try:
            reader = StockReader(filter_query=filter_query, code_list=code_list)
            self.list_df = reader.read_list(Paths.STOCK_LIST_PARQUET)
            self.fin_df = reader.read_fin(Paths.STOCK_FIN_PARQUET, Paths.STOCK_LIST_PARQUET)
            self.price_df = reader.read_price(Paths.STOCK_PRICE_PARQUET, Paths.STOCK_LIST_PARQUET)
            
            print(f"読み込み完了:")
            print(f"  銘柄リスト: {len(self.list_df)} 件")
            print(f"  財務データ: {len(self.fin_df)} 件")
            print(f"  株価データ: {len(self.price_df)} 件")
            
        except Exception as e:
            print(f"データ読み込みエラー: {e}")
            self.list_df = pd.DataFrame()
            self.fin_df = pd.DataFrame()
            self.price_df = pd.DataFrame()
    
    def _update_data(self) -> None:
        """データの更新"""
        try:
            print("銘柄リストを更新中...")
            ListUpdater(Paths.RAW_STOCK_LIST_PARQUET).update()
        except Exception as e:
            print(f"銘柄リスト更新エラー: {e}")
        
        try:
            print("財務データを更新中...")
            FinUpdater(Paths.RAW_STOCK_FIN_PARQUET).update()
        except Exception as e:
            print(f"財務データ更新エラー: {e}")
        
        try:
            print("株価データを更新中...")
            PriceUpdater(Paths.RAW_STOCK_PRICE_PARQUET).update()
        except Exception as e:
            print(f"株価データ更新エラー: {e}")
    
    def _process_data(self) -> None:
        """データの処理"""
        try:
            print("銘柄リストを処理中...")
            ListProcessor(Paths.RAW_STOCK_LIST_PARQUET, Paths.STOCK_LIST_PARQUET).process()
        except Exception as e:
            print(f"銘柄リスト処理エラー: {e}")
        
        try:
            print("財務データを処理中...")
            FinProcessor(Paths.RAW_STOCK_FIN_PARQUET, Paths.STOCK_FIN_PARQUET).process()
        except Exception as e:
            print(f"財務データ処理エラー: {e}")
        
        try:
            print("株価データを処理中...")
            PriceProcessor(Paths.RAW_STOCK_PRICE_PARQUET, Paths.STOCK_PRICE_PARQUET).process()
        except Exception as e:
            print(f"株価データ処理エラー: {e}")
    
    def get_stock_data(self, target: Literal['list', 'fin', 'price']) -> pd.DataFrame:
        """指定したデータの取得"""
        if target == 'list':
            return self.list_df
        elif target == 'fin':
            return self.fin_df
        elif target == 'price':
            return self.price_df
        else:
            raise ValueError(f"Invalid target: {target}. Must be 'list', 'fin', or 'price'")
    
    def get_stock_data_dict(self) -> Dict[str, pd.DataFrame]:
        """全データを辞書形式で取得"""
        return {
            'list': self.list_df,
            'fin': self.fin_df,
            'price': self.price_df
        }

if __name__ == "__main__":
    saf = StockAcquisitionFacade(update=True, process=True)
    stock_data = saf.get_stock_data_dict()
    print(stock_data['list'].head())
    print(stock_data['fin'].head())
    print(stock_data['price'].head())