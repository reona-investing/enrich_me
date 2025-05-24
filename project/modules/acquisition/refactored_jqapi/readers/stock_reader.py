import os
import pandas as pd
from datetime import datetime
from typing import Optional, List
from acquisition.refactored_jqapi.utils import FileHandler
from acquisition.refactored_jqapi.schemas import StockPriceSchema

class StockReader:
    """株式データ読み込みクラス"""
    
    def __init__(self, 
                 filter_query: Optional[str] = None,
                 code_list: Optional[List[str]] = None):
        """
        Args:
            filter_query: 銘柄フィルタリングクエリ
            code_list: 対象銘柄コードのリスト
        """
        self.filter_query = filter_query
        self.code_list = code_list
    
    def read_list(self, path: str) -> pd.DataFrame:
        """銘柄リストの読み込み"""
        if not FileHandler.file_exists(path):
            print(f"ファイルが見つかりません: {path}")
            return pd.DataFrame()
        
        data = FileHandler.read_parquet(path)
        return self._apply_filter(data, data, 'Code')
    
    def read_fin(self, path: str, list_path: str, 
                end_date: datetime = datetime.today()) -> pd.DataFrame:
        """財務データの読み込み"""
        if not FileHandler.file_exists(path):
            print(f"ファイルが見つかりません: {path}")
            return pd.DataFrame()
        
        data = FileHandler.read_parquet(path)
        
        # フィルタリング用のリストデータ
        if FileHandler.file_exists(list_path):
            list_data = FileHandler.read_parquet(list_path)
            # コードカラムを特定
            code_col = 'Code' if 'Code' in data.columns else 'LocalCode'
            data = self._apply_filter(data, list_data, code_col)
        
        # 日付フィルタリング
        date_columns = ['Date', 'DisclosedDate', '日付']
        for date_col in date_columns:
            if date_col in data.columns:
                try:
                    data[date_col] = pd.to_datetime(data[date_col], format='mixed', errors='coerce')
                    data = data[data[date_col] <= end_date]
                    break
                except Exception as e:
                    print(f"日付フィルタリングエラー {date_col}: {e}")
        
        return data
    
    def read_price(self, basic_path: str, list_path: str,
                  end_date: datetime = datetime.today()) -> pd.DataFrame:
        """株価データの読み込み"""
        # 年次ファイルの統合
        data = self._load_yearly_price_files(basic_path, end_date)
        
        if data.empty:
            return data
        
        # フィルタリング
        if FileHandler.file_exists(list_path):
            list_data = FileHandler.read_parquet(list_path)
            code_col = 'Code' if 'Code' in data.columns else '銘柄コード'
            data = self._apply_filter(data, list_data, code_col)
        
        # 日付フィルタリング
        date_col = 'Date' if 'Date' in data.columns else '日付'
        if date_col in data.columns:
            try:
                data[date_col] = pd.to_datetime(data[date_col], format='mixed', errors='coerce')
                data = data[data[date_col] <= end_date]
            except Exception as e:
                print(f"日付フィルタリングエラー: {e}")
        
        # 調整係数の再計算
        data = self._recalculate_adjustment_factors(data)
        
        return data
    
    def _load_yearly_price_files(self, basic_path: str, end_date: datetime) -> pd.DataFrame:
        """年次価格ファイルの読み込みと統合"""
        dfs = []
        
        for year in range(2013, end_date.year + 1):
            yearly_path = basic_path.replace('0000', str(year))
            if os.path.exists(yearly_path):
                try:
                    yearly_data = FileHandler.read_parquet(yearly_path)
                    
                    # 利用可能なカラムのみ選択
                    schema_cols = StockPriceSchema.get_column_names()
                    available_cols = [col for col in yearly_data.columns 
                                    if col in schema_cols or col in ['CumulativeAdjustmentFactor']]
                    
                    if available_cols:
                        yearly_data = yearly_data[available_cols]
                        dfs.append(yearly_data)
                except Exception as e:
                    print(f"年次ファイル読み込みエラー {yearly_path}: {e}")
        
        if not dfs:
            return pd.DataFrame()
        
        combined = pd.concat(dfs, axis=0)
        
        # 重複削除用のカラムを特定
        date_col = 'Date' if 'Date' in combined.columns else '日付'
        code_col = 'Code' if 'Code' in combined.columns else '銘柄コード'
        
        if date_col in combined.columns and code_col in combined.columns:
            return combined.drop_duplicates(subset=[date_col, code_col], keep='last')
        
        return combined.drop_duplicates()
    
    def _apply_filter(self, data: pd.DataFrame, list_data: pd.DataFrame, 
                     code_col: str) -> pd.DataFrame:
        """フィルタリングの適用"""
        if data.empty:
            return data
        
        try:
            if self.filter_query and not list_data.empty:
                # クエリベースのフィルタリング
                list_code_col = 'Code' if 'Code' in list_data.columns else code_col
                filtered_codes = list_data.query(self.filter_query)[list_code_col].astype(str).unique()
                return data[data[code_col].astype(str).isin(filtered_codes)]
            elif self.code_list:
                # コードリストベースのフィルタリング
                return data[data[code_col].astype(str).isin(self.code_list)]
        except Exception as e:
            print(f"フィルタリングエラー: {e}")
        
        return data
    
    def _recalculate_adjustment_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """調整係数の再計算（最終日=1になるように）"""
        if 'CumulativeAdjustmentFactor' not in data.columns:
            return data
        
        try:
            code_col = 'Code' if 'Code' in data.columns else '銘柄コード'
            
            # 各銘柄の最終日の累積調整係数を取得
            final_factors = data.groupby(code_col)['CumulativeAdjustmentFactor'].transform('last')
            
            # 価格・出来高カラムに適用
            price_volume_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in price_volume_cols:
                if col in data.columns:
                    data[col] *= final_factors
        except Exception as e:
            print(f"調整係数再計算エラー: {e}")
        
        return data