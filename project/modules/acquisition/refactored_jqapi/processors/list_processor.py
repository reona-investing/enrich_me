import pandas as pd
from acquisition.refactored_jqapi.processors.base_processor import BaseProcessor
from acquisition.refactored_jqapi.schemas import StockListSchema
from acquisition.refactored_jqapi.utils import DataFormatter

class ListProcessor(BaseProcessor):
    """銘柄リスト処理クラス（修正版）"""
    
    def __init__(self, raw_path: str, processed_path: str):
        super().__init__(raw_path, processed_path)
        self.schema = StockListSchema
    
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """銘柄リストデータの処理"""
        print(f"Starting list data processing with {len(data)} records")
        print(f"Available columns: {list(data.columns)}")
        
        # 利用可能なkeyを特定
        available_keys = self.schema.get_available_keys(data)
        print(f"Available keys: {available_keys}")
        
        # カラム名のリネーム（利用可能なカラムのみ）
        rename_mapping = self.schema.get_available_rename_mapping(data)
        print(f"Rename mapping: {rename_mapping}")
        data = data.rename(columns=rename_mapping)
        
        # データ型変換
        data = self._format_dtypes(data)
        
        # 個別銘柄のみ抽出（ETF等を除外）
        data = self._extract_individual_stocks(data)
        
        # スキーマに基づく検証とフォーマット
        data = self.schema.validate_dataframe(data)
        
        print(f"List data processing completed with {len(data)} records")
        print(f"Final columns: {list(data.columns)}")
        return data
    
    def _format_dtypes(self, data: pd.DataFrame) -> pd.DataFrame:
        """データ型をフォーマット"""
        data = data.copy()
        
        # 銘柄コードのフォーマット
        if 'Code' in data.columns:
            data['Code'] = DataFormatter.format_stock_code(data['Code'])
        
        # 文字列カラムの変換
        str_columns = ['Code', 'CompanyName', 'MarketCodeName', 
                      'Sector33CodeName', 'Sector17CodeName', 'ScaleCategory']
        for col in str_columns:
            if col in data.columns:
                data[col] = data[col].astype(str)
        
        # 整数カラムの変換
        if 'Listing' in data.columns:
            data['Listing'] = data['Listing'].astype(int)
        
        return data
    
    def _extract_individual_stocks(self, data: pd.DataFrame) -> pd.DataFrame:
        """個別銘柄のみを抽出"""
        if 'Sector33CodeName' in data.columns:
            return data[
                data['Sector33CodeName'] != 'その他'
            ].drop_duplicates()
        return data.drop_duplicates()

if __name__ == '__main__':
    from utils.paths import Paths
    raw = Paths.RAW_STOCK_LIST_PARQUET
    processed = Paths.STOCK_LIST_PARQUET
    lp = ListProcessor(raw, processed)
    data = lp.process()
    print(data.tail(2))