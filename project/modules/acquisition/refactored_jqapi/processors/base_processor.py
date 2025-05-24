from abc import ABC, abstractmethod
import pandas as pd
from acquisition.refactored_jqapi.utils import FileHandler

class BaseProcessor(ABC):
    """データ処理の基底クラス"""
    
    def __init__(self, raw_path: str, processed_path: str):
        self.raw_path = raw_path
        self.processed_path = processed_path
        
    def process(self) -> pd.DataFrame:
        """データ処理のメインフロー"""
        print(f"Processing data from {self.raw_path}")
        
        # データ読み込み
        raw_data = self.load_data()
        
        # データ処理
        processed_data = self.process_data(raw_data)
        
        # データ保存
        self.save_data(processed_data)
        
        print(f"Data saved to {self.processed_path}")
        print(f"Processed {len(processed_data)} records")
        
        return processed_data
    
    def load_data(self) -> pd.DataFrame:
        """データ読み込み"""
        return FileHandler.read_parquet(self.raw_path)
    
    @abstractmethod
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """データ処理（サブクラスで実装）"""
        pass
    
    def save_data(self, data: pd.DataFrame) -> None:
        """データ保存"""
        FileHandler.write_parquet(data, self.processed_path)
    
    def get_description_based_mapping(self, raw_schema_class, processed_schema_class):
        """説明文ベースのカラムマッピングを取得"""
        return raw_schema_class.find_matching_columns_by_description(raw_schema_class, processed_schema_class)