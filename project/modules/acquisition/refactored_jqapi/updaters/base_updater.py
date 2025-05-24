from abc import ABC, abstractmethod
import pandas as pd
from acquisition.refactored_jqapi.utils import FileHandler

class BaseUpdater(ABC):
    """データ更新の基底クラス"""
    
    def __init__(self, path: str):
        self.path = path
        
    def update(self) -> pd.DataFrame:
        """データ更新のメインフロー"""
        print(f"Updating data at {self.path}")
        
        # 既存データの読み込み
        existing_data = self.load_existing_data()
        
        # 新しいデータの取得
        new_data = self.fetch_new_data(existing_data)
        
        # データのマージ
        merged_data = self.merge_data(existing_data, new_data)
        
        # データの整形
        formatted_data = self.format_data(merged_data)
        
        # データの保存
        self.save_data(formatted_data)
        
        print(f"Updated data saved. Total records: {len(formatted_data)}")
        return formatted_data
    
    def load_existing_data(self) -> pd.DataFrame:
        """既存データの読み込み"""
        if FileHandler.file_exists(self.path):
            return FileHandler.read_parquet(self.path)
        return pd.DataFrame()
    
    @abstractmethod
    def fetch_new_data(self, existing_data: pd.DataFrame) -> pd.DataFrame:
        """新しいデータの取得（サブクラスで実装）"""
        pass
    
    @abstractmethod
    def merge_data(self, existing_data: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
        """データのマージ（サブクラスで実装）"""
        pass
    
    @abstractmethod
    def format_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """データの整形（サブクラスで実装）"""
        pass
    
    def save_data(self, data: pd.DataFrame) -> None:
        """データの保存"""
        FileHandler.write_parquet(data, self.path)