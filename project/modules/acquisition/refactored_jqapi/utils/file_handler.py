import os
import pandas as pd
from typing import Optional, List

class FileHandler:
    """ファイル操作ユーティリティ"""
    
    @staticmethod
    def file_exists(path: str) -> bool:
        """ファイル存在チェック"""
        return os.path.isfile(path)
    
    @staticmethod
    def read_parquet(path: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Parquetファイルの読み込み"""
        if columns is None:
            return pd.read_parquet(path)
        return pd.read_parquet(path, columns=columns)
    
    @staticmethod
    def write_parquet(data: pd.DataFrame, path: str, verbose: bool = True) -> None:
        """Parquetファイルの書き込み"""
        # ディレクトリの作成
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        data.to_parquet(path, index=False)
        
        if verbose:
            print(f"データを保存しました: {path}")