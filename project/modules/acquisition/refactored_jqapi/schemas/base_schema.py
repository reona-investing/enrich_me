from dataclasses import dataclass
from typing import Dict, Type, Union, List, Optional
from datetime import datetime
import pandas as pd

@dataclass
class ColumnDefinition:
    """カラム定義クラス（key概念を復活）"""
    key: str  # 元のYAMLのkeyに対応
    raw_name: str  # 生データでのカラム名
    processed_name: str  # 処理後のカラム名
    dtype: Type
    required: bool = True
    description: str = ""
    plus_for_merger: bool = False  # 合併時に加算するカラム

class BaseSchema:
    """スキーマの基底クラス（key概念を復活）"""
    
    @classmethod
    def get_columns(cls) -> Dict[str, ColumnDefinition]:
        """定義されたカラム情報を取得（keyでアクセス）"""
        columns = {}
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if isinstance(attr, ColumnDefinition):
                columns[attr.key] = attr
        return columns
    
    @classmethod
    def get_raw_column_names(cls) -> List[str]:
        """生データのカラム名のリストを取得"""
        return [col.raw_name for col in cls.get_columns().values()]
    
    @classmethod
    def get_processed_column_names(cls) -> List[str]:
        """処理後のカラム名のリストを取得"""
        return [col.processed_name for col in cls.get_columns().values()]
    
    @classmethod
    def get_dtype_mapping(cls) -> Dict[str, Type]:
        """処理後カラム名でのデータ型マッピングを取得"""
        return {col.processed_name: col.dtype for col in cls.get_columns().values()}
    
    @classmethod
    def get_rename_mapping(cls) -> Dict[str, str]:
        """raw_name -> processed_nameのリネームマッピングを取得"""
        return {col.raw_name: col.processed_name for col in cls.get_columns().values()}
    
    @classmethod
    def get_available_keys(cls, data: pd.DataFrame) -> List[str]:
        """データフレーム内に存在するカラムのkeyリストを取得"""
        columns = cls.get_columns()
        available_keys = []
        
        for key, col_def in columns.items():
            if col_def.raw_name in data.columns:
                available_keys.append(key)
        
        return available_keys
    
    @classmethod
    def get_available_rename_mapping(cls, data: pd.DataFrame) -> Dict[str, str]:
        """利用可能なカラムのみのリネームマッピング"""
        columns = cls.get_columns()
        mapping = {}
        
        for key, col_def in columns.items():
            if col_def.raw_name in data.columns:
                mapping[col_def.raw_name] = col_def.processed_name
        
        return mapping
    
    @classmethod
    def get_merger_additive_columns(cls) -> List[str]:
        """合併時に加算するカラム名のリスト（処理後名）"""
        return [col.processed_name for col in cls.get_columns().values() 
                if col.plus_for_merger]
    
    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """データフレームの検証とフォーマット"""
        # 必要なカラムの存在確認
        columns = cls.get_columns()
        required_keys = [key for key, col in columns.items() if col.required]
        available_keys = cls.get_available_keys(df)
        missing_keys = set(required_keys) - set(available_keys)
        
        if missing_keys:
            print(f"Warning: Required columns missing for keys: {missing_keys}")
        
        # 利用可能なカラムのみを選択
        available_processed_cols = [columns[key].processed_name for key in available_keys]
        
        # データ型の変換
        dtype_mapping = cls.get_dtype_mapping()
        for col_name in available_processed_cols:
            if col_name in df.columns and col_name in dtype_mapping:
                dtype = dtype_mapping[col_name]
                try:
                    if dtype == datetime:
                        df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
                    elif dtype in [int, float]:
                        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
                    else:
                        df[col_name] = df[col_name].astype(dtype, errors='ignore')
                except Exception as e:
                    print(f"データ型変換エラー {col_name}: {e}")
        
        return df[available_processed_cols]