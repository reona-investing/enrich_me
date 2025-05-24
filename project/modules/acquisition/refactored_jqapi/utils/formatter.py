import pandas as pd
from acquisition.refactored_jqapi.config import MergerConfig
from typing import List

class DataFormatter:
    """データフォーマッター"""
    
    @staticmethod
    def format_stock_code(code_series: pd.Series) -> pd.Series:
        """普通株の銘柄コードを4桁に変換"""
        return code_series.str[:4].where(
            (code_series.str.len() == 5) & (code_series.str[-1] == '0'), 
            code_series
        )
    
    @staticmethod
    def replace_stock_codes(code_series: pd.Series) -> pd.Series:
        """銘柄コードの置換"""
        return code_series.replace(MergerConfig.CODE_REPLACEMENT)
    
    @staticmethod
    def convert_to_datetime(date_series: pd.Series) -> pd.Series:
        """日付文字列をdatetimeに変換"""
        if date_series.dtype == 'object':
            return pd.to_datetime(date_series.astype(str).str[:10])
        return pd.to_datetime(date_series)
    
    @staticmethod
    def convert_numeric_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """数値カラムの変換"""
        df = df.copy()
        for col in columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df