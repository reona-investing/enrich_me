import pandas as pd

class FeatureDataValidator:
    """特徴量データの整合性検証に特化したクラス"""
    
    def __init__(self):
        pass
    
    def validate_data_integrity(self, df: pd.DataFrame) -> tuple[bool, list[str]]:
        """
        データの整合性を検証
        
        Args:
            df (pd.DataFrame): 検証対象のデータ
            
        Returns:
            tuple[bool, list[str]]: (検証結果, エラーメッセージリスト)
        """
        errors = []
        
        if df.empty:
            return True, []
        
        # 必須列の存在確認
        required_columns = ['Date', 'Open', 'Close', 'High', 'Low']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"必須列が不足: {missing_columns}")
        
        # 日付の重複確認
        if df['Date'].duplicated().any():
            errors.append("日付に重複があります")
        
        # 価格データの妥当性確認
        numeric_columns = ['Open', 'Close', 'High', 'Low']
        for col in numeric_columns:
            if col in df.columns and df[col].isna().all():
                errors.append(f"価格データが全てNaN: {col}")
        
        # High >= Low の確認
        if 'High' in df.columns and 'Low' in df.columns:
            invalid_prices = df['High'] < df['Low']
            if invalid_prices.any():
                errors.append(f"High < Low の無効なデータが {invalid_prices.sum()} 件あります\n{df[df['High'] < df['Low']]}")
        
        return len(errors) == 0, errors
    
    def validate_date_continuity(self, df: pd.DataFrame, max_gap_days: int = 7) -> tuple[bool, list[str]]:
        """日付の連続性を検証"""
        errors = []
        
        if df.empty or len(df) < 2:
            return True, []
        
        df_sorted = df.sort_values('Date')
        date_diffs = df_sorted['Date'].diff().dt.days.dropna()
        large_gaps = date_diffs[date_diffs > max_gap_days]
        
        if not large_gaps.empty:
            errors.append(f"{max_gap_days}日以上のデータ欠損が {len(large_gaps)} 箇所あります")
        
        return len(errors) == 0, errors