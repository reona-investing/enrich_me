import pandas as pd

class FeatureDataMerger:
    """特徴量データの結合に特化したクラス"""
    
    def __init__(self):
        pass

    def merge_feature_data(self, existing_df: pd.DataFrame, new_scraped_df: pd.DataFrame) -> pd.DataFrame:
        """
        既存の特徴量データと新しくスクレイピングしたデータを結合
        
        Args:
            existing_df (pd.DataFrame): 既存のデータ
            new_scraped_df (pd.DataFrame): 新しくスクレイピングしたデータ
            
        Returns:
            pd.DataFrame: 結合・整形されたデータ
        """
        if existing_df.empty:
            return self._format_merged_df(new_scraped_df)
        
        if new_scraped_df.empty:
            return self._format_merged_df(existing_df)
        
        df = self._merge_data(existing_df=existing_df, additional_df=new_scraped_df)
        df = self._format_merged_df(df)
        return df

    def _merge_data(self, existing_df: pd.DataFrame, additional_df: pd.DataFrame) -> pd.DataFrame:
        """
        2つのデータフレームを結合
        """
        # 日付列をdatetimeに変換
        existing_df = existing_df.copy()
        additional_df = additional_df.copy()
        
        # 既存データの日付フォーマットを正規化
        if not existing_df.empty:
            existing_df['Date'] = pd.to_datetime(existing_df['Date'])
        
        # 新規データの日付フォーマットを正規化
        if not additional_df.empty:
            additional_df['Date'] = pd.to_datetime(additional_df['Date'])

        # 重複する日付のデータを既存データから除外（新しいデータを優先）
        if not existing_df.empty and not additional_df.empty:
            existing_df = existing_df.loc[~existing_df['Date'].isin(additional_df['Date'].unique())]
        
        return pd.concat([existing_df, additional_df], ignore_index=True)

    def _format_merged_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        結合されたデータフレームを整形
        """
        if df.empty:
            return df
        
        # 欠損値削除
        df = df.dropna(axis=0, how='all')
        df = df.dropna(axis=1, how='all')
        
        # 日付列の正規化
        df['Date'] = pd.to_datetime(df['Date'])
        
        # 日付でソート
        df = df.sort_values(by='Date').reset_index(drop=True)
        
        # 数値列の型変換（カンマ区切りの数値に対応）
        numeric_columns = ['Open', 'Close', 'High', 'Low']
        for col in numeric_columns:
            if col in df.columns:
                # 文字列の場合はカンマを除去
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace(',', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 重複削除（同じ日付の場合は最新のデータを保持）
        df = df.drop_duplicates(subset=['Date'], keep='last')
        
        return df

    def validate_data_integrity(self, df: pd.DataFrame) -> bool:
        """
        データの整合性を検証
        
        Args:
            df (pd.DataFrame): 検証対象のデータ
            
        Returns:
            bool: データが有効かどうか
        """
        if df.empty:
            return True
        
        # 必須列の存在確認
        required_columns = ['Date', 'Open', 'Close', 'High', 'Low']
        if not all(col in df.columns for col in required_columns):
            return False
        
        # 日付の重複確認
        if df['Date'].duplicated().any():
            return False
        
        # 価格データの妥当性確認
        numeric_columns = ['Open', 'Close', 'High', 'Low']
        for col in numeric_columns:
            if df[col].isna().all():
                return False
        
        # High >= Low の確認
        if (df['High'] < df['Low']).any():
            return False
        
        return True