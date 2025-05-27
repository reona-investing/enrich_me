import pandas as pd

class FeatureDataMerger:
    """特徴量データの結合に特化したクラス"""
    
    def __init__(self):
        pass

    def merge_feature_data(self, existing_df: pd.DataFrame, new_scraped_df: pd.DataFrame) -> pd.DataFrame:
        """
        既存の特徴量データと新しくスクレイピングしたデータを結合
        """
        if existing_df.empty:
            return self._format_merged_df(new_scraped_df)
        
        if new_scraped_df.empty:
            return self._format_merged_df(existing_df)
        
        df = self._merge_data(existing_df=existing_df, additional_df=new_scraped_df)
        df = self._format_merged_df(df)
        return df

    def _merge_data(self, existing_df: pd.DataFrame, additional_df: pd.DataFrame) -> pd.DataFrame:
        """2つのデータフレームを結合"""
        # 既存の実装をそのまま使用
        existing_df = existing_df.copy()
        additional_df = additional_df.copy()
        
        if not existing_df.empty:
            existing_df['Date'] = pd.to_datetime(existing_df['Date'])
        
        if not additional_df.empty:
            additional_df['Date'] = pd.to_datetime(additional_df['Date'])

        if not existing_df.empty and not additional_df.empty:
            existing_df = existing_df.loc[~existing_df['Date'].isin(additional_df['Date'].unique())]
        
        return pd.concat([existing_df, additional_df], ignore_index=True)

    def _format_merged_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """結合されたデータフレームを整形"""
        # 既存の実装をそのまま使用
        if df.empty:
            return df
        
        df = df.dropna(axis=0, how='all')
        df = df.dropna(axis=1, how='all')
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date').reset_index(drop=True)
        
        numeric_columns = ['Open', 'Close', 'High', 'Low']
        for col in numeric_columns:
            if col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace(',', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.drop_duplicates(subset=['Date'], keep='last')
        return df