import pandas as pd
from datetime import datetime
from acquisition.refactored_jqapi.updaters.base_updater import BaseUpdater
from utils.jquants_api_utils import cli

class FinUpdater(BaseUpdater):
    """財務データ更新クラス"""
    
    def fetch_new_data(self, existing_data: pd.DataFrame) -> pd.DataFrame:
        """JQuants APIから財務データを取得"""
        start_date = datetime(2000, 1, 1)
        
        if not existing_data.empty:
            # 複数の可能性のある日付カラム名をチェック
            date_columns = ['DisclosedDate', '日付', 'Date']
            date_col = None
            
            for col in date_columns:
                if col in existing_data.columns:
                    date_col = col
                    break
            
            if date_col is not None:
                # 日付の解析エラーを防ぐため、mixed formatを使用
                try:
                    existing_data[date_col] = pd.to_datetime(existing_data[date_col], format='mixed', errors='coerce')
                    start_date = existing_data[date_col].max()
                    if pd.isna(start_date):
                        start_date = datetime(2000, 1, 1)
                except Exception as e:
                    print(f"日付解析エラー: {e}. デフォルト開始日を使用します。")
                    start_date = datetime(2000, 1, 1)
        
        end_date = datetime.today()
        return cli.get_statements_range(start_dt=start_date, end_dt=end_date)
    
    def merge_data(self, existing_data: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
        """既存データと新規データをマージ"""
        if existing_data.empty:
            return new_data
        
        # 開示番号で重複を除去
        key_columns = ['DisclosureNumber', '開示番号']
        key_col = None
        
        for col in key_columns:
            if col in new_data.columns:
                key_col = col
                break
        
        if key_col is None or key_col not in existing_data.columns:
            # キーが見つからない場合は単純結合
            return pd.concat([existing_data, new_data], axis=0).reset_index(drop=True)
        
        unique_existing = existing_data[
            ~existing_data[key_col].isin(new_data[key_col])
        ]
        return pd.concat([new_data, unique_existing], axis=0).reset_index(drop=True)
    
    def format_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """データの整形"""
        # ソート用カラムを特定
        sort_columns = ['DisclosureNumber', '開示番号']
        sort_col = None
        
        for col in sort_columns:
            if col in data.columns:
                sort_col = col
                break
        
        if sort_col is None:
            sort_col = data.columns[0]  # フォールバック
        
        return data.astype(str).sort_values(sort_col).reset_index(drop=True)


if __name__ == '__main__':
    from utils.paths import Paths
    path = Paths.RAW_STOCK_FIN_PARQUET
    fu = FinUpdater(path)
    df = fu.update()
    print(df.tail(2))