import pandas as pd
from acquisition.refactored_jqapi.updaters.base_updater import BaseUpdater
from utils.jquants_api_utils import cli

class ListUpdater(BaseUpdater):
    """銘柄リスト更新クラス"""
    
    def fetch_new_data(self, existing_data: pd.DataFrame) -> pd.DataFrame:
        """JQuants APIから銘柄リストを取得"""
        fetched_data = cli.get_list()
        fetched_data['Listing'] = 1
        return fetched_data
    
    def merge_data(self, existing_data: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
        """既存データと新規データをマージ"""
        if existing_data.empty:
            return new_data
        
        # 廃止銘柄の処理
        abolished_stocks = existing_data[
            ~existing_data['Code'].isin(new_data['Code'])
        ].copy()
        abolished_stocks['Listing'] = 0
        
        return pd.concat([new_data, abolished_stocks], axis=0)
    
    def format_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """データの整形"""
        return data.astype({
            'Code': str,
            'Sector17Code': str,
            'Sector33Code': str,
            'MarketCode': str,
            'MarginCode': str
        }).reset_index(drop=True)
    
if __name__ == '__main__':
    from utils.paths import Paths
    path = Paths.RAW_STOCK_LIST_PARQUET
    lu = ListUpdater(path)
    df = lu.update()
    print(df.tail(2))