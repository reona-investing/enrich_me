import pandas as pd
import paths
import os
from jquants_api_operations import cli
from jquants_api_operations.utils import to_parquet

def update_list(path: str = paths.RAW_STOCK_LIST_PARQUET) -> None:
    """
    銘柄一覧の更新を行い、指定されたパスにParquet形式で保存する。
    """
    stock_list_df = _fetch_data()
    file_exists = os.path.isfile(path)
    if file_exists:
        existing_stock_list_df = pd.read_parquet(path)
        stock_list_df = _merge(stock_list_df, existing_stock_list_df)
    stock_list_df = _format(stock_list_df)

    print(stock_list_df.tail(2))
    to_parquet(stock_list_df, path)
    

def _fetch_data() -> pd.DataFrame:
    fetched_stock_list_df = cli.get_list()
    fetched_stock_list_df['Listing'] = 1
    return fetched_stock_list_df

def _merge(fetched_stock_list_df: pd.DataFrame, existing_stock_list_df: pd.DataFrame) -> pd.DataFrame:
    abolished_stock_list_df = existing_stock_list_df.loc[~existing_stock_list_df['Code'].isin(fetched_stock_list_df['Code'])]
    abolished_stock_list_df['Listing'] = 0
    return pd.concat([fetched_stock_list_df, abolished_stock_list_df], axis=0)

def _format(stock_list_df: pd.DataFrame) -> pd.DataFrame:
    return stock_list_df.astype({
        'Code': str, 
        'Sector17Code': str, 
        'Sector33Code': str, 
        'MarketCode': str, 
        'MarginCode': str
        }).reset_index(drop=True)


if __name__ == '__main__':
    update_list()