import pandas as pd
from typing import Protocol
import os
import paths
from jquants_api_operations import cli
from jquants_api_operations.utils import FileHandler

def update_list(
    path: str = paths.RAW_STOCK_LIST_PARQUET,
    file_handler: FileHandler = FileHandler()
) -> None:
    """
    銘柄一覧の更新を行い、指定されたパスにParquet形式で保存する。
    
    :param path: Parquetファイルの保存先パス
    :param fetcher: データ取得用オブジェクト
    :param file_handler: ファイル操作用オブジェクト
    """
    stock_list_df = _fetch()

    if file_handler.file_exists(path):
        existing_stock_list_df = file_handler.read_parquet(path)
        stock_list_df = _merge(stock_list_df, existing_stock_list_df)

    stock_list_df = _format(stock_list_df)
    print(stock_list_df.tail(2))
    file_handler.write_parquet(stock_list_df, path)


def _fetch() -> pd.DataFrame:
    """JQuants APIからデータを取得する"""
    fetched_stock_list_df = cli.get_list()
    fetched_stock_list_df['Listing'] = 1
    return fetched_stock_list_df

def _merge(fetched_stock_list_df: pd.DataFrame, existing_stock_list_df: pd.DataFrame) -> pd.DataFrame:
    """
    取得したデータと既存のデータをマージする。
    
    :param fetched_stock_list_df: 取得した銘柄一覧
    :param existing_stock_list_df: 既存の銘柄一覧
    :return: マージ後の銘柄一覧
    """
    abolished_stock_list_df = existing_stock_list_df.loc[
        ~existing_stock_list_df['Code'].isin(fetched_stock_list_df['Code'])
    ]
    abolished_stock_list_df['Listing'] = 0
    return pd.concat([fetched_stock_list_df, abolished_stock_list_df], axis=0)

def _format(stock_list_df: pd.DataFrame) -> pd.DataFrame:
    """
    データフレームを指定された形式に整形する。
    
    :param stock_list_df: 整形対象のデータフレーム
    :return: 整形後のデータフレーム
    """
    return stock_list_df.astype({
        'Code': str, 
        'Sector17Code': str, 
        'Sector33Code': str, 
        'MarketCode': str, 
        'MarginCode': str
    }).reset_index(drop=True)


if __name__ == '__main__':
    update_list()