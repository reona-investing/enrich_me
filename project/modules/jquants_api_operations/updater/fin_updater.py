import pandas as pd
from datetime import datetime
import paths
from jquants_api_operations import cli
from jquants_api_operations.utils import FileHandler

def update_fin(
    path: str = paths.RAW_STOCK_FIN_PARQUET
) -> None:
    """
    財務情報の更新を行い、指定されたパスにParquet形式で保存する。
    :param path: Parquetファイルの保存先パス
    """
    existing_data = _load_existing_data(path)
    fetched_data = _fetch_data(existing_data)
    merged_data = _merge(existing_data, fetched_data, key="DisclosureNumber")
    formatted_data = _format(merged_data)
    
    print(formatted_data.tail(2))
    FileHandler.write_parquet(formatted_data, path)


def _load_existing_data(path: str) -> pd.DataFrame:
    """既存データを読み込む。ファイルが存在しない場合は空のDataFrameを返す。"""
    if FileHandler.file_exists(path):
        return FileHandler.read_parquet(path)
    return pd.DataFrame()


def _fetch_data(existing_data: pd.DataFrame) -> pd.DataFrame:
    """財務情報をAPIから取得する。"""
    start_date = datetime(2000, 1, 1)
    if not existing_data.empty:
        start_date = pd.to_datetime(existing_data["DisclosedDate"].iat[-1])
    end_date = datetime.today()
    return cli.get_statements_range(start_dt=start_date, end_dt=end_date)


def _merge(existing_data: pd.DataFrame, new_data: pd.DataFrame, key: str) -> pd.DataFrame:
    """既存データと新規データを結合し、重複を排除する。"""
    if existing_data.empty:
        return new_data
    unique_existing = existing_data.loc[~existing_data[key].isin(new_data[key])]
    return pd.concat([new_data, unique_existing], axis=0).reset_index(drop=True)


def _format(data: pd.DataFrame) -> pd.DataFrame:
    """データを指定された形式に整形する。"""
    return data.astype(str).sort_values('DisclosureNumber').reset_index(drop=True)


if __name__ == '__main__':
    update_fin()
