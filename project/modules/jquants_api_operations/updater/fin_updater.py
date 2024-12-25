import pandas as pd
import paths
from datetime import datetime
from jquants_api_operations import cli
from jquants_api_operations.utils import to_parquet

def update_fin(path: str = paths.RAW_STOCK_FIN_PARQUET) -> None:
    """
    財務情報の更新を行い、指定されたパスにParquet形式で保存する。
    """
    raw_stock_fin_df = pd.read_parquet(path)
    fetched_stock_fin_df = _fetch_data(raw_stock_fin_df)
    raw_stock_fin_df = _merge(raw_stock_fin_df, fetched_stock_fin_df, key="DisclosureNumber")
    raw_stock_fin_df = _format(raw_stock_fin_df)

    print(raw_stock_fin_df.tail(2))
    to_parquet(raw_stock_fin_df, path)


def _fetch_data(raw_stock_fin_df) -> pd.DataFrame:
    start_date = datetime(2000, 1, 1)
    if raw_stock_fin_df is not None:
        start_date = pd.to_datetime(raw_stock_fin_df["DisclosedDate"].iat[-1])
    end_date = datetime.today()
    fetched_stock_fin_df = cli.get_statements_range(start_dt=start_date, end_dt=end_date)
    return fetched_stock_fin_df

def _merge(existing_data: pd.DataFrame, new_data: pd.DataFrame, key: str) -> pd.DataFrame:
    """既存データと新規データを結合し、重複を排除."""
    if existing_data is None:
        return new_data
    unique_existing = existing_data.loc[~existing_data[key].isin(new_data[key])]
    merged_data = pd.concat([new_data, unique_existing], axis=0).reset_index(drop=True)
    return merged_data

def _format(raw_stock_fin_df: pd.DataFrame) -> pd.DataFrame:
    return raw_stock_fin_df.astype(str).sort_values('DisclosureNumber').reset_index(drop=True)


if __name__ == '__main__':
    update_fin()