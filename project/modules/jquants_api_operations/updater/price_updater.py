import pandas as pd
from datetime import datetime
import paths
import os
from jquants_api_operations.utils import to_parquet
from jquants_api_operations import cli
from FlagManager import FlagManager

def update_price(basic_path: str = paths.RAW_STOCK_PRICE_PARQUET) -> None:
    """
    価格情報の更新を行い、指定されたパスにParquet形式で保存する。
    """
    current_year = datetime.today().year
    prev_year = current_year - 1
    current_year_path = _generate_file_path(current_year, basic_path)
    prev_year_path = _generate_file_path(prev_year, basic_path)

    # 今年のデータを取得・更新
    raw_stock_price = _update_yearly_stock_price(current_year, current_year_path)

    # 今年のデータがない場合は前年データを取得・更新
    if not os.path.isfile(current_year_path):
        _update_yearly_stock_price(prev_year, prev_year_path)

    print(raw_stock_price.tail(2))


def _generate_file_path(year: int, basic_path: str) -> str:
    return basic_path.replace('0000', str(year))

def _update_yearly_stock_price(year: int, yearly_path: str) -> pd.DataFrame:
    """特定年の価格情報を取得・更新."""
    # ファイルが存在する場合は読み込み
    raw_stock_price = pd.read_parquet(yearly_path) if os.path.isfile(yearly_path) else None

    # データ取得・更新
    raw_stock_price = _update_yearly_price(year, raw_stock_price)
    raw_stock_price = raw_stock_price.drop_duplicates().dropna()

    # 保存
    to_parquet(raw_stock_price, yearly_path)
    return raw_stock_price

def _update_yearly_price(year: int, raw_stock_price: pd.DataFrame = None) -> pd.DataFrame:
    """API経由で価格情報データを取得するサブ関数."""
    if raw_stock_price is None:
        fetched_stock_price = _fetch_full_year_stock_price(year)
        _set_adjustment_flag(fetched_stock_price)
        return fetched_stock_price
    
    last_exist_date = pd.to_datetime(raw_stock_price["Date"]).iat[-1]
    if last_exist_date != datetime.today():
        fetched_stock_price = _fetch_new_stock_price(last_exist_date)
        _set_adjustment_flag(fetched_stock_price)
        raw_stock_price = _update_raw_stock_price(raw_stock_price, fetched_stock_price)
    
    return raw_stock_price

def _fetch_full_year_stock_price(year: int) -> pd.DataFrame:
    """指定された年の全期間の価格情報を取得."""
    return cli.get_price_range(
        start_dt=datetime(year, 1, 1), end_dt=datetime(year, 12, 31)
    )

def _fetch_new_stock_price(last_exist_date: datetime) -> pd.DataFrame:
    """最新の日付までの新しい価格情報を取得."""
    fetched_stock_price = cli.get_price_range(
        start_dt=last_exist_date, end_dt=datetime.today()
    )
    return fetched_stock_price

def _set_adjustment_flag(fetched_stock_price: pd.DataFrame):
    '''AdjustmentFactorが0でないものがあったときの条件チェック'''
    if any(fetched_stock_price['AdjustmentFactor'] != 1):
        FlgMng = FlagManager()
        FlgMng.flags['process_stock_price'] = True

def _update_raw_stock_price(raw_stock_price: pd.DataFrame, fetched_stock_price: pd.DataFrame) -> pd.DataFrame:
    """既存の価格情報に新しいデータを追加し、重複を削除."""
    raw_stock_price = pd.concat([raw_stock_price, fetched_stock_price], axis=0)
    raw_stock_price["Date"] = pd.to_datetime(raw_stock_price["Date"])
    return raw_stock_price[raw_stock_price['Code'].notnull()].drop_duplicates(subset=['Date', 'Code'])


if __name__ == '__main__':
    update_price()