#%% モジュールのインポート
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import jquantsapi
from dotenv import load_dotenv
from IPython.display import display

import paths
import FlagManager

#%% 環境変数の読み込み
load_dotenv(f'{paths.MODULES_FOLDER}/.env')

#%% J-Quants APIクライアントの初期化
cli = jquantsapi.Client(
    mail_address=os.getenv('JQUANTS_EMAIL'),
    password=os.getenv('JQUANTS_PASS')
)

#%% ユーティリティ関数
def _merge_new_data(existing_data: pd.DataFrame, new_data: pd.DataFrame, key: str) -> pd.DataFrame:
    """既存データと新規データを結合し、重複を排除."""
    if existing_data is None:
        return new_data
    unique_existing = existing_data.loc[~existing_data[key].isin(new_data[key])]
    merged_data = pd.concat([new_data, unique_existing], axis=0).reset_index(drop=True)
    return merged_data

def _save_to_parquet(data: pd.DataFrame, path: str) -> None:
    """データを Parquet ファイルに保存."""
    data.to_parquet(path)
    print(f"データを保存しました: {path}")


#%% 関数群
def update_stock_fin() -> None:
    """財務情報の更新."""
    raw_stock_fin = pd.read_parquet(paths.RAW_STOCK_FIN_PARQUET)
    start_date = pd.to_datetime(raw_stock_fin["DisclosedDate"].iat[-1])
    end_date = datetime.today()

    fetched_stock_fin = cli.get_statements_range(start_dt=start_date, end_dt=end_date)
    raw_stock_fin = _merge_new_data(raw_stock_fin, fetched_stock_fin, key="DisclosureNumber")
    raw_stock_fin = raw_stock_fin.astype(str).sort_values('DisclosureNumber').reset_index(drop=True)

    display(raw_stock_fin.tail(2))
    _save_to_parquet(raw_stock_fin, paths.RAW_STOCK_FIN_PARQUET)


def update_stock_list() -> None:
    """銘柄一覧の更新."""
    raw_stock_list = pd.read_parquet(paths.RAW_STOCK_LIST_PARQUET)
    fetched_stock_list = cli.get_list()
    fetched_stock_list['Listing'] = 1

    if raw_stock_list is not None:
        abolished_stock_list = raw_stock_list.loc[~raw_stock_list['Code'].isin(fetched_stock_list['Code'])]
        abolished_stock_list['Listing'] = 0
        fetched_stock_list = pd.concat([fetched_stock_list, abolished_stock_list], axis=0)

    fetched_stock_list = fetched_stock_list.astype({
        'Code': str, 'Sector17Code': str, 'Sector33Code': str, 'MarketCode': str, 'MarginCode': str
    }).reset_index(drop=True)

    display(fetched_stock_list.tail(2))
    _save_to_parquet(fetched_stock_list, paths.RAW_STOCK_LIST_PARQUET)


def update_stock_price() -> None:
    """価格情報の更新."""
    end_date = datetime.today()
    current_year_path = paths.RAW_STOCK_PRICE_PARQUET.replace('0000', str(end_date.year))
    prev_year_path = paths.RAW_STOCK_PRICE_PARQUET.replace('0000', str(end_date.year - 1))

    # 今年のデータを取得・更新
    raw_stock_price = _update_yearly_stock_price(end_date.year, current_year_path)

    # 今年のデータがない場合は前年データを取得・更新
    if not os.path.exists(current_year_path):
        _update_yearly_stock_price(end_date.year - 1, prev_year_path)

    display(raw_stock_price.tail(2))


def _update_yearly_stock_price(year: int, file_path: str) -> pd.DataFrame:
    """特定年の価格情報を取得・更新."""
    # ファイルが存在する場合は読み込み
    raw_stock_price = pd.read_parquet(file_path) if os.path.exists(file_path) else None

    # データ取得・更新
    raw_stock_price = _fetch_stock_price_from_API(year, raw_stock_price)
    raw_stock_price = raw_stock_price.drop_duplicates().dropna()

    # 保存
    _save_to_parquet(raw_stock_price, file_path)
    return raw_stock_price


def _fetch_stock_price_from_API(year: int, raw_stock_price: pd.DataFrame = None) -> pd.DataFrame:
    """API経由で価格情報データを取得するサブ関数."""
    if raw_stock_price is None:
        raw_stock_price = cli.get_price_range(
            start_dt=datetime(year, 1, 1), end_dt=datetime(year, 12, 31)
        )
        return raw_stock_price
    
    last_exist_date = pd.to_datetime(raw_stock_price["Date"]).iat[-1]
    if last_exist_date != datetime.today():
        fetched_stock_price = cli.get_price_range(
            start_dt=last_exist_date, end_dt=datetime.today()
        )
        # AdjustmentFactorの条件チェック
        if any(fetched_stock_price['AdjustmentFactor'] != 1):
            FlgMng = FlagManager.FlagManager()
            FlgMng.flags['process_stock_price'] = True

        raw_stock_price = pd.concat([raw_stock_price, fetched_stock_price], axis=0)
        raw_stock_price["Date"] = pd.to_datetime(raw_stock_price["Date"])
        raw_stock_price = raw_stock_price[raw_stock_price['Code'].notnull()].drop_duplicates(subset=['Date', 'Code'])
    return raw_stock_price


def get_next_open_date(latest_date: datetime) -> datetime:
    """翌開場日の取得."""
    from_yyyymmdd = datetime.today().strftime('%Y%m%d')
    to_yyyymmdd = (datetime.today() + relativedelta(months=1)).strftime('%Y%m%d')

    market_open_date = cli.get_markets_trading_calendar(
        holiday_division="1", from_yyyymmdd=from_yyyymmdd, to_yyyymmdd=to_yyyymmdd
    )
    market_open_date['Date'] = pd.to_datetime(market_open_date['Date'])
    next_open_date = market_open_date.loc[market_open_date['Date'] > latest_date, 'Date'].iat[0]
    return next_open_date


def update_stock_dfs(): # 全データを一括取得
    print('J-Quants APIでデータを取得します。')
    update_stock_list()
    print('銘柄リストの取得が完了しました。')
    update_stock_fin()
    print('財務情報の取得が完了しました。')
    update_stock_price()
    print('価格情報の取得が完了しました。')
    print('J-Quants APIでのデータ取得が全て完了しました。')
    print('----------------------------------------------')


#%% デバッグ
if __name__ == '__main__':
    update_stock_dfs()