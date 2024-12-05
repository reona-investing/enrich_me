#%% モジュールのインポート
#自作モジュール
import paths
import FlagManager
import data_pickler
#既存モジュール
import pandas as pd
import jquantsapi
from datetime import datetime
from dateutil.relativedelta import relativedelta
from IPython.display import display

import os
from dotenv import load_dotenv
load_dotenv(f'{paths.MODULES_FOLDER}/.env')

#%% J-quants APIのクライアントを立てる（インポート時）
# リフレッシュトークンの取得。
cli = jquantsapi.Client(mail_address = os.getenv('JQUANTS_EMAIL'),
                        password = os.getenv('JQUANTS_PASS'))

#%% 関数群
def update_stock_fin(cli:object = cli): # 財務情報の更新
    #csvの読み込み
    raw_stock_fin = pd.read_parquet(paths.RAW_STOCK_FIN_PARQUET)
    #データの取得範囲
    start_date = pd.to_datetime(raw_stock_fin["DisclosedDate"].iat[-1])
    end_date = datetime.today()
    #データ取得
    fetched_stock_fin: pd.DataFrame = cli.get_statements_range(start_dt=start_date, end_dt=end_date)
    #既存ファイルとの結合
    if raw_stock_fin is None:
        raw_stock_fin = fetched_stock_fin.copy() #raw_stock_finがない場合、取得したdfをそのまま採用
    else:
        existing_stock_fin = raw_stock_fin.loc[~raw_stock_fin['DisclosureNumber'].isin(fetched_stock_fin['DisclosureNumber'].unique()), :] #新規取得以外の情報を既存のdfから抽出
        raw_stock_fin = pd.concat([fetched_stock_fin, existing_stock_fin], axis=0) #新旧のデータフレームを縦に連結
    #並べ替え
    raw_stock_fin = raw_stock_fin.astype(str)
    raw_stock_fin = raw_stock_fin.sort_values('DisclosureNumber').reset_index(drop=True)
    #データ取得の確認
    display(raw_stock_fin.tail(2))
    #データの保存

    raw_stock_fin.to_parquet(paths.RAW_STOCK_FIN_PARQUET)

def update_stock_list(cli:object = cli): # 銘柄一覧の更新
    #csvの読み込み
    raw_stock_list = pd.read_parquet(paths.RAW_STOCK_LIST_PARQUET)
    #データ取得
    fetched_stock_list = cli.get_list() #現時点の情報を取得
    fetched_stock_list['Listing'] = 1 #現在上場中の銘柄を"1"とする。
    #既存ファイルとの結合
    if raw_stock_list is not None:
        abolished_stock_list = raw_stock_list.loc[~raw_stock_list['Code'].isin(fetched_stock_list['Code'].unique()), :].copy()
        abolished_stock_list['Listing'] = 0 #上場廃止した銘柄を"0"とする
        raw_stock_list = pd.concat([fetched_stock_list, abolished_stock_list], axis=0) #新旧のデータフレームを縦に連結。
    raw_stock_list = raw_stock_list.reset_index(drop=True)
    #データ型の設定
    raw_stock_list[['Code', 'Sector17Code', 'Sector33Code', 'MarketCode', 'MarginCode']] = \
    raw_stock_list[['Code', 'Sector17Code', 'Sector33Code', 'MarketCode', 'MarginCode']].astype(str)
    #データ取得の確認
    display(raw_stock_list.tail(2))
    #データの書き出し
    raw_stock_list.to_parquet(paths.RAW_STOCK_LIST_PARQUET)

def _fetch_stock_price_from_API(year:int, raw_stock_price:pd.DataFrame=None): # API経由で価格情報データを取得するサブ関数
    should_process_stock_price = False
    #データフレームを入力しなかった場合,1年分すべてを取得
    if raw_stock_price is None:
        raw_stock_price = cli.get_price_range(start_dt=datetime(year, 1, 1), end_dt=datetime(year, 12, 31))
    #データフレームを入力した場合、最終データの日から取得
    else:
        last_exist_date = pd.to_datetime(raw_stock_price["Date"]).iat[-1]
        if last_exist_date != datetime.today():
            fetched_stock_price = cli.get_price_range(start_dt=last_exist_date, end_dt=datetime.today())
            if any(fetched_stock_price['AdjustmentFactor'] != 1):
                should_process_stock_price = True
            raw_stock_price = pd.concat([raw_stock_price, fetched_stock_price], axis=0)
            raw_stock_price["Date"] = pd.to_datetime(raw_stock_price["Date"]) # 型変換：object→datetime64[ns]
            raw_stock_price = raw_stock_price[raw_stock_price['Code'].notnull()].drop_duplicates(subset=['Date', 'Code'], keep='last')
            raw_stock_price = raw_stock_price.drop([x for x in raw_stock_price if 'Unnamed' in x], axis=1)
    return raw_stock_price, should_process_stock_price

def update_stock_price(): # 価格情報の更新
    # 終了日を規定
    end_date = datetime.today()
    # 今年のファイルが存在するか判定．
    this_year_path = paths.RAW_STOCK_PRICE_PARQUET.replace('0000', str(end_date.year))
    this_year_file_exists = os.path.exists(this_year_path)

    '''株価取得'''
    # 株価情報を年ごとに別ファイルとして保存
    # 今年
    if this_year_file_exists:
        raw_stock_price = pd.read_parquet(this_year_path)
        raw_stock_price, _ = _fetch_stock_price_from_API(end_date.year, raw_stock_price)
        raw_stock_price = raw_stock_price.drop_duplicates().dropna()
        raw_stock_price.to_parquet(this_year_path)
    else:
        raw_stock_price, should_process1 = _fetch_stock_price_from_API(end_date.year)
        raw_stock_price = raw_stock_price.drop_duplicates().dropna()
        raw_stock_price.to_parquet(this_year_path)
        # 今年のファイルがない場合は去年も取得
        prev_year_path = paths.RAW_STOCK_PRICE_PARQUET.replace('0000', str(end_date.year - 1))
        raw_stock_price = pd.read_parquet(prev_year_path)
        raw_last_stock_price, should_process2 = _fetch_stock_price_from_API(end_date.year - 1, raw_stock_price)
        raw_last_stock_price = raw_last_stock_price.drop_duplicates().dropna()
        raw_stock_price.to_parquet(prev_year_path)
        FlgMng = FlagManager.FlagManager()
        if should_process1 or should_process2:
            FlgMng.flags['process_stock_price'] = True
    display(raw_stock_price.tail(2))

def get_next_open_date(latest_date:datetime, cli:object = cli) -> datetime: # 翌開場日の取得
    from_yyyymmdd = datetime.today().strftime('%Y%m%d')
    to_yyyymmdd = (datetime.today() + relativedelta(months=1)).strftime('%Y%m%d')
    market_open_date = cli.get_markets_trading_calendar(holiday_division="1", from_yyyymmdd=from_yyyymmdd, to_yyyymmdd=to_yyyymmdd)
    market_open_date['Date'] = pd.to_datetime(market_open_date['Date'])
    next_open_date = market_open_date.loc[market_open_date['Date']>latest_date, 'Date'].iat[0]
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