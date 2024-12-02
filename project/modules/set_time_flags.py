#%% モジュールのインポート
from datetime import datetime, timedelta
import pandas as pd
import os

import jquants_api_fetcher as fetcher
import paths
import FlagManager

#%% 関数
if os.name == 'posix':
    _nowtime = datetime.now() + timedelta(hours=9)
else:
    _nowtime = datetime.now()

print(_nowtime)

def _get_market_open_day(nowtime:datetime=_nowtime) -> str:
    # 今日が営業日かどうかの判定
    market_open_day_df = fetcher.cli.get_markets_trading_calendar(
            from_yyyymmdd=(nowtime).strftime('%Y%m%d'),
            to_yyyymmdd=(nowtime).strftime('%Y%m%d')
        )
    is_market_open_day =  market_open_day_df['HolidayDivision'].iat[0] == '1' #平日：1、土日：0、祝日：3
    return is_market_open_day

def _determine_whether_take_positions(nowtime:datetime=_nowtime) -> bool:
    # 発注時刻（朝7時台～8時台）かどうかの判定
    should_take_positions = (nowtime.hour >= 7) & (nowtime.hour <= 8)
    print(f'予測＆新規建：{should_take_positions}')

    return should_take_positions

def _determine_whether_take_additionals(nowtime:datetime=_nowtime) -> bool:
    # 追加発注（朝9時～9時半）かどうかの判定
    should_take_additionals = (nowtime.hour == 9) & (nowtime.minute <= 30)
    print(f'追加の新規建：{should_take_additionals}')

    return should_take_additionals

def _determine_whether_settle_positions(nowtime:datetime=_nowtime) -> bool:
    #決済時刻（11時台半～14時台）かどうかの判定
    eleven_thirty = nowtime.replace(hour=11, minute=30, second=0, microsecond=0)
    should_settle_positions = (nowtime >= eleven_thirty) & (nowtime.hour <= 14)
    print(f'建玉決済：{should_settle_positions}')

    return should_settle_positions


def _determine_whether_fetch_invest_result(nowtime:datetime=_nowtime):
    #現時点での最新データの日付を取得
    invest_result = pd.read_csv(paths.TRADE_HISTORY_CSV)
    invest_result['日付'] = pd.to_datetime(invest_result['日付'])
    latest_date = invest_result['日付'].iloc[-1]
    #開場日、当日データ未取得、かつ15時台～23時台の場合、当日取引情報を取得
    should_fetch_invest_result = (latest_date.date() != datetime.today().date()) & (nowtime.hour >= 15) & (nowtime.hour <= 23)
    print(f'取引情報取得：{should_fetch_invest_result}')
    return should_fetch_invest_result

#%% フラグの読み込み
def set_time_flags(should_take_positions: bool = None,
                   should_take_additionals: bool = None,
                   should_update_historical_data: bool = None,
                   should_settle_positions: bool = None,
                   should_fetch_invest_result: bool = None):
    '''
    時刻に応じてフラグを読み込む関数。
    時刻に関係なくTrue/Falseを設定したい場合のみ引数指定。
    '''
    now_this_model = FlagManager.launch()
    is_market_open_day = _get_market_open_day()
    # マーケット開場日であれば、時刻に応じたフラグを設定
    if is_market_open_day:
        now_this_model.should_take_positions = \
        now_this_model.should_update_historical_data =\
            _determine_whether_take_positions()
        now_this_model.should_take_additionals = _determine_whether_take_additionals()
        now_this_model.should_settle_positions = _determine_whether_settle_positions()
        now_this_model.should_fetch_invest_result = _determine_whether_fetch_invest_result()
    # 閉場日以外はすべてFalse
    else:
        now_this_model.should_take_positions = \
        now_this_model.should_take_addiionals = \
        now_this_model.should_update_historical_data = \
        now_this_model.should_settle_positions = \
        now_this_model.should_fetch_invest_result = \
            False
    # 引数で個別指定がある場合は上書き
    if should_take_positions is not None:
        now_this_model.should_take_positions = should_take_positions
    if should_take_additionals is not None:
        now_this_model.should_take_additionals = should_take_additionals
    if should_update_historical_data is not None:
        now_this_model.should_update_historical_data = should_update_historical_data
    if should_settle_positions is not None:
        now_this_model.should_settle_positions = should_settle_positions
    if should_fetch_invest_result is not None:
        now_this_model.should_fetch_invest_result = should_fetch_invest_result