from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
from utils.jquants_api_utils import cli

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

def is_market_open(date: datetime) -> bool:
    market_open_day_df = cli.get_markets_trading_calendar(
            from_yyyymmdd=(date).strftime('%Y%m%d'),
            to_yyyymmdd=(date).strftime('%Y%m%d')
        )
    return market_open_day_df['HolidayDivision'].iat[0] == '1' #平日：1、土日：0、祝日：3