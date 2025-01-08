import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, List, Dict
from FlagManager import flag_manager, Flags
import paths
from jquants_api_operations.processor.formatter import Formatter
from jquants_api_operations.utils import FileHandler
from jquants_api_operations.processor.code_replacement_info import manual_adjustment_dict_list,codes_to_replace_dict


def process_price(raw_basic_path: str = paths.RAW_STOCK_PRICE_PARQUET,
                  processing_basic_path: str = paths.STOCK_PRICE_PARQUET):
    """
    価格情報を加工して、機械学習用に整形します。

    Args:
        raw_basic_path (str): 生の株価データが保存されているパス。
        processing_basic_path (str): 加工後の株価データを保存するパス。
    """
    end_date = datetime.today()
    temp_cumprod = {}
    
    for year in range(end_date.year, 2012, -1):
        is_latest_file = year == end_date.year
        should_process = is_latest_file or flag_manager.flags[Flags.PROCESS_STOCK_PRICE]
        if should_process:
            stock_price = _load_yearly_raw_data(raw_basic_path, year)
            if stock_price.empty:
                continue
            stock_price, temp_cumprod = _process_stock_price(stock_price, temp_cumprod, is_latest_file)
            _save_yearly_data(stock_price, processing_basic_path, year)


# サブプロセス
def _load_yearly_raw_data(raw_basic_path: str, year: int) -> pd.DataFrame:
    '''取得したままの年次株価データを読み込みます。'''
    raw_path = raw_basic_path.replace('0000', str(year))
    usecols = ['Date', 'Code', 'Open', 'High', 'Low', 'Close', 'Volume', 'TurnoverValue', 'AdjustmentFactor']
    return FileHandler.read_parquet(raw_path, usecols=usecols)

def _process_stock_price(stock_price: pd.DataFrame, temp_cumprod: dict[str, float], is_latest_file: bool) -> pd.DataFrame:
    """
    価格データを加工します。
    Args:
        stock_price (pd.DataFrame): 加工前の株価データ
        temp_cumprod (dict[str, float]): 
            処理時点での銘柄ごとの暫定の累積調整係数を格納（キー: 銘柄コード、値：暫定累積調整係数）
        is_latest_file (bool): stock_priceが最新期間のファイルかどうか
    Returns:
        pd.DataFrame: 加工された株価データ。
    """
    stock_price['Code'] = Formatter.format_stock_code(stock_price['Code'].astype(str))
    stock_price['Code'] = _replace_code(stock_price['Code'])
    stock_price = _fill_suspension_period(stock_price)
    stock_price = _format_dtypes(stock_price)
    stock_price = _remove_system_failure_day(stock_price)
    stock_price, temp_cumprod = _apply_cumulative_adjustment_factor(
        stock_price, temp_cumprod, is_latest_file, manual_adjustment_dict_list
    )
    return _finalize_price_data(stock_price), temp_cumprod

def _replace_code(code_column: pd.Series) -> pd.Series:
    """ルールに従い、銘柄コードを置換します。"""
    return code_column.replace(codes_to_replace_dict)

def _fill_suspension_period(stock_price: pd.DataFrame) -> pd.DataFrame:
    """銘柄コード変更前後の欠損期間のデータを埋めます。"""
    rows_to_add = []
    date_list = stock_price['Date'].unique()
    codes_after_replacement = codes_to_replace_dict.values()

    for code_replaced in codes_after_replacement:
        dates_to_fill = _get_missing_dates(stock_price, code_replaced, date_list)
        rows_to_add.extend(_create_missing_rows(stock_price, code_replaced, dates_to_fill))

    return pd.concat([stock_price] + rows_to_add, axis=0)

def _get_missing_dates(stock_price: pd.DataFrame, code_replaced: str, date_list: List[str]) -> List[str]:
    """データが欠損している日付を取得します。"""
    existing_dates = stock_price.loc[stock_price['Code'] == code_replaced, 'Date'].unique()
    return [x for x in date_list if x not in existing_dates]

def _create_missing_rows(stock_price: pd.DataFrame, code: str, dates_to_fill: List[str]) -> List[pd.DataFrame]:
    """欠損期間の行を作成します。"""
    rows = []
    if len(dates_to_fill) <= 5:
        for date in dates_to_fill:
            last_date = stock_price.loc[(stock_price['Code'] == code) & (stock_price['Date'] <= date), 'Date'].max()
            value_to_fill = stock_price.loc[(stock_price['Code'] == code) & (stock_price['Date'] == last_date), 'Close'].values[0]
            row_to_add = {'Date': date, 'Code': code, 'Open': value_to_fill, 'Close': value_to_fill,
                          'High': value_to_fill, 'Low': value_to_fill, 'Volume': 0,
                          'TurnoverValue': 0, 'AdjustmentFactor': 1}
            rows.append(pd.DataFrame([row_to_add], index=[0]))
    return rows

def _format_dtypes(stock_price: pd.DataFrame) -> pd.DataFrame:
    """データ型を整形します。"""
    stock_price['Code'] = stock_price['Code'].astype(str)
    stock_price['Date'] = pd.to_datetime(stock_price['Date'])
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'AdjustmentFactor', 'TurnoverValue']
    stock_price[numeric_cols] = stock_price[numeric_cols].astype(np.float64)
    return stock_price

def _remove_system_failure_day(stock_price: pd.DataFrame) -> pd.DataFrame:
    """システム障害によるデータ欠損日を除外します。"""
    return stock_price[stock_price['Date'] != '2020-10-01']

def _apply_cumulative_adjustment_factor(
    stock_price: pd.DataFrame, temp_cumprod: dict, is_latest_file: bool,
    manual_adjustment_dict_list: List[Dict]
) -> Tuple[pd.DataFrame, dict]:
    """価格データ（OHLCV）に累積調整係数を適用します。"""
    stock_price = stock_price.sort_values(['Code', 'Date']).set_index('Code', drop=True)
    stock_price = stock_price.groupby('Code', group_keys=False).apply(_calculate_cumulative_adjustment_factor).reset_index(drop=False)

    if not is_latest_file:
        stock_price = _inherit_cumulative_values(stock_price, temp_cumprod)

    temp_cumprod = stock_price.groupby('Code')['CumulativeAdjustmentFactor'].last().to_dict()
    stock_price = _apply_manual_adjustments(stock_price, manual_adjustment_dict_list)

    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        stock_price[col] /= stock_price['CumulativeAdjustmentFactor']

    return stock_price, temp_cumprod

def _calculate_cumulative_adjustment_factor(stock_price: pd.DataFrame) -> pd.DataFrame:
    """累積調整係数を計算します。"""
    stock_price = stock_price.sort_values('Date', ascending=False)
    stock_price['AdjustmentFactor'] = stock_price['AdjustmentFactor'].shift(-1).fillna(1.0)
    stock_price['CumulativeAdjustmentFactor'] = 1 / stock_price['AdjustmentFactor'].cumprod()
    return stock_price.sort_values('Date')

def _inherit_cumulative_values(stock_price: pd.DataFrame, temp_cumprod: dict) -> pd.DataFrame:
    """計算途中の暫定累積調整係数を引き継ぎます。"""
    stock_price['InheritedValue'] = stock_price['Code'].map(temp_cumprod).fillna(1)
    stock_price['CumulativeAdjustmentFactor'] *= stock_price['InheritedValue']
    return stock_price.drop(columns='InheritedValue')

def _apply_manual_adjustments(stock_price: pd.DataFrame, manual_adjustments: List[Dict]) -> pd.DataFrame:
    """元データで未掲載の株式分割・併合について、累積調整係数をマニュアルで調整する"""
    for adjustment in manual_adjustments:
        condition = (stock_price['Code'] == adjustment['Code']) & (stock_price['Date'] < adjustment['Date'])
        stock_price.loc[condition, 'CumulativeAdjustmentFactor'] *= adjustment['Rate']
    return stock_price

def _finalize_price_data(stock_price: pd.DataFrame) -> pd.DataFrame:
    """最終的なデータ整形を行う。"""
    stock_price = stock_price.dropna(subset=['Code'])
    stock_price = stock_price.drop_duplicates(subset=['Date', 'Code'], keep='last')
    return stock_price.sort_values(['Date', 'Code']).reset_index(drop=True)[
        ['Date', 'Code', 'Open', 'High', 'Low', 'Close', 'Volume', 'AdjustmentFactor', 'CumulativeAdjustmentFactor', 'TurnoverValue']
    ]

def _save_yearly_data(df: pd.DataFrame, processing_basic_path: str, year: int) -> None:
    '''年次の加工後価格データを保存する。'''
    save_path = processing_basic_path.replace('0000', str(year))
    FileHandler.write_parquet(df, save_path)


if __name__ == '__main__':

    flag_manager.set_flag(Flags.PROCESS_STOCK_PRICE, True)
    process_price()