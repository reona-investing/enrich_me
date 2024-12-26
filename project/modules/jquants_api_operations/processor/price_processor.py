import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, List, Dict
from FlagManager import flag_manager
import paths
from jquants_api_operations.processor.formatter import Formatter
from jquants_api_operations.utils import FileHandler
from jquants_api_operations.processor.code_replacement_info import manual_adjustment_dict_list,codes_to_replace_dict


def process_price():
    """価格情報の加工"""
    end_date = datetime.today()
    temp_cumprod = None
    
    for year in range(end_date.year, 2012, -1):
        is_latest_file = year == end_date.year

        if is_latest_file or flag_manager.flags['process_stock_price']:
            stock_price = _load_yearly_raw_data(year)
            stock_price = _process_stock_price(stock_price, temp_cumprod, is_latest_file)
            _save_yearly_data(stock_price, year)


# サブプロセス
def _load_yearly_raw_data(year: int) -> pd.DataFrame:
    raw_path = paths.RAW_STOCK_PRICE_PARQUET.replace('0000', str(year))
    usecols = ['Date', 'Code', 'Open', 'High', 'Low', 'Close', 'Volume', 'TurnoverValue', 'AdjustmentFactor']
    return FileHandler.read_parquet(raw_path, usecols=usecols)

def _process_stock_price(stock_price: pd.DataFrame, temp_cumprod: dict, is_latest_file: bool) -> pd.DataFrame:
    """価格データを加工する"""
    stock_price['Code'] = stock_price['Code'].astype(str)
    stock_price = Formatter.format_stock_code(stock_price)
    stock_price['Code'] = _replace_code(stock_price['Code'])
    stock_price = _fill_suspension_period(stock_price)
    stock_price = _format_dtypes(stock_price)
    stock_price = _remove_system_failure_day(stock_price)
    stock_price, temp_cumprod = _apply_cumulative_adjustment_factor(
        stock_price, temp_cumprod, is_latest_file, manual_adjustment_dict_list
    )
    return _finalize_price_data(stock_price)

def _replace_code(code_row: pd.Series) -> pd.Series:
    """銘柄コードを置換する"""
    return code_row.replace(codes_to_replace_dict)

def _fill_suspension_period(stock_price: pd.DataFrame) -> pd.DataFrame:
    """欠損期間のデータを埋める"""
    rows_to_add = []
    date_list = stock_price['Date'].unique()
    codes_after_replacement = codes_to_replace_dict.values()

    for code_replaced in codes_after_replacement:
        dates_to_fill = _get_missing_dates(stock_price, code_replaced, date_list)
        rows_to_add.extend(_create_missing_rows(stock_price, code_replaced, dates_to_fill))

    return pd.concat([stock_price] + rows_to_add, axis=0)

def _get_missing_dates(stock_price: pd.DataFrame, code_replaced: str, date_list: List[str]) -> List[str]:
    """欠損している日付を取得"""
    existing_dates = stock_price.loc[stock_price['Code'] == code_replaced, 'Date'].unique()
    return [x for x in date_list if x not in existing_dates]

def _create_missing_rows(stock_price: pd.DataFrame, code: str, dates_to_fill: List[str]) -> List[pd.DataFrame]:
    """欠損期間の行を作成する"""
    rows = []
    if len(dates_to_fill) <= 5:
        for date in dates_to_fill:
            last_date = stock_price.loc[(stock_price['Code'] == code) & (stock_price['Date'] <= date), 'Date'].max()
            value_to_fill = stock_price.loc[(stock_price['Code'] == code) & (stock_price['Date'] == last_date), 'Close'].values[0]
            row_to_add = pd.DataFrame([[np.nan] * len(stock_price.columns)], columns=stock_price.columns)
            row_to_add.update({'Date': date, 'Code': code, 'Open': value_to_fill, 'Close': value_to_fill,
                               'High': value_to_fill, 'Low': value_to_fill, 'Volume': 0,
                               'TurnoverValue': 0, 'AdjustmentFactor': 1})
            rows.append(row_to_add)
    return rows

def _format_dtypes(stock_price: pd.DataFrame) -> pd.DataFrame:
    """データ型を整形する"""
    stock_price['Code'] = stock_price['Code'].astype(str)
    stock_price['Date'] = pd.to_datetime(stock_price['Date'])
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'AdjustmentFactor', 'TurnoverValue']
    stock_price[numeric_cols] = stock_price[numeric_cols].astype(np.float64)
    return stock_price

def _remove_system_failure_day(stock_price: pd.DataFrame) -> pd.DataFrame:
    """システム障害日を除外する"""
    return stock_price[stock_price['Date'] != '2020-10-01']

def _apply_cumulative_adjustment_factor(
    stock_price: pd.DataFrame, temp_cumprod: dict, is_latest_file: bool,
    manual_adjustment_dict_list: List[Dict]
) -> Tuple[pd.DataFrame, dict]:
    """累積調整係数を適用"""
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
    """累積調整係数を計算する"""
    stock_price = stock_price.sort_values('Date', ascending=False)
    stock_price['AdjustmentFactor'] = stock_price['AdjustmentFactor'].shift(-1).fillna(1.0)
    stock_price['CumulativeAdjustmentFactor'] = 1 / stock_price['AdjustmentFactor'].cumprod()
    return stock_price.sort_values('Date')

def _inherit_cumulative_values(stock_price: pd.DataFrame, temp_cumprod: dict) -> pd.DataFrame:
    """累積値を引き継ぐ"""
    stock_price['InheritedValue'] = stock_price['Code'].map(temp_cumprod).fillna(1)
    stock_price['CumulativeAdjustmentFactor'] *= stock_price['InheritedValue']
    return stock_price.drop(columns='InheritedValue')

def _apply_manual_adjustments(stock_price: pd.DataFrame, manual_adjustments: List[Dict]) -> pd.DataFrame:
    """手動調整を適用する"""
    for adjustment in manual_adjustments:
        condition = (stock_price['Code'] == adjustment['Code']) & (stock_price['Date'] < adjustment['Date'])
        stock_price.loc[condition, 'CumulativeAdjustmentFactor'] *= adjustment['Rate']
    return stock_price

def _finalize_price_data(stock_price: pd.DataFrame) -> pd.DataFrame:
    """最終的なデータ整形"""
    stock_price = stock_price.dropna(subset=['Code'])
    stock_price = stock_price.drop_duplicates(subset=['Date', 'Code'], keep='last')
    return stock_price.sort_values(['Date', 'Code']).reset_index(drop=True)[
        ['Date', 'Code', 'Open', 'High', 'Low', 'Close', 'Volume', 'AdjustmentFactor', 'CumulativeAdjustmentFactor', 'TurnoverValue']
    ]

def _save_yearly_data(df: pd.DataFrame, year: int) -> None:
    save_path = paths.STOCK_PRICE_PARQUET.replace('0000', str(year))
    FileHandler.write_parquet(df, save_path)


if __name__ == '__main__':
    process_price()