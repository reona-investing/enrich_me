import pandas as pd
import numpy as np
import paths
from jquants_api_operations.processor.formatter import Formatter
from jquants_api_operations.utils import FileHandler
from jquants_api_operations.processor.code_replacement_info import codes_to_merge_dict, manual_adjustment_dict_list,codes_to_replace_dict


def process_fin(raw_path: str = paths.RAW_STOCK_FIN_PARQUET,
                processing_path: str = paths.STOCK_FIN_PARQUET) -> None: # 財務情報の加工
    '''raw_stock_finを、機械学習に使える形に加工します。
    Args:
        raw_path (str): 生の財務データが保存されているパス
        processing_path (str): 加工後の財務データを保存するパス
    '''

    #必要列を抜き取ってデータ型を指定する
    stock_fin = _load_fin_data(raw_path)
    stock_fin = _format_dtypes(stock_fin)
    stock_fin = _drop_duplicated_data(stock_fin)
    stock_fin = _calculate_additional_fins(stock_fin) # 追加の要素を算出
    stock_fin = _process_merger(stock_fin) # 合併処理を実施
    stock_fin = _finalize_df(stock_fin)
    FileHandler.write_parquet(stock_fin, paths.STOCK_FIN_PARQUET)


def _load_fin_data(raw_path: str) -> pd.DataFrame:
    '''生の財務データを読み込みます。'''
    stock_fin = FileHandler.read_parquet(raw_path)
    return stock_fin

def _format_dtypes(raw_stock_fin: pd.DataFrame) -> pd.DataFrame:
    '''各カラムに適切なデータ型を設定します。'''
    dtypes_spec_df = pd.read_csv(paths.DTYPES_STOCK_FIN_CSV)
    dtypes_spec_dict = {row['列名']:eval(row['型']) for _, row in dtypes_spec_df.iterrows()} #eval関数で、文字列'str'をデータ型オブジェクトstrに変換
    columns_list = dtypes_spec_dict.keys()
    stock_fin = raw_stock_fin[[x for x in columns_list]].replace('', np.nan).infer_objects().copy()
    stock_fin = stock_fin.astype(dtypes_spec_dict)

    stock_fin = stock_fin.rename(columns={'LocalCode':'Code', 'DisclosedDate':'Date'})  

    datetime_columns = ['Date', 'CurrentPeriodEndDate', 'CurrentFiscalYearStartDate', 'CurrentFiscalYearEndDate']
    for column in datetime_columns:
        stock_fin[column]= stock_fin[column].astype(str)
        stock_fin[column]= stock_fin[column].str[:10]
        stock_fin[column] = pd.to_datetime(stock_fin[column])
  
    stock_fin = Formatter.format_stock_code(stock_fin)
    stock_fin['Code'] = stock_fin['Code'].replace(codes_to_replace_dict)
    return stock_fin

def _drop_duplicated_data(stock_fin: pd.DataFrame) -> pd.DataFrame:
    '''同一日に同一銘柄の複数レコードが存在する場合、重複を削除します。'''
    # 発表の訂正などで、同一日に複数回のリリースがある場合がある。当日最後の発表を最新の発表として残す。
    stock_fin = stock_fin.loc[~stock_fin["TypeOfDocument"].isin(
        ['DividendForecastRevision', 'EarnForecastRevision', 'REITDividendForecastRevision', 'REITEarnForecastRevision'])]
    return stock_fin.sort_values("DisclosedTime").drop_duplicates(subset=["Code", "Date"], keep="last") #コードと開示日が重複しているデータを、最後を残して削除


def _calculate_additional_fins(stock_fin: pd.DataFrame) -> pd.DataFrame:
    '''追加の財務要素を算出します。'''
    #期末発行済株式数の算出
    stock_fin['OutstandingShares'] = np.nan
    stock_fin.loc[stock_fin['NumberOfTreasuryStockAtTheEndOfFiscalYear'].notnull(), 'OutstandingShares'] = \
      stock_fin['NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock'] -\
      stock_fin['NumberOfTreasuryStockAtTheEndOfFiscalYear']
    stock_fin.loc[stock_fin['NumberOfTreasuryStockAtTheEndOfFiscalYear'].isnull(), 'OutstandingShares'] = \
      stock_fin['NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock']
    #その他、追加で必要な列を算出
    stock_fin["CurrentFiscalYear"] = stock_fin["CurrentFiscalYearEndDate"].dt.strftime("%Y")
    stock_fin["ForecastFiscalYearEndDate"] = stock_fin["CurrentFiscalYearEndDate"].dt.strftime("%Y/%m")
    stock_fin.loc[stock_fin["TypeOfCurrentPeriod"] == "FY", 'ForecastFiscalYearEndDate'] = (
        stock_fin.loc[stock_fin["TypeOfCurrentPeriod"] == "FY", "CurrentFiscalYearEndDate"] + pd.offsets.DateOffset(years=1)).dt.strftime("%Y/%m")
    return stock_fin

def _process_merger(stock_fin:pd.DataFrame) -> pd.DataFrame:
    '''企業合併時の財務情報合成処理を行います'''
    #合併前の各社のデータを足し合わせる項目
    plus_when_merging = ['TotalAssets', 'Equity','CashAndEquivalents',	'NetSales',	'OperatingProfit',
                          'OrdinaryProfit',	'Profit', 'ForecastNetSales',	'ForecastOperatingProfit',
                          'ForecastOrdinaryProfit',	'ForecastProfit', 'CashFlowsFromOperatingActivities',
                          'CashFlowsFromInvestingActivities', 'CashFlowsFromFinancingActivities']
    #合併リストの中身の分だけ処理を繰り返し
    for key, value in codes_to_merge_dict.items():
        merger1 = stock_fin.loc[stock_fin['Code']==value['Code1']].sort_values('Date') #合併前1（存続）
        merger2 = stock_fin.loc[stock_fin['Code']==value['Code2']].sort_values('Date') #合併前2（消滅）
        #存続会社のデータを、合併前後に分ける。
        merger1_before = merger1[merger1['Date']<=value['MergerDate']]
        merger1_after = merger1[merger1['Date']>value['MergerDate']]
        #インデックスを揃える
        merger1_before = pd.merge(merger1_before, merger2['Date'], how='outer', on='Date').sort_values('Date').reset_index(drop=True)
        merger2 = pd.merge(merger2, merger1_before['Date'], how='outer', on='Date').sort_values('Date').reset_index(drop=True)
        #NaN値を埋める
        merger1_before = merger1_before.ffill()
        merger1_before = merger1_before.bfill()
        merger2 = merger2.ffill()
        merger2 = merger2.bfill()
        #merger1_beforeの値を書き換えていく。
        merger1_before['Code'] = key
        merger1_before['OutstandingShares'] = merger1_before['OutstandingShares'] + \
                                              merger2['OutstandingShares'] * value['ExchangeRate']
        merger1_before[plus_when_merging] =  merger1_before[plus_when_merging] + merger2[plus_when_merging]
        #合併前後のデータを結合する。
        merged = pd.concat([merger1_before, merger1_after], axis=0).sort_values('Date')
        stock_fin = stock_fin[(stock_fin['Code']!=value['Code1'])&(stock_fin['Code']!=value['Code2'])]
        return pd.concat([stock_fin, merged], axis=0).sort_values(['Date', 'Code'])

def _finalize_df(stock_fin: pd.DataFrame) -> pd.DataFrame:
    '''データフレームに最終処理を施します。'''
    return stock_fin.drop(['DisclosedTime'], axis=1).drop_duplicates(keep='last').reset_index(drop=True) # データフレームの最終処理


if __name__ == '__main__':
    process_fin()