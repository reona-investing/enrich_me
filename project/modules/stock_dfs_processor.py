#%% モジュールのインポート
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, List

import paths
import FlagManager

#%% コード変更・合併の情報の記述
'''持株会社化や合併などでコードが変わった銘柄について、コードを置換する'''
'''対象期間：2013年4月5日～'''
#持株会社化などでコードが変わった会社を格納した辞書
codes_to_replace_dict = pd.read_csv(paths.CODES_TO_REPLACE_CSV, dtype=str).set_index("code_before")["code_after"].to_dict()
#株式分割設定用の辞書をリスト化
#証券コード変更のタイミングで株式分割・併合が行われた場合、API上に情報がないため、手動でAdjustmentRateに掛ける。
manual_adjustment_dict_list = [
  {'Code':'1333', 'Date':datetime(2014,4,1), 'Rate':0.1},
  {'Code':'5021', 'Date':datetime(2015,10,1), 'Rate':0.1}
                      ]
#合併用の辞書を辞書化
codes_to_merge_dict = {'7167':{'Code1':'7167', 'Code2':'8333', 'MergerDate':datetime(2016,12,19), 'ExchangeRate':1.17}} #足利HD + 常陽銀行 → めぶきHD

#%% ヘルパー関数群
def _format_stock_code(stock_df: pd.DataFrame) -> pd.DataFrame:
    '''普通株の銘柄コードを4桁に変換するヘルパー関数'''
    stock_df["Code"] = \
        stock_df["Code"].str.rstrip("0").where(stock_df["Code"].str.len() == 5, stock_df["Code"])
    return stock_df


def _fill_suspension_period(stock_price:pd.DataFrame) \
                                -> pd.DataFrame:
    '''銘柄コード変更前後の売買停止期間のデータを埋める'''
    rows_to_add = []
    date_list_of_stock_price = stock_price['Date'].unique()
    codes_after_replacement =  codes_to_replace_dict.values()
    for code_replaced in codes_after_replacement:
        dates_to_fill = stock_price.loc[stock_price['Code']==code_replaced, 'Date'].unique()
        dates_to_fill = [x for x in date_list_of_stock_price if x not in dates_to_fill]
        if len(dates_to_fill) <= 5: #上場前のためデータがない場合を除去するため、欠損データが5以下のものを対象とする。
            for date in dates_to_fill:
                last_date = stock_price.loc[(stock_price['Code'] == code_replaced)&(stock_price['Date'] <= date), 'Date'].max()
                value_to_fill = stock_price.loc[(stock_price['Code'] == code_replaced)&(stock_price['Date'] == last_date), 'Close'].values[0]
                #追加すべき行を作成する。
                row_to_add = pd.DataFrame([[np.nan] * len(stock_price.columns)], columns=stock_price.columns)
                print(f'{date} {code_replaced} {value_to_fill}')
                row_to_add['Date'] = date
                row_to_add['Code'] = code_replaced
                row_to_add[['Open', 'Close', 'High', 'Low']] = value_to_fill
                row_to_add[['Volume', 'TurnoverValue']] = 0
                row_to_add['AdjustmentFactor'] = 1
                rows_to_add.append(row_to_add)
    return pd.concat([stock_price] + rows_to_add, axis=0)


def _calculate_cumulative_adjustment_factor(stock_price:pd.DataFrame) -> pd.DataFrame:
    '''累積調整係数を作成（現在が1、過去に遡り累積積の逆数を算出）'''
    stock_price = stock_price.sort_values("Date", ascending=False) #調整係数を作成するために逆順にソートする
    stock_price.loc[:, "AdjustmentFactor"] = stock_price["AdjustmentFactor"].shift(-1).fillna(1.0) #分割併合等の係数を適用日に変更（デフォルトだと適用前日？）
    stock_price.loc[:, "CumulativeAdjustmentFactor"] = 1 / stock_price["AdjustmentFactor"].cumprod() #累積株価調整係数を作成
    return stock_price.sort_values("Date")  #ソート順を昇順にする


def _apply_cumulative_adjustment_factor( # 累積調整係数を適用
    stock_price:pd.DataFrame, temp_cumprod:dict, is_latest_file:bool, 
    manual_adjustment_dict_list:list=manual_adjustment_dict_list) -> Tuple[pd.DataFrame, dict]:
    
    stock_price = stock_price.sort_values(["Code", "Date"])
    stock_price = stock_price.groupby("Code", group_keys=False).apply(
        _calculate_cumulative_adjustment_factor).reset_index(drop=True)
    if is_latest_file == False: #1つ目のdf以外は、累積調整係数を前のdfから引き継ぐ。
        stock_price['InheritedValue'] = stock_price['Code'].map(temp_cumprod).fillna(1) #前のdfまでの累積を追加
        stock_price["CumulativeAdjustmentFactor"] = stock_price["CumulativeAdjustmentFactor"] * stock_price['InheritedValue'] #累積積を補正
        stock_price = stock_price.drop('InheritedValue', axis=1) #一時使用した'result'列を削除。
    temp_cumprod = stock_price.set_index('Code').groupby('Code')["CumulativeAdjustmentFactor"].head(1).to_dict() #各Codeの最終的な累積積を辞書型で取得
    #manual_adjust_dict_listで指定した銘柄について、累積積にさらに係数を掛ける
    for dictionary in manual_adjustment_dict_list:
        rows_to_apply_manual_adjustment = (stock_price['Code']==dictionary['Code'])&(stock_price['Date']<dictionary['Date'])
        stock_price.loc[rows_to_apply_manual_adjustment, 'CumulativeAdjustmentFactor'] = \
          stock_price.loc[rows_to_apply_manual_adjustment, 'CumulativeAdjustmentFactor'] * dictionary['Rate']
    #OHCLVに累積調整係数をかける。
    for column in ['Open', 'High', 'Low', 'Close', 'Volume']:
        stock_price[column] = stock_price[column] / stock_price["CumulativeAdjustmentFactor"]
    return stock_price, temp_cumprod


#%% メイン関数群
def process_stock_list() -> None:
    """
    銘柄リストデータを加工して、機械学習用に整形する。
    """
    raw_stock_list = pd.read_parquet(paths.RAW_STOCK_LIST_PARQUET)
    str_columns = ['Code', 'CompanyName', 'MarketCodeName', 'Sector33CodeName', 'Sector17CodeName', 'ScaleCategory']
    int_columns = ['Listing']
    stock_list = raw_stock_list[str_columns + int_columns].copy()
    stock_list[str_columns] = stock_list[str_columns].astype(str)
    stock_list[int_columns] = stock_list[int_columns].astype(int)
    stock_list = _format_stock_code(stock_list)
    # ETF等を除外し、株式のみのデータに絞る。
    stock_list = stock_list[stock_list['Sector33CodeName'] != 'その他'].drop_duplicates()
    stock_list.to_parquet(paths.STOCK_LIST_PARQUET)


def process_stock_price(): # 価格情報の加工
    '''raw_stock_priceを、機械学習に使える形に加工'''
    #年ごとにファイルを分割している（CumulativeAdjustmentFactorの計算時にメモリオーバーを起こさないために）
    end_date = datetime.today()
    temp_cumprod = None
    for year in range(end_date.year, 2012, -1):
        is_latest_file = year == end_date.year
        FlgMng = FlagManager.FlagManager()
        if is_latest_file or FlgMng.flags['process_stock_price']:
            raw_path = paths.RAW_STOCK_PRICE_PARQUET.replace('0000', str(year))
            usecols = ['Date', 'Code', 'Open', 'High', 'Low', 'Close', 'Volume', 'TurnoverValue', 'AdjustmentFactor']
            stock_price = pd.read_parquet(raw_path, columns=usecols)
            stock_price['Code']  = stock_price['Code'].astype(str)
            '''銘柄コード関連の処理'''
            # 普通株 (5桁で末尾が0) の銘柄コードを4桁にします
            stock_price = _format_stock_code(stock_price)
            #持株会社化などで証券コードが変わる場合の処理
            stock_price['Code'] = stock_price['Code'].replace(codes_to_replace_dict)
            stock_price = _fill_suspension_period(stock_price)
            #型置換
            stock_price['Code'] = stock_price['Code'].astype(str)
            stock_price["Date"] = pd.to_datetime(stock_price["Date"])
            stock_price[['Open', 'High', 'Low', 'Close', 'Volume', 'AdjustmentFactor', 'TurnoverValue']] = \
              stock_price[['Open', 'High', 'Low', 'Close', 'Volume', 'AdjustmentFactor', 'TurnoverValue']].astype(np.float64)
            #2020年10月01日（システム障害で終日取引停止）の行を削除
            stock_price = stock_price.loc[stock_price['Date']!='2020-10-01', :]
            # 累積調整係数を追加
            stock_price, temp_cumprod = _apply_cumulative_adjustment_factor(stock_price, temp_cumprod, is_latest_file)
            #その他、データを整える
            stock_price = stock_price.loc[stock_price['Code'].notnull(), :] #場中にデータ取得すると，インデックス情報のみの列ができるため、それを除去
            stock_price = stock_price.sort_values(by=['Date', 'Code']).reset_index(drop=True)
            stock_price = stock_price.drop_duplicates(subset=['Date', 'Code'], keep='last')
            stock_price = stock_price[['Date', 'Code', 'Open', 'High', 'Low', 'Close', 'Volume', 'AdjustmentFactor', 'CumulativeAdjustmentFactor', 'TurnoverValue']]
            #csvファイルの出力
            save_path = paths.STOCK_PRICE_PARQUET.replace('0000', str(year))
            stock_price.to_parquet(save_path)


def process_stock_fin(): # 財務情報の加工
    '''raw_stock_finを、機械学習に使える形に加工'''
    #pklの読み込み
    raw_stock_fin = pd.read_parquet(paths.RAW_STOCK_FIN_PARQUET)
    #必要列を抜き取ってデータ型を指定する
    dtypes_spec_df = pd.read_csv(paths.DTYPES_STOCK_FIN_CSV)
    dtypes_spec_dict = {row['列名']:eval(row['型']) for _, row in dtypes_spec_df.iterrows()} #eval関数で、文字列'str'をデータ型オブジェクトstrに変換
    columns_list = dtypes_spec_dict.keys()
    stock_fin = raw_stock_fin[[x for x in columns_list]].replace('', np.nan).infer_objects().copy()
    stock_fin = stock_fin.astype(dtypes_spec_dict)
    stock_fin = stock_fin.rename(columns={'LocalCode':'Code', 'DisclosedDate':'Date'})
    #datetime型に変換
    datetime_columns = ['Date', 'CurrentPeriodEndDate', 'CurrentFiscalYearStartDate', 'CurrentFiscalYearEndDate']
    for column in datetime_columns:
        stock_fin[column]= stock_fin[column].astype(str)
        stock_fin[column]= stock_fin[column].str[:10]
        stock_fin[column] = pd.to_datetime(stock_fin[column])
    # 普通株 (5桁で末尾が0) の銘柄コードを4桁に
    stock_fin = _format_stock_code(stock_fin)
    #証券コードが変わった銘柄について、銘柄コードの置換
    stock_fin['Code'] = stock_fin['Code'].replace(codes_to_replace_dict)
    # 財務情報の同一日に複数レコードが存在することに対応します。
    # ある銘柄について同一日に複数の開示が行われた場合レコードが重複します。
    # ここでは簡易的に処理するために特定のTypeOfDocumentを削除した後に、開示時間順に並べて一番最後に発表された開示情報を採用しています。
    stock_fin = stock_fin.loc[~stock_fin["TypeOfDocument"].isin(
        ['DividendForecastRevision', 'EarnForecastRevision', 'REITDividendForecastRevision', 'REITEarnForecastRevision'])]
    stock_fin = stock_fin.sort_values("DisclosedTime").drop_duplicates(subset=["Code", "Date"], keep="last") #コードと開示日が重複しているデータを、最後を残して削除
    stock_fin = _calculate_additional_fins(stock_fin) # 追加の要素を算出
    stock_fin = _process_merger(stock_fin) # 合併処理を実施
    stock_fin = stock_fin.drop(['DisclosedTime'], axis=1).drop_duplicates(keep='last').reset_index(drop=True) # データフレームの最終処理
    stock_fin.to_parquet(paths.STOCK_FIN_PARQUET) # 最終成果物の保存


def _calculate_additional_fins(stock_fin: pd.DataFrame) -> pd.DataFrame:
    '''追加の要素を算出'''
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

def _process_merger(stock_fin:pd.DataFrame) -> pd.DataFrame: # 企業合併時の財務情報処理
    #合併前の各社のデータを足し合わせる項目
    plus_when_merging = ['TotalAssets', 'Equity','CashAndEquivalents',	'NetSales',	'OperatingProfit',
                          'OrdinaryProfit',	'Profit', 'ForecastNetSales',	'ForecastOperatingProfit',
                          'ForecastOrdinaryProfit',	'ForecastProfit', 'CashFlowsFromOperatingActivities',
                          'CashFlowsFromInvestingActivities', 'CashFlowsFromFinancingActivities']
    #合併リストの中身の分だけ処理を繰り返し
    for key, value in codes_to_merge_dict.items():
        #合併前の2社分のデータについて、インデックスを揃える
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
        stock_fin = pd.concat([stock_fin, merged], axis=0).sort_values(['Date', 'Code'])

        return stock_fin


def process_stock_dfs() -> None:
    """
    全ての株式関連データを一括で加工する。
    """
    print("株式データの加工を開始します...")
    process_stock_list()
    process_stock_price()
    process_stock_fin()
    print("全ての株式データの加工が完了しました。")

# %%デバッグ
if __name__ == '__main__':
    process_stock_dfs()
    print(pd.read_parquet(paths.STOCK_FIN_PARQUET))
