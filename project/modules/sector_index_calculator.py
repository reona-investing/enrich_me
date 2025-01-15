#%% モジュールのインポート
import paths
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

#%% 関数群
def calc_new_sector_price(stock_dfs_dict:dict, SECTOR_REDEFINITIONS_CSV:str, SECTOR_INDEX_PARQUET:str) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    セクターインデックスを算出します。
    Args:
        stock_dfs_dict (dict): 'stock_list', 'stock_fin', 'stock_price'が定義されたデータフレーム
        SECTOR_REDEFINITIONS_CSV (str): セクター定義の設定ファイルのパス
        SECTOR_INDEX_PARQUET (str): セクターインデックスを出力するparquetファイルのパス
    Returns:
        pd.DataFrame: セクターインデックスを格納
        pd.DataFrame: 発注用に、個別銘柄の終値と時価総額を格納
    '''
    stock_price = stock_dfs_dict['stock_price']
    stock_fin = stock_dfs_dict['stock_fin']
    #価格情報に発行済み株式数の情報を結合
    stock_price_for_order = calc_marketcap(stock_price, stock_fin)
    #インデックス値の算出
    new_sector_price = _calc_index_value(stock_price_for_order, SECTOR_REDEFINITIONS_CSV)
    #データフレームを保存して、インデックスを設定
    new_sector_price = new_sector_price.reset_index()
    new_sector_price.to_parquet(SECTOR_INDEX_PARQUET)
    new_sector_price = new_sector_price.set_index(['Date', 'Sector'])
    print('セクターのインデックス値の算出が完了しました。')
    #TODO 2つのデータフレームを返す関数は分けたほうがよさそう。

    return new_sector_price, stock_price_for_order


def calc_marketcap(stock_price: pd.DataFrame, stock_fin: pd.DataFrame) -> pd.DataFrame:
    '''
    各銘柄の日ごとの時価総額を算出する。
    Args:
        stock_price (pd.DataFrame): 価格情報
        stok_fin (pd.DataFrame): 財務情報
    Returns:
        pd.DataFrame: 価格情報に時価総額を付記
    '''
    # 価格情報に発行済み株式数の情報を照合
    stock_price_with_shares = _merge_stock_price_and_shares(stock_price, stock_fin)
    # 発行済み株式数の補正係数を算出
    stock_price_cap = _calc_adjustment_factor(stock_price_with_shares, stock_price)
    #print(stock_price_cap[(stock_price_cap['Code']=='2175')&((stock_price_cap['Date']==datetime(2013,12,30))|(stock_price_cap['Date']==datetime(2014,1,6)))])
    stock_price_cap = _adjust_shares(stock_price_cap)
    # 時価総額と指数計算用の補正値を算出
    stock_price_cap = _calc_marketcap(stock_price_cap)
    stock_price_cap = _calc_correction_value(stock_price_cap)
    
    df = stock_price_cap[stock_price_cap['Code']=='9101']
    #df = df[['Date', 'Code', 'Close', 'MarketCapClose']]
    df['CapRet'] = df['MarketCapClose'].pct_change(1)
    df['CloseRet'] = df['Close'].pct_change(1)
    df.to_csv('test.csv')
    
    return stock_price_cap


def _merge_stock_price_and_shares(stock_price: pd.DataFrame, stock_fin: pd.DataFrame) -> pd.DataFrame:
    """
    期末日以降最初の営業日時点での発行済株式数を結合。
    Args:
        stock_price (pd.DataFrame): 価格情報
        stock_fin (pd.DataFrame): 財務情報
    Returns:
        pd.DataFrame: 価格情報に発行済株式数を付記
    """
    business_days = stock_price['Date'].unique()
    shares_df = _calc_shares_at_end_period(stock_fin)
    shares_df = _append_next_period_start_date(shares_df, business_days)
    merged_df = _merge_with_stock_price(stock_price, shares_df)
    return merged_df

def _calc_shares_at_end_period(stock_fin: pd.DataFrame) -> pd.DataFrame:
    """
    期末日時点での発行済株式数を計算する。
    Args:
        stock_fin (pd.DataFrame): 財務情報
    Returns:
        pd.DataFrame: 財務情報に発行済株式数を付記
    """
    shares_df = stock_fin[['Code', 'Date', 'OutstandingShares', 'CurrentPeriodEndDate']].copy()
    shares_df = shares_df.sort_values('Date').drop('Date', axis=1)
    shares_df = shares_df.drop_duplicates(subset=['CurrentPeriodEndDate', 'Code'], keep='last')
    shares_df['NextPeriodStartDate'] = pd.to_datetime(shares_df['CurrentPeriodEndDate']) + timedelta(days=1)
    shares_df['isSettlementDay'] = True #あとで価格情報から抽出するために決算日フラグを設定しておく。
    return shares_df

def _append_next_period_start_date(shares_df: pd.DataFrame, business_days: np.array) -> pd.DataFrame:
    """
    次期開始日を営業日ベースで計算する。
    Args:
        shares_df (pd.DataFrame): 財務情報に発行済株式数を付記
        business_days (np.array): 営業日リスト
    Returns:
        pd.DataFrame: 財務情報に発行済株式数と時期開始日を付記
    """
    shares_df['NextPeriodStartDate'] = shares_df['NextPeriodStartDate'].apply(
        _find_next_business_day, business_days=business_days
    )
    return shares_df

def _find_next_business_day(date:pd.Timestamp, business_days:np.array) -> pd.Timestamp:
    '''
    任意の日付を参照し、翌営業日を探します。
    Args:
        date (pd.Timestamp): 任意の日付
        business_days (np.array): 営業日の一覧
    Returns:
        pd.TimeStamp: 翌営業日の日付
    '''
    if pd.isna(date):
        return date
    while date not in business_days:
        date += np.timedelta64(1, 'D')
    return date

def _merge_with_stock_price(stock_price: pd.DataFrame, shares_df: pd.DataFrame) -> pd.DataFrame:
    """
    価格データに発行済株式数情報を結合する。
    Args:
        stock_price (pd.DataFrame): 価格情報データフレーム
        shares_df (pd.DataFrame): 発行済株式数を含むデータフレーム
    Returns:
        pd.DataFrame: 結合されたデータフレーム
    """
    merged_df = pd.merge(stock_price,
                         shares_df[['Code', 'OutstandingShares', 'NextPeriodStartDate', 'isSettlementDay']],
                         left_on=['Date', 'Code'], 
                         right_on=['NextPeriodStartDate', 'Code'], 
                         how='left'
                         ).drop('NextPeriodStartDate', axis=1)
    merged_df['isSettlementDay'] = merged_df['isSettlementDay'].astype(bool).fillna(False)
    return merged_df


def _calc_adjustment_factor(stock_price_with_shares: pd.DataFrame, stock_price: pd.DataFrame) -> pd.DataFrame:
    """株式分割・併合による発行済み株式数の変化を調整します。
    Args:
        stock_price_with_shares (pd.DataFrame): 価格情報に発行済み株式数を併記
        stock_price (pd.DataFrame): 価格情報
    Returns:
        pd.DataFrame: 調整後の価格情報データフレーム
    """
    stock_price_to_adjust = _extract_rows_to_adjust(stock_price_with_shares)
    stock_price_to_adjust = _calc_shares_rate(stock_price_to_adjust)
    adjusted_stock_price = _correct_shares_rate_for_non_adjustment(stock_price_to_adjust)
    stock_price = _merge_shares_rate(stock_price, adjusted_stock_price)
    stock_price = _handle_special_cases(stock_price)
    return _calc_cumulative_shares_rate(stock_price)

def _extract_rows_to_adjust(stock_price_with_shares_df: pd.DataFrame) -> pd.DataFrame:
    """
    株式分割・併合の対象行を抽出します。
    Args:
        stock_price_with_shares_df (pd.DataFrame): 価格情報に発行済み株式数を併記
    Returns:
        pd.DataFrame: 引数のdfから株式分割・併合対象行のみを抽出
    """
    condition = (
        stock_price_with_shares_df['OutstandingShares'].notnull() |
        (stock_price_with_shares_df['AdjustmentFactor'] != 1)
    )
    return stock_price_with_shares_df.loc[condition].copy()

def _calc_shares_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    株式分割・併合による発行済み株式数の変化率を計算します。
    Args:
        df (pd.DataFrame): 株式分割・併合対象行のデータフレーム
    Returns:
        pd.DataFrame: 発行済み株式数の比率を計算したデータフレーム
    """
    df['OutstandingShares'] = df.groupby('Code')['OutstandingShares'].bfill()
    df['SharesRate'] = (df.groupby('Code')['OutstandingShares'].shift(-1) / df['OutstandingShares']).round(1)
    return df

def _correct_shares_rate_for_non_adjustment(df: pd.DataFrame) -> pd.DataFrame:
    """
    株式分割・併合由来でない発行済み株式数変更（株価補正なし）について、補正比率を1に修正します。
    Args:
        df (pd.DataFrame): 株式分割・併合対象行のデータフレーム
    Returns:
        pd.DataFrame: 調整後のデータフレーム
    """
    # 補正係数の出るタイミングと実際に補正の必要なタイミングが±2データぶんずれる場合がある。
    shift_days = [1, 2, -1, -2]
    shift_columns = [f'Shift_AdjustmentFactor{i}' for i in shift_days]
    for shift_column, i in zip(shift_columns, shift_days):
        df[shift_column] = df.groupby('Code')['AdjustmentFactor'].shift(i).fillna(1)
    df.loc[((df[shift_columns] == 1).all(axis=1) | (df['SharesRate'] == 1)), 'SharesRate'] = 1
    df[df['Code']=='9101'].to_csv('test.csv')
    return df

def _merge_shares_rate(stock_price: pd.DataFrame, df_to_calc_shares_rate: pd.DataFrame) -> pd.DataFrame:
    """
    株価調整用の発行済株式数比率を元の価格情報データフレームに結合。
    Args:
        stock_price (pd.DataFrame): 価格情報
        df_to_calc_shares_rate (pd.DataFrame): 発行済株式数比率を含むデータフレーム
    Returns:
        pd.DataFrame: 価格情報に発行済株式数比率を併記
    """
    df_to_calc_shares_rate = df_to_calc_shares_rate[df_to_calc_shares_rate['isSettlementDay']]
    df_to_calc_shares_rate['SharesRate'] = df_to_calc_shares_rate.groupby('Code')['SharesRate'].shift(1)
    stock_price = pd.merge(
        stock_price,
        df_to_calc_shares_rate[['Date', 'Code', 'OutstandingShares', 'SharesRate']],
        how='left',
        on=['Date', 'Code']
    )
    stock_price['SharesRate'] = stock_price.groupby('Code')['SharesRate'].shift(-1)
    stock_price['SharesRate'] = stock_price['SharesRate'].fillna(1)
    return stock_price

def _handle_special_cases(stock_price: pd.DataFrame) -> pd.DataFrame:
    """
    最初の決算発表データ前の発行済株式数比率を手動で補正します。
    Args:
        stock_price (pd.DataFrame): 価格情報データフレーム
    Returns:
        pd.DataFrame: 補正後のデータフレーム
    """
    stock_price.loc[(stock_price['Code'] == '3064') & (stock_price['Date'] <= datetime(2013, 7, 25)), 'SharesRate'] = 1
    stock_price.loc[(stock_price['Code'] == '6920') & (stock_price['Date'] <= datetime(2013, 8, 9)), 'SharesRate'] = 1
    return stock_price

def _calc_cumulative_shares_rate(stock_price: pd.DataFrame) -> pd.DataFrame:
    """
    発行済み株式数比率の累積積を計算します。
    Args:
        stock_price (pd.DataFrame): 価格情報データフレーム
    Returns:
        pd.DataFrame: 累積積を計算したデータフレーム
    """
    stock_price = stock_price.sort_values('Date', ascending=False)
    stock_price['CumulativeSharesRate'] = stock_price.groupby('Code')['SharesRate'].cumprod()
    stock_price = stock_price.sort_values('Date', ascending=True)
    stock_price['CumulativeSharesRate'] = stock_price['CumulativeSharesRate'].fillna(1)
    return stock_price


def _adjust_shares(df:pd.DataFrame) -> pd.DataFrame:
    '''
    発行済株式数を調整します。
    Args:
        df (pd.DataFrame): 価格情報に補正係数を併記
    Returns:
        pd.DataFrame: 引数のdfの発行済株式数を補正したもの
    '''
    df['OutstandingShares'] = df.groupby('Code', as_index=False)['OutstandingShares'].ffill() #決算発表時以外が欠測値なので、後埋めする。
    df['OutstandingShares'] = df.groupby('Code', as_index=False)['OutstandingShares'].bfill() #初回決算発表以前の分を前埋め。
    df['OutstandingShares'] = df['OutstandingShares'] * df['CumulativeSharesRate'] #株式併合・分割以降、決算発表までの期間の発行済み株式数を調整
    #不要行の削除
    return df.drop(['SharesRate', 'CumulativeSharesRate'], axis=1)


def _calc_marketcap(df: pd.DataFrame) -> pd.DataFrame:
    '''
    時価総額を算出します。
    Args:
        df (pd.DataFrame): 価格情報に発行済株式数（補正済）を併記
    Returns:
        pd.DataFrame: 引数のdfに時価総額のOHLCを追加したもの
    '''
    df['MarketCapOpen'] = df['Open'] * df['OutstandingShares']
    df['MarketCapClose'] = df['Close'] * df['OutstandingShares']
    df['MarketCapHigh'] = df['High'] * df['OutstandingShares']
    df['MarketCapLow'] = df['Low'] * df['OutstandingShares']
    return df


def _calc_correction_value(df: pd.DataFrame) -> pd.DataFrame:
    '''
    指数算出補正値を算出します。
    Args:
        df (pd.DataFrame): 価格情報に時価総額を併記
    Returns:
        pd.DataFrame: 引数のdfに指数算出補正値を併記したもの
    '''
    df['OutstandingShares_forCorrection'] = df.groupby('Code')['OutstandingShares'].shift(1)
    df['OutstandingShares_forCorrection'] = df['OutstandingShares_forCorrection'].fillna(0)
    df['MarketCapClose_forCorrection'] = df['Close'] * df['OutstandingShares_forCorrection']
    df['CorrectionValue'] = df['MarketCapClose'] - df['MarketCapClose_forCorrection']
    return df


def _calc_index_value(stock_price: pd.DataFrame, SECTOR_REDEFINITIONS_CSV:str) -> pd.DataFrame:
    '''
    指標を算出します。
    Args:
        stock_price (pd.DataFrame): 価格情報に時価総額と指数算出補正値を併記
        SECTOR_REDEFINITIONS_CSV (str): セクター定義の設定ファイルのパス
    Returns:
        pd.DataFrame: セクターインデックス
    '''
    new_sector_list = pd.read_csv(SECTOR_REDEFINITIONS_CSV).dropna(how='any', axis=1)
    new_sector_list['Code'] = new_sector_list['Code'].astype(str)
    new_sector_price = pd.merge(new_sector_list, stock_price, how='right', on='Code')

    #必要列を抜き出したデータフレームを作る。
    new_sector_price = new_sector_price.groupby(['Date', 'Sector'])\
      [['MarketCapOpen', 'MarketCapClose','MarketCapHigh', 'MarketCapLow', 'OutstandingShares', 'CorrectionValue']].sum()
    new_sector_price['1d_return'] = new_sector_price['MarketCapClose'] \
                                    / (new_sector_price.groupby('Sector')['MarketCapClose'].shift(1) \
                                    + new_sector_price['CorrectionValue']) - 1
    new_sector_price['1d_rate'] = 1 + new_sector_price['1d_return']

    #初日の終値を1とすると、各日の終値は1d_rateのcumprodで求められる。→OHLは、Cとの比率で求められる。
    new_sector_price['Close'] = new_sector_price.groupby('Sector')['1d_rate'].cumprod()
    new_sector_price['Open'] = new_sector_price['Close']  * new_sector_price['MarketCapOpen'] / new_sector_price['MarketCapClose']
    new_sector_price['High'] = new_sector_price['Close']  * new_sector_price['MarketCapHigh'] / new_sector_price['MarketCapClose']
    new_sector_price['Low'] = new_sector_price['Close']  * new_sector_price['MarketCapLow'] / new_sector_price['MarketCapClose']

    return new_sector_price


#%% デバッグ
if __name__ == '__main__':
    from jquants_api_operations import run_jquants_api_operations
    SECTOR_REDEFINITIONS_CSV = f'{paths.SECTOR_REDEFINITIONS_FOLDER}/48sectors_2024-2025.csv'
    SECTOR_INDEX_PARQUET = f'{paths.SECTOR_REDEFINITIONS_FOLDER}/New48sectors_price_test.parquet'
    list_df, fin_df, price_df = run_jquants_api_operations(
        read = True,
        filter= "(Listing==1)&((ScaleCategory=='TOPIX Core30')|(ScaleCategory=='TOPIX Large70')|(ScaleCategory=='TOPIX Mid400'))"
      )
    stock_dfs_dict = {'stock_list': list_df,
                      'stock_fin': fin_df,
                      'stock_price': price_df}
    new_sector_price, price_for_order = calc_new_sector_price(stock_dfs_dict, SECTOR_REDEFINITIONS_CSV, SECTOR_INDEX_PARQUET)
    print((new_sector_price[['1d_return']].fillna(0).unstack() + 1).cumprod())

    print(new_sector_price[['1d_return']].dropna().sort_values(by='1d_return'))

    '''
    sector = '海運'
    start_date = datetime(2017, 9, 26)
    end_date = datetime(2017, 9, 30)
    
    sector_redefinitions = pd.read_csv(SECTOR_REDEFINITIONS_CSV)
    sector_condition = (price_for_order['Code'].isin(sector_redefinitions.loc[sector_redefinitions['Sector'] == sector, 'Code'].astype(str)))
    duration = (price_for_order['Date'] >= start_date) & (price_for_order['Date'] <= end_date)
    print(price_for_order.loc[sector_condition & duration, ['Date', 'Code', 'MarketCapClose_forCorrection']].set_index(['Code', 'Date']).unstack())
    #print(price_for_order.loc[sector_condition & duration, ['Date', 'Code', 'CorrectionValue']].set_index(['Code', 'Date']).unstack())
    '''
