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
    stock_price_cap = _merge_stock_price_and_shares(stock_price, stock_fin)
    # 発行済み株式数の補正係数を算出
    stock_price_cap = _calc_shares_rate(stock_price_cap, stock_price)
    stock_price_cap = _adjust_shares(stock_price_cap)
    # 時価総額と指数計算用の補正値を算出
    stock_price_cap = _calc_marketcap(stock_price_cap)
    return _calc_correction_value(stock_price_cap)


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
    shares_df['Settlement'] = 1 #TODO この列の追加理由をコメントアウトしたほうがいいかも
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
    return pd.merge(stock_price, 
                    shares_df[['Code', 'OutstandingShares', 'NextPeriodStartDate', 'Settlement']],
                    left_on=['Date', 'Code'], 
                    right_on=['NextPeriodStartDate', 'Code'], 
                    how='left'
                    ).drop('NextPeriodStartDate', axis=1)


def _calc_shares_rate(stock_price_with_shares_df:pd.DataFrame, stock_price:pd.DataFrame) -> pd.DataFrame:
    '''株式分割・併合による発行済み株式数の変化を調整'''
    rows_to_calc_marketcap = \
        (stock_price_with_shares_df['OutstandingShares'].notnull())|(stock_price_with_shares_df['AdjustmentFactor'] != 1)
      #'OutstandingShares'の値が存在（=期初日）+'AdjustmentFactor'が1でない（株式分割・併合日）を時価総額の計算対象とする。
    df_to_calc_shares_rate = stock_price_with_shares_df.loc[rows_to_calc_marketcap].copy() #決算発表日+株式分割・併合日のみ残す。

    #'SharesRate'（前の行との発行済み株式数の比率）を算出する。
    df_to_calc_shares_rate['OutstandingShares'] = df_to_calc_shares_rate.groupby('Code')['OutstandingShares'].bfill()
    df_to_calc_shares_rate['SharesRate'] = \
      df_to_calc_shares_rate.groupby('Code')['OutstandingShares'].shift(-1) / df_to_calc_shares_rate['OutstandingShares'] #前の行との比率を算出
    df_to_calc_shares_rate['SharesRate'] = df_to_calc_shares_rate['SharesRate'].round(1) #小数第一位までで四捨五入

    #'AdjustmentFactor'が1でない行（株式分割・併合日）の上下2行に'SharesRate'と同じ値がない=発行株数の変化は株式分割・併合によるものではないとして補正しない
    shift_days = [1, 2, -1, -2]
    shift_columns = [f'Shift_AdjustmentFactor{i}' for i in shift_days]
    for shift_column, i in zip(shift_columns, shift_days):
        df_to_calc_shares_rate[shift_column] = df_to_calc_shares_rate.groupby('Code')['AdjustmentFactor'].shift(i).fillna(1)
    df_to_calc_shares_rate.loc[(
        (df_to_calc_shares_rate[shift_columns] == 1).all(axis=1)) |
        (df_to_calc_shares_rate['SharesRate'] == 1), 'SharesRate'] = 1

    #決算日の行のみに絞り、もとの価格情報データフレームと再結合する
    df_to_calc_shares_rate = df_to_calc_shares_rate[df_to_calc_shares_rate['Settlement']==1]
    df_to_calc_shares_rate['SharesRate'] = df_to_calc_shares_rate.groupby('Code')['SharesRate'].shift(1)
    stock_price = pd.merge(stock_price, df_to_calc_shares_rate[['Date', 'Code', 'OutstandingShares', 'SharesRate']], how='left', on=['Date', 'Code'])
    stock_price['SharesRate'] = stock_price.groupby('Code')['SharesRate'].shift(-1)
    stock_price['SharesRate'] = stock_price['SharesRate'].fillna(1)

    #以下の2銘柄は初回財務情報発表前に株式分割・併合が行われているため、その期間をすべて1で埋める
    stock_price.loc[(stock_price['Code']=='3064')&(stock_price['Date']<=datetime(2013,7,25)), 'SharesRate'] = 1
    stock_price.loc[(stock_price['Code']=='6920')&(stock_price['Date']<=datetime(2013,8,9)), 'SharesRate'] = 1

    #SharesRateの累積積を求める
    stock_price = stock_price.sort_values('Date', ascending=False) #累積積を求めるために、データを降順に並べ替える
    stock_price['CumulativeSharesRate'] = stock_price.groupby('Code')['SharesRate'].cumprod() #累積積を求める
    stock_price = stock_price.sort_values('Date', ascending=True) #元の昇順に戻す
    stock_price['CumulativeSharesRate'] = stock_price['CumulativeSharesRate'].fillna(1) #欠損値を1で埋める。

    return stock_price


def _adjust_shares(stock_price:pd.DataFrame) -> pd.DataFrame:
    '''発行済み株式数の調整'''
    stock_price['OutstandingShares'] = stock_price.groupby('Code', as_index=False)['OutstandingShares'].ffill() #決算発表時以外が欠測値なので、後埋めする。
    stock_price['OutstandingShares'] = stock_price.groupby('Code', as_index=False)['OutstandingShares'].bfill() #初回決算発表以前の分を前埋め。
    stock_price['OutstandingShares'] = stock_price['OutstandingShares'] * stock_price['CumulativeSharesRate'] #株式併合・分割以降、決算発表までの期間の発行済み株式数を調整
    #不要行の削除
    stock_price = stock_price.drop(['SharesRate', 'CumulativeSharesRate'], axis=1)

    return stock_price


def _calc_marketcap(stock_price: pd.DataFrame) -> pd.DataFrame:
    '''時価総額を算出'''
    stock_price['MarketCapOpen'] = stock_price['Open'] * stock_price['OutstandingShares']
    stock_price['MarketCapClose'] = stock_price['Close'] * stock_price['OutstandingShares']
    stock_price['MarketCapHigh'] = stock_price['High'] * stock_price['OutstandingShares']
    stock_price['MarketCapLow'] = stock_price['Low'] * stock_price['OutstandingShares']

    return stock_price


def _calc_correction_value(stock_price: pd.DataFrame) -> pd.DataFrame:
    '''指数計算用の補正値を算出'''
    stock_price['OutstandingShares_forCorrection'] = stock_price.groupby('Code')['OutstandingShares'].shift(1)
    stock_price['OutstandingShares_forCorrection'] = stock_price['OutstandingShares_forCorrection'].fillna(0)
    stock_price['MarketCapClose_forCorrection'] = stock_price['Close'] * stock_price['OutstandingShares_forCorrection']
    stock_price['CorrectionValue'] = stock_price['MarketCapClose'] - stock_price['MarketCapClose_forCorrection']

    return stock_price


def _calc_index_value(stock_price: pd.DataFrame, SECTOR_REDEFINITIONS_CSV:str) -> pd.DataFrame:
    '''指標の算出'''
    new_sector_list = pd.read_csv(SECTOR_REDEFINITIONS_CSV).dropna(how='any', axis=1)
    new_sector_list['Code'] = new_sector_list['Code'].astype(str)

    #必要列を抜き出したデータフレームを作る。
    new_sector_price = pd.merge(new_sector_list, stock_price, how='right', on='Code')
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
    #new_sector_price = new_sector_price[['Open', 'Close', 'High', 'Low']].dropna(how='all', axis=0)

    return new_sector_price


def filter_stocks(stock_dfs_dict:dict, filter:str): #対象銘柄の抜き取り
    '''【対象銘柄の抜き取り】'''
    stock_list = stock_dfs_dict['stock_list']
    stock_price = stock_dfs_dict['stock_price']
    stock_fin = stock_dfs_dict['stock_fin']
    #フィルターの設定
    filtered_code_list = stock_list.query(filter)['Code'].astype(str).values
    #dfからの抜き取り
    stock_dfs_dict['stock_list'] = stock_list[stock_list['Code'].astype(str).isin(filtered_code_list)]
    stock_dfs_dict['stock_fin'] = stock_fin[stock_fin['Code'].astype(str).isin(filtered_code_list)]
    stock_dfs_dict['stock_price'] = stock_price[stock_price['Code'].astype(str).isin(filtered_code_list)]
    return stock_dfs_dict



#%% デバッグ
if __name__ == '__main__':
    from IPython.display import display
    from jquants_api_operations import run_jquants_api_operations
    SECTOR_REDEFINITIONS_CSV = f'{paths.SECTOR_REDEFINITIONS_FOLDER}/New48sectors_list.csv'
    SECTOR_INDEX_PARQUET = f'{paths.SECTOR_REDEFINITIONS_FOLDER}/New48sectors_price.pkl.gz'
    list_df, fin_df, price_df = run_jquants_api_operations(filter= \
      "(Listing==1)&((ScaleCategory=='TOPIX Core30')|(ScaleCategory=='TOPIX Large70')|(ScaleCategory=='TOPIX Mid400'))"
      )
    stock_dfs_dict = {'stock_list': list_df,
                      'stock_fin': fin_df,
                      'stock_price': price_df}
    new_sector_price, price_for_order = calc_new_sector_price(stock_dfs_dict, SECTOR_REDEFINITIONS_CSV, SECTOR_INDEX_PARQUET)
    display(new_sector_price)