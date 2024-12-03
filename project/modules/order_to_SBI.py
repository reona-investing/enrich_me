#%% モジュールのインポート
import jquants_api_fetcher as fetcher #JQuantsAPIでのデータ取得
import SBI
import paths

import shutil
import pandas as pd
from typing import Tuple
from datetime import datetime
from IPython.display import display
import nodriver as uc
import asyncio

#%% get_unitのサブ関数
def _get_ideal_costs(df:pd.DataFrame, maxcost:int) -> pd.DataFrame:
    '''
    以下の3つのコストを算出。
    IdealCost：完全にインデックス通りの銘柄配分となる理想コストを算出
    MaxCostWithinIdeal：理想コスト以下の最大コスト
    MinCostExceedingIdeal：理想コスト以上の最小コスト
    '''
    #完全にインデックス通りの銘柄配分となる理想コストを算出。
    df = df.sort_values('Weight', ascending=False).reset_index(drop=True).copy()
    df['IdealCost'] = (df['Weight'] * maxcost).astype(int)
    #理想コスト以下の最大コストと、理想コスト以上の最小コストを算出。
    df['MaxUnitWithinIdeal'] = df['IdealCost'] // df['EstimatedCost']
    df['MaxCostWithinIdeal'] = df['EstimatedCost'] * df['MaxUnitWithinIdeal']
    df['MinCostExceedingIdeal'] = df['MaxCostWithinIdeal'] + df['EstimatedCost']
    return df


def _draft_portfolio(df:pd.DataFrame) -> pd.DataFrame:
    '''
    予算を無視して、理想コストに最も近い購入単位数を算出。
    '''
    #WithinとOverで、どちらが理想コストに近いかを算出。
    df['ReductionRate'] = abs(df['IdealCost'] - df['MaxCostWithinIdeal']) / abs(df['IdealCost'] - df['MinCostExceedingIdeal'])

    #各行について、最適な単位数（単位：100株）を割り出す。
    for index, row in df.iterrows():
        #理想コストに近いほうをトータルコストとしていったん格納。
        if row['ReductionRate'] <= 1:
            df.loc[index, 'TotalCost'] = row['MaxCostWithinIdeal']
        else:
            df.loc[index, 'TotalCost'] = row['MinCostExceedingIdeal']
        #理想コストから購入単位数を逆算
        df.loc[index, 'Unit'] = df.loc[index, 'TotalCost'] / row['EstimatedCost']
    return df


def _reduce_units(df:pd.DataFrame, maxcost:int) -> pd.DataFrame:
    '''
    金額がオーバーしているとき、購入数量を減らす。
    購入対象銘柄の購入コストが全て理想コスト以内，かつトータルコストが予算内に収まるまで繰り返し。
    ReductionRate：大きいほど優先的に購入数量を減らす。小さいほど優先的に購入数量を増やす。
    '''
    #候補銘柄の抜き取り
    candidate_tickers_df = \
          df.loc[(df['ReductionRate'] >= 1) & (df['Unit'] != 0), :].sort_values('ReductionRate', ascending=False)
    # 'MaxUnit'が存在する場合、'Unit'を制限
    for i in range(len(candidate_tickers_df)):
        if 'MaxUnit' in df.columns:
            index_to_reduce = candidate_tickers_df.index[i]
            if df.loc[index_to_reduce, 'Unit'] > df.loc[index_to_reduce, 'MaxUnit']:
                df.loc[index_to_reduce, 'Unit'] = df.loc[index_to_reduce, 'MaxUnit']
    # 予算オーバーの場合、予算内に収まるまで購入数量を減らす。
    for i in range(len(candidate_tickers_df)):
        if df['TotalCost'].sum() <= maxcost:
            break # トータルコストが予算内に収まったら繰り返し終了
        #ReductionRateの値が最も大きい銘柄について、購入数量を1単位減らす。
        index_to_reduce = candidate_tickers_df.index[i]
        df.loc[index_to_reduce, 'TotalCost'] -= df.loc[index_to_reduce, 'EstimatedCost']
        df.loc[index_to_reduce, 'Unit'] -= 1
    return df


def _increase_units(df:pd.DataFrame, maxcost:int) -> pd.DataFrame:
    '''
    maxcostギリギリまで購入数量を増やす。
    diff_within < diff_overかつ、差が最小になる場合を判定したい。
    '''
    while True:
        free_cost = maxcost - df['TotalCost'].sum() #余裕コスト（最大これだけ増やせる）
        #購入数量を増やす銘柄を選定
        filtered_df = df[df['EstimatedCost'] <= free_cost]
        if filtered_df.empty:
            break #購入数量を増やせる銘柄がなくなったら繰り返し終了。
        min_rate_row = filtered_df['ReductionRate'].idxmin()
        #購入数量とコスト、ReductionRateを更新
        df.loc[min_rate_row, ['TotalCost', 'MaxCostWithinIdeal', 'MinCostExceedingIdeal']] += df.loc[min_rate_row, 'EstimatedCost']
        df.loc[min_rate_row, 'Unit'] += 1
        df.loc[min_rate_row, 'ReductionRate'] = \
          abs(df.loc[min_rate_row, 'IdealCost'] - df.loc[min_rate_row, 'MaxCostWithinIdeal']) / \
            abs(df.loc[min_rate_row, 'IdealCost'] - df.loc[min_rate_row, 'MinCostExceedingIdeal'])
        #print(f'現在の購入金額：{df["TotalCost"].sum()}円')

    return df


def _get_unit(df:pd.DataFrame, maxcost:int) -> pd.DataFrame:
    '''
    売買銘柄とその株数を算出する関数
    df：売買対象の銘柄を格納したデータフレーム
    maxcost：最大コスト
    '''

    df = _get_ideal_costs(df, maxcost) #各種コストを算出
    df = _draft_portfolio(df) #仮ポートフォリオ（最もインデックスに近い購入単位数）を作成
    df = _reduce_units(df, maxcost)
    df = _increase_units(df, maxcost)
    df = df.drop(['MaxUnitWithinIdeal', 'MaxCostWithinIdeal',	'MinCostExceedingIdeal', 'ReductionRate'], axis=1)

    return df

#%% 発注関連のサブ関数
def _get_cost_and_volume_df(price_for_order_df:pd.DataFrame) -> pd.DataFrame:
    '''終値時点での「単位株の購入価格」、「過去5営業日の出来高平均」、「時価総額」を算出'''
    price_for_order_df['Past5daysVolume'] = price_for_order_df.groupby('Code')['Volume'].rolling(5).mean().reset_index(level=0, drop=True)
    price_for_order_df['EstimatedCost'] = price_for_order_df['Close'] * 100
    cost_and_volume_df = price_for_order_df.loc[price_for_order_df['Date']==price_for_order_df['Date'].iloc[-1], :]
    cost_and_volume_df = cost_and_volume_df[['Code', 'EstimatedCost', 'Past5daysVolume', 'MarketCapClose']].reset_index(drop=True)

    return cost_and_volume_df


def _get_weight_df(new_sector_list_df:pd.DataFrame, cost_and_volume_df:pd.DataFrame) -> pd.DataFrame:
    '''インデックス算出のための銘柄ごとのウエイトを算出する。'''
    new_sector_list_df['Code'] = new_sector_list_df['Code'].astype(str)
    weight_df = pd.merge(new_sector_list_df[['Code', 'CompanyName', 'Sector']], cost_and_volume_df, how='right', on='Code')
    weight_df['Code'] = weight_df['Code'].astype(str)
    weight_df = weight_df[weight_df['Sector'].notnull()]
    sum_values = weight_df.groupby('Sector')['MarketCapClose'].sum()
    weight_df['Weight'] = weight_df.apply(lambda row: row['MarketCapClose'] / sum_values[row['Sector']], axis=1)
    weight_df['Weight'] = weight_df['Weight'].fillna(0)
    return weight_df


def _get_todays_pred_df(y_test_df:pd.DataFrame) -> pd.DataFrame:
    '''最新日の予測結果dfを抽出'''
    y_test_df = y_test_df.reset_index().set_index('Date', drop=True)
    todays_pred_df = y_test_df[y_test_df.index==y_test_df.index[-1]].sort_values('Pred')
    todays_pred_df['Rank'] = todays_pred_df['Pred'].rank(ascending=False).astype(int)
    return todays_pred_df


async def _get_tradable_dfs(new_sector_list_df:pd.DataFrame, tab:uc.core.tab.Tab=None) -> Tuple[uc.core.tab.Tab, pd.DataFrame, pd.DataFrame]:
    '''売買それぞれについて、可能な銘柄を算出する。'''
    tab = await SBI.sign_in(tab)
    tab, buyable_dict, sellable_dict = await SBI.get_trade_possibility(tab)
    buyable_tickers_df = new_sector_list_df[new_sector_list_df['Code'].isin(buyable_dict.keys())].copy()
    sellable_tickers_df = new_sector_list_df[new_sector_list_df['Code'].isin(sellable_dict.keys())].copy()
    buyable_tickers_df['Code'] = buyable_tickers_df['Code'].astype(str)
    sellable_tickers_df['Code'] = sellable_tickers_df['Code'].astype(str)
    sellable_tickers_df['MaxUnit'] = sellable_tickers_df['Code'].map(sellable_dict)
    return tab, buyable_tickers_df, sellable_tickers_df


def _append_tradability(todays_pred_df:pd.DataFrame, new_sector_list_df:pd.DataFrame,
                      buyable_tickers_df:pd.DataFrame, sellable_tickers_df:pd.DataFrame) -> pd.DataFrame:
    '''売買それぞれについて、可能な業種一覧を追加'''
    #売買可能業種数と、全体の業種数を算出
    buy = buyable_tickers_df.groupby('Sector')[['Code']].count().rename(columns={'Code':'Buy'}).reset_index()
    sell = sellable_tickers_df.groupby('Sector')[['Code']].count().rename(columns={'Code':'Sell'}).fillna(0).astype(int).reset_index()
    total = new_sector_list_df.groupby('Sector')[['Code']].count().rename(columns={'Code':'Total'}).fillna(0).astype(int).reset_index()
    #上記の内容をdfに落とし込む
    todays_pred_df = pd.merge(todays_pred_df, buy, how='left', on='Sector')
    todays_pred_df = pd.merge(todays_pred_df, sell, how='left', on='Sector')
    todays_pred_df = pd.merge(todays_pred_df, total, how='left', on='Sector')
    todays_pred_df[['Buy', 'Sell']] = todays_pred_df[['Buy', 'Sell']].fillna(0).astype(int)
    #半分以上の銘柄が売買可能な時，売買可能業種とする．
    todays_pred_df['Buyable'] = (todays_pred_df['Buy'] / todays_pred_df['Total'] >= 0.5).astype(int)
    todays_pred_df['Sellable'] = (todays_pred_df['Sell'] / todays_pred_df['Total'] >= 0.5).astype(int)
    return todays_pred_df


def _determine_sectors_to_trade(todays_pred_df:pd.DataFrame,
                                trading_sector_num:int=3,
                                candidate_sector_num:int=5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    売買対象の業種を決定
    todays_pred_df：直近の日の予測結果df
    trading_sector_num：上位・下位いくつの業種を最終的な売買対象とするか？
    trading_sector_num：上位・下位いくつの業種を売買"候補"とするか？
    '''
    #売買可能な上位・下位3業種を抽出
    sell_sectors_df = todays_pred_df.iloc[:candidate_sector_num].copy()
    sell_sectors_df = sell_sectors_df[sell_sectors_df['Sellable']==1].iloc[:trading_sector_num]
    buy_sectors_df = todays_pred_df.iloc[-candidate_sector_num:].copy()
    buy_sectors_df = buy_sectors_df[buy_sectors_df['Buyable']==1].iloc[-trading_sector_num:]

    print('本日のロング予想業種')
    display(buy_sectors_df)
    print('--------------------------------------------')
    print('本日のショート予想業種')
    display(sell_sectors_df)
    print('--------------------------------------------')
    return buy_sectors_df, sell_sectors_df


def _determine_whether_ordering_ETF(buy_sectors_df:pd.DataFrame, sell_sectors_df:pd.DataFrame) -> Tuple[int, int, int]:
    '''
    売買する業種数を判定し、不足分をETFで補う
    sectors_to_trade_num：売買する業種数
    buy_adjuster, sell_adjuster：何業種分をETFで置き換えるか？
    '''
    #buy_sectors_df, sell_sectors_dfのうち大きいほうの値をsectorsに格納
    sectors_to_trade_num = max(len(buy_sectors_df), len(sell_sectors_df))

    #buy_sector、sell_sectorがともに0の場合、何もしない
    if sectors_to_trade_num == 0:
        print('売買できる業種がありませんでした。')
    else:
        should_set_buy_adjuster = len(sell_sectors_df) > len(buy_sectors_df)
        if should_set_buy_adjuster:
            buy_adjuster_num = len(sell_sectors_df) - len(buy_sectors_df)
        else:
            buy_adjuster_num = 0
        should_set_sell_adjuster = len(buy_sectors_df) > len(sell_sectors_df)
        if should_set_sell_adjuster:
            sell_adjuster_num = len(buy_sectors_df) - len(sell_sectors_df)
        else:
            sell_adjuster_num = 0

    return sectors_to_trade_num, buy_adjuster_num, sell_adjuster_num


def _get_tickers_to_trade_df(new_sector_list_df:pd.DataFrame,
                             sectors_to_trade_df:pd.DataFrame,
                             tradable_tickers_df:pd.DataFrame,
                             weight_df:pd.DataFrame) -> pd.DataFrame:
    '''売買それぞれの対象業種を抽出する'''
    tickers_to_trade_df = new_sector_list_df[new_sector_list_df['Sector'].isin(sectors_to_trade_df['Sector'])].copy()
    tickers_to_trade_df = pd.merge(tickers_to_trade_df, sectors_to_trade_df[['Sector', 'Rank']])
    tickers_to_trade_df['Code'] = tickers_to_trade_df['Code'].astype(str)
    tickers_to_trade_df = pd.merge(tickers_to_trade_df, weight_df[['Code', 'Weight', 'EstimatedCost']], on='Code', how='left')
    tickers_to_trade_df = tickers_to_trade_df[tickers_to_trade_df['Code'].isin(tradable_tickers_df['Code'])]
    if 'MaxUnit' in tradable_tickers_df.columns:
        tickers_to_trade_df = pd.merge(tickers_to_trade_df, tradable_tickers_df[['Code', 'MaxUnit']], how='left', on='Code')
    tickers_to_trade_df['Weight'] = tickers_to_trade_df['Weight'] / tickers_to_trade_df.groupby('Sector')['Weight'].transform('sum')
    return tickers_to_trade_df


def _calculate_ETF_orders(long_df:pd.DataFrame, short_df:pd.DataFrame,
                          buy_adjuster_num:int, sell_adjuster_num:int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if max(buy_adjuster_num, sell_adjuster_num):
        if buy_adjuster_num != 0:
            bull_list = fetcher.cli.get_list(code='1308') #上場インデックスファンドTOPIX
            bull_price = fetcher.cli.get_prices_daily_quotes(code='1308')
            for i in range (0, buy_adjuster_num):
                bull_list['Code'] = bull_list['Code'].astype(str)[:4]
                bull_list = bull_list[['Code', 'CompanyName', 'Sector33CodeName']]
                bull_list['Sector'] = f'ETF{i}'
                bull_list['Weight'] = 1
                bull_list['EstimatedCost'] = bull_price['Close'].iat[-1] * 10
                long_df = pd.concat([long_df, bull_list], axis=0)
        if sell_adjuster_num != 0:
            bear_list = fetcher.cli.get_list(code='1356') #TOPIXベア2倍上場投信
            bear_price = fetcher.cli.get_prices_daily_quotes(code='1356')
            for i in range (0, sell_adjuster_num):
                bear_list['Code'] = bear_list['Code'].astype(str)[:4]
                bear_list = bear_list[['Code', 'CompanyName', 'Sector33CodeName']]
                bear_list['Sector'] = f'ETF{i}'
                bear_list['Weight'] = 0.5 #ベア2倍のため、購入単位を半分にする
                bear_list['EstimatedCost'] = bear_price['Close'].iat[-1] * 10
                short_df = pd.concat([short_df, bear_list], axis=0)
    long_df = long_df.dropna(subset=['Weight', 'EstimatedCost'])
    short_df = short_df.dropna(subset=['Weight', 'EstimatedCost'])

    return long_df, short_df


async def _determine_orders(long_df:pd.DataFrame, short_df:pd.DataFrame, 
                            sectors_to_trade_num:int, top_slope: float, 
                            tab:uc.core.tab.Tab=None) -> Tuple[uc.core.tab.Tab, pd.DataFrame, pd.DataFrame]:
    '''発注銘柄と発注単位の算出'''
    #業種ごとの発注限度額の算出
    tab = await SBI.sign_in(tab)
    tab, margin_buying_power, _ = await SBI.get_buying_power(tab)
    maxcost_per_sector = margin_buying_power / (sectors_to_trade_num * 2)
    #トップ業種に傾斜をつけない場合
    if top_slope == 1:
        print(f'業種ごとの発注限度額：{maxcost_per_sector}円')
        long_df = long_df.groupby('Sector').apply(_get_unit, maxcost=maxcost_per_sector).reset_index(drop=True)
        short_df = short_df.groupby('Sector').apply(_get_unit, maxcost=maxcost_per_sector).reset_index(drop=True)
    #トップ業種に傾斜をつける場合
    else:
        maxcost_top_sector = maxcost_per_sector * top_slope
        maxcost_except_top = maxcost_per_sector * (1 - (top_slope - 1) / (sectors_to_trade_num - 1))
        print('業種ごとの発注限度額')
        print(f'トップ業種{maxcost_top_sector}円')
        print(f'トップ以外{maxcost_except_top}円')
        long_top = long_df[long_df['Rank']==long_df['Rank'].min()]
        long_except_top = long_df[long_df['Rank']!=long_df['Rank'].min()]
        long_top = _get_unit(long_top, maxcost=maxcost_top_sector)
        long_except_top = long_except_top.groupby('Sector').apply(_get_unit, maxcost=maxcost_except_top).reset_index(drop=True)
        long_df = pd.concat([long_top, long_except_top], axis=0).reset_index(drop=True)
        short_top = short_df[short_df['Rank']==short_df['Rank'].max()]
        short_except_top = short_df[short_df['Rank']!=short_df['Rank'].max()]
        short_top = short_top.groupby('Sector').apply(_get_unit, maxcost=maxcost_top_sector).reset_index(drop=True)
        short_except_top = short_except_top.groupby('Sector').apply(_get_unit, maxcost=maxcost_except_top).reset_index(drop=True)
        short_df = pd.concat([short_top, short_except_top], axis=0).reset_index(drop=True)


    #発注量の算出
    long_orders = long_df[long_df['Unit']!=0].reset_index(drop=True) #買いの銘柄一覧
    short_orders = short_df[short_df['Unit']!=0].reset_index(drop=True) #売りの銘柄一覧
    #算出結果の表示
    pd.set_option('display.max_rows', None)
    print('ロング予想銘柄一覧')
    print(f'予想額：{long_orders["TotalCost"].sum()}円')
    display(long_orders)
    print('---------------------------')
    print('ショート予想銘柄一覧')
    print(f'予想額：{short_orders["TotalCost"].sum()}円')
    display(short_orders)
    pd.set_option('display.max_rows', 10)
    return tab, long_orders, short_orders

#%% 取引結果取得関連のサブ関数
async def _update_trade_history(trade_history_path: str, sector_list_path: str, 
                                tab:uc.core.tab.Tab=None) -> Tuple[uc.core.tab.Tab, pd.DataFrame]:
    '''
    当日の取引結果を取得し、trade_historyのファイルを更新します。
    trade_history_path：過去の取引履歴を記録したdfのファイルパス
    sector_list：銘柄とセクターとの対応を記録したdf
    '''
    print('取引履歴を更新します。')
    # 取引履歴の更新
    trade_history = pd.read_csv(trade_history_path)
    sector_list = pd.read_csv(sector_list_path)
    trade_history['日付'] = pd.to_datetime(trade_history['日付']).dt.date # 後で同じ変換をするが、この処理いる？
    tab = await SBI.sign_in(tab)
    tab, today_contracts = await SBI.fetch_today_contracts(tab, sector_list)
    trade_history = pd.concat([trade_history, today_contracts], axis=0).reset_index(drop=True)
    trade_history['日付'] = pd.to_datetime(trade_history['日付']).dt.date
    trade_history = trade_history.sort_values(['日付', '売or買', '業種', '銘柄コード']).reset_index(drop=True)

    trade_history.to_csv(trade_history_path, index=False)
    trade_history = pd.read_csv(trade_history_path).drop_duplicates(keep='last') # なぜか読み込み直さないとdrop_duplicatesが効かない（データ型の問題？要検討）
    trade_history['日付'] = pd.to_datetime(trade_history['日付']).dt.date
    trade_history.to_csv(trade_history_path, index=False)
    shutil.copy(trade_history_path, paths.TRADE_HISTORY_BACKUP)

    return tab, trade_history


async def _update_buying_power_history(buying_power_history_path: str, trade_history: pd.DataFrame, 
                                       tab:uc.core.tab.Tab=None) -> Tuple[uc.core.tab.Tab, pd.DataFrame]:
    '''
    実行時点での買付余力を取得し、dfに反映します。
    buying_power_history_path：買付余力の推移を記録したdfのファイルパス
    '''
    buying_power_history = pd.read_csv(buying_power_history_path, index_col=None)
    buying_power_history['日付'] = pd.to_datetime(buying_power_history['日付']).dt.date

    # 買付余力の取得
    tab = await SBI.sign_in(tab)
    tab, _, buying_power = await SBI.get_buying_power(tab)
    # 今日の日付の行がなければ追加、あれば更新
    today = datetime.today().date()
    if buying_power_history[buying_power_history['日付'] == today].empty:
        new_row = pd.DataFrame([[today, buying_power]], columns=['日付', '買付余力'])
        buying_power_history = pd.concat([buying_power_history, new_row], axis=0).reset_index(drop=True)
    else:
        buying_power_history.loc[buying_power_history['日付']==today, '買付余力'] = buying_power

    # 取引がなかった日の行を除去する。
    days_traded = trade_history['日付'].unique()
    did_trade = buying_power_history['日付'].isin(days_traded)
    buying_power_history = buying_power_history[did_trade]
    buying_power_history = buying_power_history.set_index('日付', drop=True)

    print('買付余力の履歴')
    display(buying_power_history.tail(5))
    buying_power_history.to_csv(buying_power_history_path)
    shutil.copy(buying_power_history_path, paths.BUYING_POWER_HISTORY_BACKUP)

    return tab, buying_power_history


async def _update_deposit_history(deposit_history_df_path: str, buying_power_history: pd.DataFrame, 
                                  tab:uc.core.tab.Tab=None) -> Tuple[uc.core.tab.Tab, pd.DataFrame]:
    '''
    総入金額を算出します。
    '''
    #入出金明細と現物の売買をスクレイピングして、当日以降の「譲渡益税源泉徴収金」と「譲渡益税還付金」を除いた入出金分を加減する。
    deposit_history_df = pd.read_csv(deposit_history_df_path)
    deposit_history_df['日付'] = pd.to_datetime(deposit_history_df['日付']).dt.date
    deposit_history_df = deposit_history_df.set_index('日付', drop=True)
    # 当日の入出金履歴をとる
    tab = await SBI.sign_in(tab)
    tab, in_out_df = await SBI.fetch_in_out(tab)
    if in_out_df is None:
        capital_diff = 0
    else:
        in_out_df = in_out_df[(in_out_df['日付']>buying_power_history.index[-2])&(in_out_df['日付']<=buying_power_history.index[-1])]
        capital_diff = in_out_df['入出金額'].sum()
    # 現物の売買による資金の増減をとる
    tab, spots_df = await SBI.fetch_today_spots(tab) #現物の売買
    if spots_df is None:
        spots_diff = 0
    else:
        spots_diff = spots_df['買付余力増減'].sum()

    # 最新日のデータがすでに存在する場合は置換、存在しない場合は追加
    if deposit_history_df.index[-1] != buying_power_history.index[-1]:
        deposit_history_df.loc[buying_power_history.index[-1], '総入金額'] = deposit_history_df.loc[deposit_history_df.index[-1], '総入金額'] + capital_diff + spots_diff
    else:
        deposit_history_df.loc[buying_power_history.index[-1], '総入金額'] = deposit_history_df.loc[deposit_history_df.index[-2], '総入金額'] + capital_diff + spots_diff
    deposit_history_df = deposit_history_df.astype(int)
    print('総入金額の履歴')
    display(deposit_history_df.tail(5))
    deposit_history_df.to_csv(deposit_history_df_path)
    shutil.copy(deposit_history_df_path, paths.DEPOSIT_HISTORY_BACKUP)

    return tab, deposit_history_df


def _show_latest_result(trade_history: pd.DataFrame) -> tuple[str, float]:
    date = trade_history['日付'].iloc[-1]
    amount = trade_history.loc[trade_history['日付'] == date, '利益（税引前）'].sum()
    rate = amount / trade_history.loc[trade_history['日付'] == date, '取得価格'].sum()
    amount = "{:,.0f}".format(amount)
    print(f'{date.strftime("%Y-%m-%d")}： 利益（税引前）{amount}円（{round(rate * 100, 3)}%）')
    return amount, rate

#%% メイン関数
async def select_stocks(order_price_df:pd.DataFrame, new_sector_list_csv:str, y_test_df:pd.DataFrame,
                  trading_sector_num:int, candidate_sector_num:int, top_slope:float = 1.0, 
                  tab:uc.core.tab.Tab=None) -> Tuple[uc.core.tab.Tab, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    売買対象銘柄を選択する。
    price_for_order_df：
    new_sector_list_df：
    y_test_df：
    trading_sector_num：最終的な売買対象業種数
    candidate_sector_num：売買候補とする業種数
    '''
    # new_sector_listを読み込む
    new_sector_list_df = pd.read_csv(new_sector_list_csv)
    # 終値時点での「単位株の購入価格」、「過去5営業日の出来高平均」、「時価総額」を算出
    cost_and_volume_df = _get_cost_and_volume_df(order_price_df)
    # インデックス算出のための銘柄ごとのウエイトを算出する。
    weight_df = _get_weight_df(new_sector_list_df, cost_and_volume_df)
    # 最新日の予測結果をdfとして抽出
    todays_pred_df = _get_todays_pred_df(y_test_df)
    # 売買それぞれについて、可能な銘柄をdfとして抽出
    tab, buyable_tickers_df, sellable_tickers_df = await _get_tradable_dfs(new_sector_list_df, tab)
    # その業種の半分以上の銘柄が売買可能であれば、売買可能業種に設定
    todays_pred_df = _append_tradability(todays_pred_df, new_sector_list_df, buyable_tickers_df, sellable_tickers_df)
    # 売買対象の業種をdfとして抽出
    buy_sectors_df, sell_sectors_df = _determine_sectors_to_trade(todays_pred_df, trading_sector_num, candidate_sector_num)
    # 売買対象銘柄をdfとして抽出
    long_df = _get_tickers_to_trade_df(new_sector_list_df, buy_sectors_df, buyable_tickers_df, weight_df)
    short_df = _get_tickers_to_trade_df(new_sector_list_df, sell_sectors_df, sellable_tickers_df, weight_df)
    # 対象業種数が売買で異なる場合、何業種分をETFで補うかを算出
    sectors_to_trade_num, buy_adjuster_num, sell_adjuster_num = _determine_whether_ordering_ETF(buy_sectors_df, sell_sectors_df)
    # ETFの必要注文数を算出
    long_df, short_df = _calculate_ETF_orders(long_df, short_df, buy_adjuster_num, sell_adjuster_num)
    # 注文する銘柄と注文単位数を算出
    tab, long_orders, short_orders = await _determine_orders(long_df, short_df, sectors_to_trade_num, top_slope, tab)
    # Long, Shortそれぞれの累積コストを算出
    long_orders['CumCost_byLS'] = long_orders['TotalCost'].cumsum()
    short_orders['CumCost_byLS'] = short_orders['TotalCost'].cumsum()
    # Long, Shortの選択銘柄をCSVとして出力しておく
    long_orders.to_csv(paths.LONG_ORDERS_CSV, index=False)
    short_orders.to_csv(paths.SHORT_ORDERS_CSV, index=False)
    # Google Drive上にバックアップ
    shutil.copy(paths.LONG_ORDERS_CSV, paths.LONG_ORDERS_BACKUP)
    shutil.copy(paths.SHORT_ORDERS_CSV, paths.SHORT_ORDERS_BACKUP)

    return tab, long_orders, short_orders, todays_pred_df

async def _make_orders(orders_df, nariyuki_value, tab):
    failed_order_list = []
    failed_tickers = []
    for ticker, unit, L_or_S in zip(orders_df['Code'], orders_df['Unit'], orders_df['LorS']):
        unit = int(unit * 100)
        if L_or_S == 'Long':
            trade_type = '信用新規買'
        elif L_or_S == 'Short':
            if ticker == '1356': #TOPIXベア2倍上場投信の場合は買いポジションを取る
                trade_type = '信用新規買'
            else:
                trade_type = '信用新規売'
        _, has_successfully_ordered = await SBI.make_order(tab=tab,
                        trade_type=trade_type, ticker=ticker, unit=unit, order_type="成行", nariyuki_value=nariyuki_value,
                        limit_order_price=None, stop_order_trigger_price=None, stop_order_type="成行", stop_order_price=None,
                        period_type="当日中", period_value=None, period_index=None, trade_section="特定預り",
                        margin_trade_section="制度")
        if has_successfully_ordered == False:
            failed_order_list.append(f'{trade_type}: {ticker} {unit}株')
            failed_tickers.append(ticker)
    failed_orders_df = orders_df.loc[orders_df['Code'].isin(failed_tickers), :]
    # 発注失敗した銘柄をdfとして保存
    failed_orders_df.to_csv(paths.FAILED_ORDERS_CSV)
    failed_orders_df.to_csv(paths.FAILED_ORDERS_BACKUP)
    return failed_order_list

async def make_new_order(long_orders:pd.DataFrame, short_orders:pd.DataFrame, 
                         tab:uc.core.tab.Tab=None) -> Tuple[uc.core.tab.Tab, bool, list]:
    '''
    新規注文を発注する。
    '''
    #現時点での注文リストをSBI証券から取得
    tab = await SBI.sign_in(tab)
    tab, orders = await SBI._extract_order_list(tab)

    #発注処理の条件に当てはまるときのみ処理実行
    if len(orders) > 0:
        position_list = [x[:2] for x in orders['取引'].unique()]
        #信用新規がある場合のみ注文キャンセル
        if '信新' in position_list:
            return tab, None, None

    # Long, Shortそれぞれの発注リストの結合
    long_orders['LorS'] = 'Long'
    short_orders['LorS']= 'Short'
    orders_df = pd.concat([long_orders, short_orders], axis=0).sort_values('CumCost_byLS', ascending=True)
    # ポジションを発注
    failed_order_list = await _make_orders(orders_df = orders_df, nariyuki_value='寄成', tab = tab)

    return tab, failed_order_list

async def make_additional_order(tab:uc.core.tab.Tab=None):
    #現時点での注文リストをSBI証券から取得
    tab = await SBI.sign_in(tab)
    orders_df = pd.read_csv(paths.FAILED_ORDERS_CSV)
    orders_df['Code'] = orders_df['Code'].astype(str)
    #ポジションの発注
    failed_order_list = await _make_orders(orders_df = orders_df, nariyuki_value=None, tab = tab)

    return tab, failed_order_list

async def settle_all_margins(tab:uc.core.tab.Tab=None) -> uc.core.tab.Tab:
    '''
    決済注文を発注する。
    [返り値]
    tab: nodriverのタブ
    '''
    tab = await SBI.sign_in(tab)
    _, error_list = await SBI.settle_all_margins(tab)

    return tab, error_list

async def update_information(sector_list_df_path: str, trade_history_path: str, 
                             buying_power_history_path: str, deposit_history_df_path: str, 
                             tab:uc.core.tab.Tab=None) -> Tuple[uc.core.tab.Tab, pd.DataFrame, pd.DataFrame, pd.DataFrame, float, str]:
    '''
    SBI証券からスクレイピングして、取引情報、買付余力、総入金額を更新
    '''
    tab, trade_history = await _update_trade_history(trade_history_path, sector_list_df_path, tab)
    tab, buying_power_history = await _update_buying_power_history(buying_power_history_path, trade_history, tab)
    tab, deposit_history = await _update_deposit_history(deposit_history_df_path, buying_power_history, tab)

    amount, rate = _show_latest_result(trade_history)

    return tab, trade_history, buying_power_history, deposit_history, rate, amount


def load_information(trade_history_path: str, buying_power_history_path: str, deposit_history_df_path: str) \
                                                                  -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    '''
    取引情報、買付余力、総入金額を更新せず読み込み
    '''
    trade_history = pd.read_csv(trade_history_path, index_col=None)
    trade_history['日付'] = pd.to_datetime(trade_history['日付'])
    buying_power_history = pd.read_csv(buying_power_history_path, index_col=['日付'])
    buying_power_history.index = pd.to_datetime(buying_power_history.index)
    deposit_history = pd.read_csv(deposit_history_df_path, index_col=['日付'])
    deposit_history.index = pd.to_datetime(deposit_history.index)

    rate, amount = _show_latest_result(trade_history)

    return trade_history, buying_power_history, deposit_history, rate, amount

#%% デバッグ
if __name__ == '__main__':
    import paths
    import stock_dfs_reader as reader
    import sector_index_calculator
    import MLDataset
    '''
    NEW_SECTOR_LIST_CSV = f'{paths.SECTOR_REDEFINITIONS_FOLDER}/49sectors_list.csv' #別でファイルを作っておく
    NEW_SECTOR_PRICE_PKLGZ = f'{paths.SECTOR_REDEFINITIONS_FOLDER}/49sectors_price.pkl.gz' #出力のみなのでファイルがなくてもOK
    #ユニバースを絞るフィルタ
    universe_filter = "(Listing==1)&((ScaleCategory=='TOPIX Core30')|(ScaleCategory=='TOPIX Large70')|(ScaleCategory=='TOPIX Mid400'))" #現行のTOPIX500
    stock_dfs_dict = reader.read_stock_dfs(filter= universe_filter)
    _, order_price_df = sector_index_calculator.calc_new_sector_price(stock_dfs_dict, NEW_SECTOR_LIST_CSV, NEW_SECTOR_PRICE_PKLGZ)
    ML_DATASET_PATH = f'{paths.ML_DATASETS_FOLDER}/test.pkl.gz'
    ml_dataset = MLDataset.MLDataset(ML_DATASET_PATH)
    tab, long_orders, short_orders, todays_pred_df = \
        asyncio.run(select_stocks(order_price_df, NEW_SECTOR_LIST_CSV, ml_dataset.pred_result_df, trading_sector_num=3, candidate_sector_num=5, top_slope = 1.5))
    tab, take_position, failed_order_list = asyncio.run(make_new_order(long_orders, short_orders, tab))
    '''
    tab = asyncio.run(SBI.sign_in())
    tab = asyncio.run(SBI.cancel_all_orders(tab))
    #tab = asyncio.run(settle_all_margins(tab))
