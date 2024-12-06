#%% モジュールのインポート
import paths

import jquants_api_fetcher as fetcher
import sector_index_calculator

import pandas as pd
import pickle
from typing import Literal

#%% 関数群
def _calculate_1day_return(row:pd.Series) -> pd.DataFrame:
    if row['is_adopted']:
        #既存のデータフレームを読み込み
        raw_df = pd.read_parquet(row['Path'])
        #リターンの算出
        a_feature_df = pd.DataFrame()
        a_feature_df['Date'] = pd.to_datetime(raw_df['Date'])
        if row['Group'] == 'bond':
            a_feature_df[f'{row["Name"]}_1d_return'] = raw_df['Close'].diff(1) #目的変数を得る。
        else:
            a_feature_df[f'{row["Name"]}_1d_return'] = raw_df['Close'].pct_change(1) #目的変数を得る。
    else:
        a_feature_df = None

    return a_feature_df


def _calculate_1day_return_commodity_JPY(row:pd.Series, USDJPY_path:str) -> pd.DataFrame:
    if row['is_adopted']:
        #既存のデータフレームを読み込み
        raw_df = pd.read_parquet(row['Path'])
        USDJPY_df = pd.read_parquet(USDJPY_path)
        USDJPY_df = USDJPY_df.rename(columns={'Close': 'USDJPYClose'})
        raw_df = pd.merge(raw_df, USDJPY_df[['Date', 'USDJPYClose']], on='Date', how='left')
        #リターンの算出
        a_feature_df = pd.DataFrame()
        a_feature_df['Date'] = pd.to_datetime(raw_df['Date'])
        a_feature_df[f'{row["Name"]}_1d_return'] = (raw_df['Close'] * raw_df['USDJPYClose']).pct_change(1) #目的変数を得る。
    else:
        a_feature_df = None

    return a_feature_df


def _process_bond(features_indices_df:pd.DataFrame) -> pd.DataFrame:
    #通貨の相対強度（JPY, USD, EUR, AUD）
    #通貨が強い（=高い）ほど値が大きくなる
    '''いずれ1d_return以外にも対応したい'''
    bond_features = [x for x in features_indices_df.columns if 'bond10' in x]
    for x in bond_features:
        features_indices_df[x.replace("10", "_diff")] = features_indices_df[x] - features_indices_df[x.replace("10", "2")]
        features_indices_df.drop([x.replace("10", "2")], axis=1, inplace=True)
    return features_indices_df


def _process_currency(features_indices_df:pd.DataFrame) -> pd.DataFrame:
    #通貨の相対強度（JPY, USD, EUR, AUD）
    #通貨が強い（=高い）ほど値が大きくなる
    '''いずれ1d_return以外にも対応したい'''
    for i in ['1d_return']:#, f'{duration1}d_return', f'{duration2}d_return']:
        features_indices_df[f'JPY_{i}'] = (- features_indices_df[f'USDJPY_{i}'] - features_indices_df[f'EURJPY_{i}'] - features_indices_df[f'AUDJPY_{i}']) / 3
        features_indices_df[f'USD_{i}'] = (features_indices_df[f'USDJPY_{i}'] - features_indices_df[f'EURUSD_{i}'] - features_indices_df[f'AUDUSD_{i}']) / 3
        features_indices_df[f'AUD_{i}'] = (features_indices_df[f'AUDJPY_{i}'] + features_indices_df[f'AUDUSD_{i}'] - features_indices_df[f'EURAUD_{i}']) / 3
        features_indices_df[f'EUR_{i}'] = (features_indices_df[f'EURJPY_{i}'] + features_indices_df[f'EURUSD_{i}'] + features_indices_df[f'EURAUD_{i}']) / 3
        features_indices_df.drop([f'USDJPY_{i}', f'EURJPY_{i}', f'AUDJPY_{i}', f'EURUSD_{i}', f'AUDUSD_{i}', f'EURAUD_{i}'], axis=1, inplace=True)
    return features_indices_df


def select_features_to_scrape(groups_setting:dict={}, names_setting:dict={}):
    features_to_scrape_df = pd.read_csv(paths.FEATURES_TO_SCRAPE_CSV)
    features_to_scrape_df['Path'] = paths.SCRAPED_DATA_FOLDER + '/' + features_to_scrape_df['Group'] + '/' + features_to_scrape_df['Path']
    #グループごとに特徴量として採用するか？
    if groups_setting is not None:
        features_to_scrape_df['is_adopted'] = features_to_scrape_df['Group'].map(groups_setting).fillna(features_to_scrape_df['is_adopted'])
    if names_setting is not None:
        features_to_scrape_df['is_adopted'] = features_to_scrape_df['Name'].map(names_setting).fillna(features_to_scrape_df['is_adopted'])
    features_to_scrape_df['is_adopted'] = features_to_scrape_df['is_adopted'].replace({'TRUE': True, 'FALSE': False}).astype(bool)
    return features_to_scrape_df


def calculate_features_indices(features_to_scrape_df:pd.DataFrame, currencies_type:str, commodity_type:str) -> pd.DataFrame:
    '''特徴量の算出'''
    features_indices_df = pd.DataFrame()
    features_indices_df['Date'] = None
    for _, row in features_to_scrape_df.iterrows():
        '''いずれ1d_return以外にも対応したい'''
        #commodity_typeがJPYのとき、コモディティのリターンを円建てで補正
        should_convert_to_JPY = (row['Group'] == 'commodity') & (commodity_type == 'JPY')
        if should_convert_to_JPY:
            print()
            USDJPY_path = features_to_scrape_df.loc[features_to_scrape_df['Name']=='USDJPY', 'Path'].values[0]
            a_feature_df = _calculate_1day_return_commodity_JPY(row, USDJPY_path=USDJPY_path)
        else:
            a_feature_df = _calculate_1day_return(row)
        if a_feature_df is not None:
            features_indices_df = pd.merge(features_indices_df, a_feature_df, how='outer', on='Date').sort_values(by='Date')
    #最終的に返すデータフレームを成型
    features_indices_df = features_indices_df.set_index('Date')
    features_indices_df = features_indices_df.ffill()

    #通貨特徴量の処理
    currency_is_adopted = features_to_scrape_df.loc[features_to_scrape_df['Group'] == 'currencies', 'is_adopted'].all()
    if currency_is_adopted:
        if currencies_type not in ['relative', 'raw']:
            raise ValueError("引数'currencies_type'には、'relative'か'raw'のいずれかを設定してください。")
        elif currencies_type == 'relative':
            features_indices_df = _process_currency(features_indices_df)
    #債券特徴量の処理
    bond_is_adopted = features_to_scrape_df.loc[features_to_scrape_df['Group'] == 'bond', 'is_adopted'].any()
    if bond_is_adopted:
        features_indices_df = _process_bond(features_indices_df)

    print('各特徴量の1day_returnを算出しました。')

    return features_indices_df


def calculate_features_price(
        new_sector_price:pd.DataFrame,
        new_sector_list:pd.DataFrame,
        stock_dfs_dict:dict,
        adopt_1d_return:bool = True,
        mom_duration:list = [5, 21], 
        vola_duration:list = [5, 21],
        adopt_size_factor:bool = True,
        adopt_eps_factor:bool = True,
        adopt_sector_categorical:bool = True,
        add_rank:bool = True,
        ) -> pd.DataFrame:
    '''
    価格系の特徴量を生成する関数。
    new_sector_price: 業種別インデックスの価格情報
    new_sector_list: 各銘柄の業種設定
    stock_dfs_dict: J-Quants API由来のデータを含んだ辞書
    return_duration: リターン算出日数をまとめたリスト
    mom_duration: モメンタム算出日数をまとめたリスト
    vola_duration: ボラティリティ算出日数をまとめたリスト
    adopt_size_factor: サイズファクターを特徴量とするか
    adopt_eps_factor: EPSを特徴量とするか
    adopt_sector_categorical: セクターをカテゴリカル変数として採用するか
    add_rank: 各日・各指標のランクを特徴量として追加するか
    '''
    features_price_df = pd.DataFrame()
    # リターンの算出
    if adopt_1d_return:
        features_price_df['1d_return'] = new_sector_price['1d_return']
        if add_rank:
            features_price_df[f'1d_return_rank'] = features_price_df[f'1d_return'].groupby('Date').rank(ascending=False)
    # モメンタムの算出
    if mom_duration is not None:
        assert isinstance(mom_duration, list), "mom_durationにはリストまたはNoneを指定してください。"
        assert '1d_return' in features_price_df.columns, "momを算出するには、最低でも1day_returnを取得するようにしてください。"
        days_to_exclude = 1
        for n in mom_duration:
            features_price_df[f'{n}d_mom'] \
                = features_price_df['1d_return'].groupby('Sector').rolling(n - days_to_exclude).mean().reset_index(0, drop=True)
            features_price_df[f'{n}d_mom'] = features_price_df[f'{n}d_mom'].groupby('Sector').shift(days_to_exclude)
            days_to_exclude = n
            if add_rank:
                features_price_df[f'{n}d_mom_rank'] = features_price_df[f'{n}d_mom'].groupby('Date').rank(ascending=False)
    # ボラティリティの算出
    if vola_duration is not None:
        assert isinstance(vola_duration, list), "vola_durationにはリストまたはNoneを指定してください。"
        assert '1d_return' in features_price_df.columns, "volaを算出するには、最低でも1day_returnを取得するようにしてください。"
        days_to_exclude = 1
        for n in vola_duration:
            features_price_df[f'{n}d_vola'] \
                = features_price_df['1d_return'].groupby('Sector').rolling(n - days_to_exclude).std().reset_index(0, drop=True)
            features_price_df[f'{n}d_vola'] = features_price_df[f'{n}d_vola'].groupby('Sector').shift(days_to_exclude)
            days_to_exclude = n
            if add_rank:
                features_price_df[f'{n}d_vola_rank'] = features_price_df[f'{n}d_vola'].groupby('Date').rank(ascending=False)
    
    # サイズファクター（業種内の平均サイズファクター）
    if adopt_size_factor:
        new_sector_list['Code'] = new_sector_list['Code'].astype(str)
        stock_price_cap = sector_index_calculator.calc_marketcap(stock_dfs_dict['stock_price'], stock_dfs_dict['stock_fin'])
        stock_price_cap = stock_price_cap[stock_price_cap['Code'].isin(new_sector_list['Code'])]
        stock_price_cap = pd.merge(stock_price_cap, new_sector_list[['Code', 'Sector']], on='Code', how='left')
        stock_price_cap = stock_price_cap[['Date', 'Code', 'Sector', 'MarketCapClose']]
        stock_price_cap = stock_price_cap.groupby(['Date', 'Sector'])[['MarketCapClose']].mean()
        features_price_df['MarketCapAtClose'] = stock_price_cap['MarketCapClose']
        if add_rank:
            features_price_df['MarketCap_rank'] = features_price_df['MarketCapAtClose'].groupby('Date').rank(ascending=False)

    # EPSファクター
    if adopt_eps_factor:
        eps_df = stock_dfs_dict['stock_fin'][[
            'Code', 'Date', 'EarningsPerShare', 'ForecastEarningsPerShare', 'NextYearForecastEarningsPerShare',
            ]].copy()
        eps_df.loc[
            (eps_df['ForecastEarningsPerShare'].notnull()) & (eps_df['NextYearForecastEarningsPerShare'].notnull()), 
            'NextYearForecastEarningsPerShare'] = 0
        eps_df[['ForecastEarningsPerShare', 'NextYearForecastEarningsPerShare']] = \
            eps_df[['ForecastEarningsPerShare', 'NextYearForecastEarningsPerShare']].fillna(0)
        eps_df['ForecastEPS'] = eps_df['ForecastEarningsPerShare'].values + eps_df['NextYearForecastEarningsPerShare'].values
        eps_df = eps_df[['Code', 'Date', 'ForecastEPS']]
        eps_df = pd.merge(stock_dfs_dict['stock_price'][['Date', 'Code']], eps_df, how='outer', on=['Date', 'Code'])
        eps_df = pd.merge(new_sector_list[['Code', 'Sector']], eps_df, on='Code', how='right')
        eps_df['ForecastEPS'] = eps_df.groupby('Code')['ForecastEPS'].ffill()
        eps_df['ForecastEPS'] = eps_df.groupby('Code')['ForecastEPS'].bfill()
        eps_df = pd.merge(stock_dfs_dict['stock_price'][['Date', 'Code']], eps_df, how='left', on=['Date', 'Code'])
        eps_df['ForecastEPS'] = eps_df.groupby('Code')['ForecastEPS'].ffill()
        eps_df['ForecastEPS'] = eps_df.groupby('Code')['ForecastEPS'].bfill()
        eps_df = eps_df.groupby(['Date', 'Sector'])[['ForecastEPS']].mean()
        eps_df['ForecastEPS_rank'] = eps_df.groupby('Date')['ForecastEPS'].rank(ascending=False) 
        features_price_df[['ForecastEPS', 'ForecastEPS_rank']] = eps_df[['ForecastEPS', 'ForecastEPS_rank']].copy()

    # セクターをカテゴリ変数として追加
    if adopt_sector_categorical:
        sector_replace_dict = {x: i for i, x in enumerate(features_price_df.index.get_level_values(1).unique())}
        features_price_df['Sector_cat'] = features_price_df.index.get_level_values(1)
        features_price_df['Sector_cat'] = features_price_df['Sector_cat'].replace(sector_replace_dict).infer_objects(copy=False)

    print('価格系特徴量の算出が完了しました。')

    return features_price_df

def merge_features(features_indices_df:pd.DataFrame, features_price_df:pd.DataFrame) -> pd.DataFrame:
    '''features_indicesとfeatures_priceを結合'''
    features_df = pd.merge(features_indices_df.reset_index(), features_price_df.reset_index(), on=['Date'], how='outer')
    features_df = features_df.dropna(subset=['Sector'])
    features_sector = features_df['Sector'].values #groupbyすると'Sector'列が消えるため、復元のためにあらかじめ値を取得
    features_df = features_df.groupby('Sector').ffill()
    features_df['Sector'] = features_sector

    features_df = features_df.set_index(['Date', 'Sector'], drop=True)

    return features_df

#%% メイン関数
def calculate_features(new_sector_price:pd.DataFrame,
                       new_sector_list: pd.DataFrame,
                       stock_dfs_dict: dict,
                       adopts_features_indices:bool = True,
                       adopts_features_price:bool = True,
                       groups_setting:dict = {},
                       names_setting:dict = {},
                       currencies_type:Literal['relative', 'raw'] = 'relative',
                       commodity_type:Literal['JPY', 'raw'] = 'raw',
                       adopt_1d_return:bool = True, 
                       mom_duration:list = [5, 21], 
                       vola_duration:list = [5, 21],
                       adopt_size_factor:bool = True,
                       adopt_eps_factor:bool = True,
                       adopt_sector_categorical:bool = True,
                       add_rank:bool = True):
    """
    特徴量を計算する。
    :param pd.DataFrame new_sector_price: セクター価格
    :param pd.DataFrame new_sector_list: 各銘柄の業種設定
    :param dict stock_dfs_dict: J-quants API由来のデータを格納した辞書
    :param bool adopt_features_indices: インデックス系特徴量の採否
    :param bool adopt_features_price: 価格系特徴量の採否
    :param dict groups_setting: （インデックス）特徴量グループの採否
    :param dict names_setting: （インデックス）特徴量の採否
    :param str currencies_type: 通貨を'relative'なら相対強度(例：'JPY')、'raw'ならそのまま(例：'USDJPY')
    :param str commodity_type: コモディティを円建てに補正するか否か
    :param list return_duration: （価格）何日間のリターンを特徴量とするか
    :param list mom_duration: （価格）何日間のモメンタムを特徴量とするか
    :param list vola_duration: （価格）何日間のボラティリティを特徴量とするか
    :param bool adopt_size_factor: （価格）サイズファクターを特徴量とするか
    :param bool adopt_eps_factor: （価格）EPSを特徴量とするか
    :param bool adopt_sector_categorical: （価格）セクターをカテゴリ変数として採用するか
    :param bool add_rank: （価格）各日・各指標のの業種別ランキング
    """
    if adopts_features_indices:
        features_to_scrape_df = select_features_to_scrape(groups_setting=groups_setting, names_setting=names_setting)
        features_indices_df = calculate_features_indices(features_to_scrape_df=features_to_scrape_df, currencies_type=currencies_type, commodity_type=commodity_type)
        if adopts_features_price:
            features_price_df = calculate_features_price(new_sector_price=new_sector_price,
                                                         new_sector_list=new_sector_list,
                                                         stock_dfs_dict=stock_dfs_dict,
                                                         adopt_1d_return=adopt_1d_return, 
                                                         mom_duration=mom_duration,
                                                         vola_duration=vola_duration,
                                                         adopt_size_factor=adopt_size_factor,
                                                         adopt_eps_factor=adopt_eps_factor,
                                                         adopt_sector_categorical=adopt_sector_categorical,
                                                         add_rank=add_rank)
        else:
            features_price_df = pd.DataFrame(index=new_sector_price.index)
        features_df = merge_features(features_indices_df, features_price_df)
        features_df = features_df.sort_index()
    elif adopts_features_price:
        features_df = calculate_features_price(new_sector_price=new_sector_price,
                                               new_sector_list=new_sector_list,
                                               stock_dfs_dict=stock_dfs_dict,
                                               adopt_1d_return=adopt_1d_return, 
                                               mom_duration=mom_duration,
                                               vola_duration=vola_duration,
                                               adopt_size_factor=adopt_size_factor,
                                               adopt_eps_factor=adopt_eps_factor,
                                               adopt_sector_categorical=adopt_sector_categorical,
                                               add_rank=add_rank)
    else:
        return None

    return features_df

#%% デバッグ
if __name__ == '__main__':

    import stock_dfs_reader as reader # 加工したデータの読み込み
    import sector_index_calculator
    from IPython.display import display

    '''パス類'''
    NEW_SECTOR_LIST_CSV = f'{paths.SECTOR_REDEFINITIONS_FOLDER}/New48sectors_list.csv' # 別でファイルを作っておく
    NEW_SECTOR_PRICE_PKLGZ = f'{paths.SECTOR_REDEFINITIONS_FOLDER}/New48sectors_price.pkl.gz' # 出力のみなのでファイルがなくてもOK
    '''ユニバースを絞るフィルタ'''
    universe_filter = \
        "(Listing==1)&((ScaleCategory=='TOPIX Core30')|(ScaleCategory=='TOPIX Large70')|(ScaleCategory=='TOPIX Mid400'))" #現行のTOPIX500

    stock_dfs_dict = reader.read_stock_dfs(filter = universe_filter)
    new_sector_price_df, order_price_df = \
        sector_index_calculator.calc_new_sector_price(stock_dfs_dict, NEW_SECTOR_LIST_CSV, NEW_SECTOR_PRICE_PKLGZ)
    
    new_sector_list = pd.read_csv(NEW_SECTOR_LIST_CSV)

    features_df = calculate_features(new_sector_price_df,
                                     new_sector_list,
                                     stock_dfs_dict,
                                     adopts_features_indices=True, 
                                     adopts_features_price=True,
                                     groups_setting=None,
                                     names_setting=None,
                                     currencies_type='relative',
                                     adopt_1d_return=True, 
                                     mom_duration=[5, 21],
                                     vola_duration=[5, 21],
                                     adopt_size_factor=True,
                                     adopt_eps_factor=True,
                                     adopt_sector_categorical=True,
                                     add_rank=True,
                                     )
    from datetime import datetime
    display(features_df[features_df.index.get_level_values(0)>=datetime(2022,1,1)].iloc[:-48])