from utils.paths import Paths
from utils.metadata import FeatureMetadata
import pandas as pd
from numpy.typing import NDArray
from typing import Literal
from calculation.sector_index_calculator import SectorIndexCalculator
import re
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.ERROR)

class IndexFeatureCalculator:
    """
    インデックス特徴量を計算するためのクラス。
    """
    def __init__(self, currencies_type: Literal['relative', 'raw'] = 'relative'):
        """
        IndexFeatureCalculatorのコンストラクタ。

        Args:
            currencies_type (str): 通貨の計算方法を指定（"relative" または "raw"）。
        """
        self.features_df = pd.DataFrame(columns=['Date'])
        self.metadata_list = []
        self.currencies_type = currencies_type
    
    def calculate_return(self, feature_metadata: FeatureMetadata, days: int):
        """
        指定された特徴量のリターンを計算します。

        Args:
            feature_metadata (FeatureMetadata): 特徴量のメタデータ。
            days (int): リターンを計算する日数。
        """
        try:
            if feature_metadata.group == "currencies":
                feature = self._calculate_currency_return(feature_metadata, days)
            elif feature_metadata.group == "commodity":
                feature = self._calculate_commodity_return(feature_metadata, days)
            elif feature_metadata.group == "bond":
                feature = self._calculate_bond_return(feature_metadata, days)
            else:
                feature = self._calculate_plain_return(feature_metadata, days)

            self.features_df = pd.merge(self.features_df, feature, how='outer', on='Date')
            self.metadata_list.append(feature_metadata)

        except FileNotFoundError as e:
            logging.error(f"ファイルが見つかりません: {feature_metadata.filepath} | Error: {e}")
        except Exception as e:
            logging.error(f'{feature_metadata.name}の算出中にエラーが発生しました: {e}')

    def _calculate_currency_return(self, metadata: FeatureMetadata, days: int) -> pd.DataFrame:
        """
        通貨系特徴量のリターンを計算します。

        Args:
            metadata (FeatureMetadata): 特徴量のメタデータ。
            days (int): リターンを計算する日数。

        Returns:
            pd.DataFrame: 計算されたリターンを含むデータフレーム。
        """
        raw_df = pd.read_parquet(metadata.parquet_path)
        feature = pd.DataFrame()
        feature["Date"] = raw_df["Date"]
        feature[f"{metadata.name}_{days}d_return"] = raw_df["Close"].pct_change(days)  # 1日のリターン計算

        return feature

    def _calculate_commodity_return(self, metadata: FeatureMetadata, days: int) -> pd.DataFrame:
        """
        商品系特徴量のリターンを計算します。

        Args:
            metadata (FeatureMetadata): 特徴量のメタデータ。
            days (int): リターンを計算する日数。

        Returns:
            pd.DataFrame: 計算されたリターンを含むデータフレーム。
        """
        return self._calculate_plain_return(metadata, days)

    def _calculate_bond_return(self, metadata: FeatureMetadata, days: int) -> pd.DataFrame:
        """
        債券系特徴量のリターンを計算します。

        Args:
            metadata (FeatureMetadata): 特徴量のメタデータ。
            days (int): リターンを計算する日数。

        Returns:
            pd.DataFrame: 計算されたリターンを含むデータフレーム。
        """
        raw_df = pd.read_parquet(metadata.parquet_path)
        feature = pd.DataFrame()
        feature["Date"] = raw_df["Date"]
        feature[f"{metadata.name}_{days}d_return"] = raw_df["Close"].diff(days)
        return feature

    def _calculate_plain_return(self, metadata: FeatureMetadata, days:int):
        """
        基本的なリターンの計算方法を定義します。

        Args:
            metadata (FeatureMetadata): 特徴量のメタデータ。
            days (int): リターンを計算する日数。

        Returns:
            pd.DataFrame: 計算されたリターンを含むデータフレーム。
        """
        raw_df = pd.read_parquet(metadata.parquet_path)
        feature = pd.DataFrame()
        feature["Date"] = raw_df["Date"]
        feature[f"{metadata.name}_{days}d_return"] = raw_df["Close"].pct_change(days)
        return feature        

    def finalize(self) -> pd.DataFrame:
        """
        計算した特徴量をまとめてデータフレームとして出力します。

        Returns:
            pd.DataFrame: 算出された特徴量を含むデータフレーム。
        """
        features_df = self.features_df.set_index('Date').ffill()
        if self.currencies_type == 'relative':
            features_df = self._relativizate_currency(features_df)
        
        self._process_bond(features_df)
        return features_df

    def _relativizate_currency(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        通貨系特徴量を相対化します。

        Args:
            features_df (pd.DataFrame): 特徴量を格納したデータフレーム。

        Returns:
            pd.DataFrame: 相対化された通貨特徴量を含むデータフレーム。
        """
        # 通貨ペア・通貨系特徴量・接尾辞を抽出
        pairs = [metadata.name for metadata in self.metadata_list if metadata.group == 'currency']
        currency_features = [col for pair in pairs for col in features_df.columns if pair in col]
        unique_suffixes = {x.split('_', 1)[1] for x in currency_features if '_' in x}
        
        for suffix in unique_suffixes:
            #接尾辞をキーにして通貨を抽出
            features = [x for x in currency_features if suffix in x]
            unique_currencies = {pair[i:i+3] for x in features for i in (0, 3) for pair in [x[:6]]}
            currency_counts = {currency: 0 for currency in unique_currencies}
            # 各通貨列を作成して初期化
            for currency in unique_currencies:
                features_df[f'{currency}{suffix}'] = 0
            
            for feature in features:
                # 相対化のために加算・減算を行う
                for i in (0, 3):
                    currency = feature[i:i+3]
                    features_df[f'{currency}{suffix}'] += features_df[feature] * (1 if i == 0 else -1)
                    currency_counts[currency] += 1
                features_df = features_df.drop(feature, axis=1)
            
            for currency in unique_currencies:
                # 各通貨の平均を計算
                features_df[f'{currency}{suffix}'] /= currency_counts[currency]

        return features_df

    def _process_bond(self, features_df:pd.DataFrame) -> pd.DataFrame:
        """
        債券系特徴量の差分を計算します。

        Args:
            features_df (pd.DataFrame): 特徴量を格納したデータフレーム。

        Returns:
            pd.DataFrame: 債券系特徴量の差分を含むデータフレーム。
        """
        # 債券・債券系特徴量・接尾辞を抽出
        bonds = [metadata.name for metadata in self.metadata_list if metadata.group == 'bond']
        bond_features = [col for bond in bonds for col in features_df.columns if bond in col]
        unique_suffixes = {x.split('_', 1)[1] for x in bond_features if '_' in x}
        
        for suffix in unique_suffixes:
            #接尾辞をキーにして債券を抽出
            features_with_suffix = [x for x in bond_features if suffix in x]
            unique_bonds = set([x[:x.find('bond') + len('bond')] for x in bond_features if suffix in x])
            for bond in unique_bonds:
                # 債券特徴量をリストしてソート化（年数降順）
                features = [x[:x.find('_')] for x in features_with_suffix if bond in x]
                sorted_bond_list = sorted(features, key=lambda x: int(re.search(r'\d+', x).group()), reverse=True)
                # 長期-短期で差分をとる
                features_df[f'{bond}_diff_{suffix}'] = \
                    features_df[f'{sorted_bond_list[0]}_{suffix}'] - features_df[f'{sorted_bond_list[1]}_{suffix}']
        return features_df



@dataclass
class PriceFeatureParams:
    '''
    return_duration: リターン算出日数をまとめたリスト
    mom_duration: モメンタム算出日数をまとめたリスト
    vola_duration: ボラティリティ算出日数をまとめたリスト
    adopt_size_factor: サイズファクターを特徴量とするか
    adopt_eps_factor: EPSを特徴量とするか
    adopt_sector_categorical: セクターをカテゴリカル変数として採用するか
    '''
    return_duration: list[int] = [1, 5, 21]
    vola_duration: list[int] = [5, 21]
    adopt_size_factor: bool = True
    adopt_eps_factor: bool = True
    adopt_sector_categorical: bool = True

class PriceFeatureCalculator:
    def __init__(self, stock_list: pd.DataFrame, stock_fin: pd.DataFrame, stock_price: pd.DataFrame):
        self.stock_list = stock_list
        self.stock_fin = stock_fin
        self.stock_price = stock_price
        
    def calculate(self, price_timeseries: pd.DataFrame, feature_name: str, feature_params: PriceFeatureParams, 
                  date_col: str = 'Date', price_col: str = 'Close'):
        '''
        価格系の特徴量を算出します。
        Args:
            price_timeseries (pd.DataFrame): 算出対象の価格の時系列データ
            feature_name (str): 特徴量の名称。列名に使用
            feature_params (PriceFeatureParams): 算出時のパラメータ
            date_col (str): price_timeseriesで日にちを規定している列の名前
            price_col (str): price_timeseriesで価格（終値）を規定している列の名前
        '''
        price_timeseries = self._sort_by_date(price_timeseries, date_col)
        features_price_df = pd.DataFrame(price_timeseries[date_col], columns = [date_col])
        
        excluding_days = 0
        for i in feature_params.return_duration:
            features_price_df[f'{feature_name}_{i}d_return'] = \
                self._calculate_return(price_timeseries, price_col = price_col, days = i, excluding_days = excluding_days)
            excluding_days = i

        df_for_feature_calc = self._get_df_for_feature_calc(price_timeseries, features_price_df, feature_name, feature_params, date_col, price_col)

        excluding_days = 0
        for i in feature_params.vola_duration:
            features_price_df[f'{feature_name}_{i}d_vola'] = \
                self._calculate_vola(df_for_feature_calc, price_col = price_col, days = i, excluding_days = excluding_days)
            excluding_days = i



    def _sort_by_date(self, timeseries_df: pd.DataFrame, date_col: str):
        '''
        データフレームを日付昇順に並べ替えます。
        Args:
            timeseries_df (pd.DataFrame): 並べ替え対象の時系列データ
            date_col (str): timeseries_dfで日にちを規定している列の名前
        Return:
            pd.DataFrame: 日付昇順に並べ替えたtimeseries_df
        '''
        timeseries_df[date_col] = pd.to_datetime(timeseries_df[date_col])
        return timeseries_df.sort_values(date_col, ascending = True)


    def _get_df_for_feature_calc(self, price_timeseries: pd.DataFrame, features_price_df: pd.DataFrame, 
                                 feature_params: PriceFeatureParams, date_col: str, price_col) -> pd.DataFrame:
        '''
        価格系の特徴量を算出するための情報（1日リターン）を含んだデータフレームを返します。
        Args:
            price_timeseries (pd.DataFrame): 算出対象の価格の時系列データ
            features_price_df (pd.DataFrame): 算出済の特徴量データ
            feature_params (PriceFeatureParams): 算出時のパラメータ
            date_col (str): price_timeseriesで日にちを規定している列の名前
            price_col (str): price_timeseriesで価格（終値）を規定している列の名前
        Returns:
            pd.DataFrame: 列 '1d_return'を持つ。
        '''
        if 1 in feature_params.return_duration:
            df_for_feature_calc = features_price_df[[date_col, f'1d_return']]
        else:
            df_for_feature_calc = features_price_df[[date_col]]
            array_for_feature_calc = \
                self._calculate_return(price_timeseries, price_col = price_col, days = 1, excluding_days = 0)
            df_for_feature_calc[f'1d_return'] = array_for_feature_calc
        return df_for_feature_calc


            

        
        # サイズファクター（業種内の平均サイズファクター）
        if adopt_size_factor:
            new_sector_list['Code'] = new_sector_list['Code'].astype(str)
            stock_price_cap = SectorIndexCalculator.calc_marketcap(stock_dfs_dict['price'], stock_dfs_dict['fin'])
            stock_price_cap = stock_price_cap[stock_price_cap['Code'].isin(new_sector_list['Code'])]
            stock_price_cap = pd.merge(stock_price_cap, new_sector_list[['Code', 'Sector']], on='Code', how='left')
            stock_price_cap = stock_price_cap[['Date', 'Code', 'Sector', 'MarketCapClose']]
            stock_price_cap = stock_price_cap.groupby(['Date', 'Sector'])[['MarketCapClose']].mean()
            features_price_df['MarketCapAtClose'] = stock_price_cap['MarketCapClose']
            if add_rank:
                features_price_df['MarketCap_rank'] = features_price_df['MarketCapAtClose'].groupby('Date').rank(ascending=False)

        # EPSファクター
        if adopt_eps_factor:
            eps_df = stock_dfs_dict['fin'][[
                'Code', 'Date', 'EarningsPerShare', 'ForecastEarningsPerShare', 'NextYearForecastEarningsPerShare',
                ]].copy()
            eps_df.loc[
                (eps_df['ForecastEarningsPerShare'].notnull()) & (eps_df['NextYearForecastEarningsPerShare'].notnull()), 
                'NextYearForecastEarningsPerShare'] = 0
            eps_df[['ForecastEarningsPerShare', 'NextYearForecastEarningsPerShare']] = \
                eps_df[['ForecastEarningsPerShare', 'NextYearForecastEarningsPerShare']].fillna(0)
            eps_df['ForecastEPS'] = eps_df['ForecastEarningsPerShare'].values + eps_df['NextYearForecastEarningsPerShare'].values
            eps_df = eps_df[['Code', 'Date', 'ForecastEPS']]
            eps_df = pd.merge(stock_dfs_dict['price'][['Date', 'Code']], eps_df, how='outer', on=['Date', 'Code'])
            eps_df = pd.merge(new_sector_list[['Code', 'Sector']], eps_df, on='Code', how='right')
            eps_df['ForecastEPS'] = eps_df.groupby('Code')['ForecastEPS'].ffill()
            eps_df['ForecastEPS'] = eps_df.groupby('Code')['ForecastEPS'].bfill()
            eps_df = pd.merge(stock_dfs_dict['price'][['Date', 'Code']], eps_df, how='left', on=['Date', 'Code'])
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
    
    def _calculate_return(self, price_timeseries: pd.DataFrame, price_col: str, days: int, excluding_days: int) -> NDArray:
        '''
        指定した日数のリターンを算出します。
        Args:
            price_timeseries (pd.DataFrame): 算出対象の価格時系列データ
            price_col (str): price_timetableの価格列の名称
            days (int): 何日間のリターンを算出するか
            excluding_days (int): 直近何日間の価格情報を除外するか
        Returns:
            NDArray: 指定日数の時系列リターンを格納
        '''
        if excluding_days == 0:
            return price_timeseries[price_col].pct_change(days).values
        else:
            return price_timeseries[price_col].pct_change(days - excluding_days).shift(excluding_days).values
    

    def _calculate_vola(self, df_for_feature_calc: pd.DataFrame, days: int, excluding_days: int) -> NDArray:
        '''
        指定した日数のリターンを算出します。
        Args:
            df_for_feature_calc (pd.DataFrame): 算出用のデータ（1d_return）を持つデータフレーム
            days (int): 何日間のボラティリティを算出するか
            excluding_days (int): 直近何日間のボラティリティを除外するか
        Returns:
            NDArray: 指定日数の時系列ボラティリティを格納
        '''
        if excluding_days == 0:
            return df_for_feature_calc['1d_return'].rolling(days).std().values
        else:
            return df_for_feature_calc['1d_return'].rolling(days - excluding_days).std().shift(excluding_days).values





if feature_params.add_rank:
    features_price_df[f'{i}d_return_rank'] = features_price_df[f'{i}d_return'].groupby('Date').rank(ascending=False)
if feature_params.add_rank:
    features_price_df[f'{n}d_vola_rank'] = features_price_df[f'{n}d_vola'].groupby('Date').rank(ascending=False)


class FeaturesCalculator:
    @staticmethod
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
                        add_rank:bool = True) -> pd.DataFrame:
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
            features_to_scrape_df = \
                FeaturesCalculator.select_features_to_scrape(groups_setting=groups_setting, names_setting=names_setting)
            features_indices_df = \
                FeaturesCalculator.calculate_features_indices(features_to_scrape_df=features_to_scrape_df, 
                                                              currencies_type=currencies_type, 
                                                              commodity_type=commodity_type)
            if adopts_features_price:
                features_price_df = FeaturesCalculator.calculate_features_price(new_sector_price=new_sector_price,
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
            features_df = FeaturesCalculator.merge_features(features_indices_df, features_price_df)
            features_df = features_df.sort_index()
        elif adopts_features_price:
            features_df = FeaturesCalculator.calculate_features_price(new_sector_price=new_sector_price,
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

    @staticmethod
    def select_features_to_scrape(groups_setting:dict={}, names_setting:dict={}):
        features_to_scrape_df = pd.read_csv(Paths.FEATURES_TO_SCRAPE_CSV)
        features_to_scrape_df['Path'] = Paths.SCRAPED_DATA_FOLDER + '/' + features_to_scrape_df['Group'] + '/' + features_to_scrape_df['Path']
        #グループごとに特徴量として採用するか？
        if groups_setting is not None:
            features_to_scrape_df['is_adopted'] = features_to_scrape_df['Group'].map(groups_setting).fillna(features_to_scrape_df['is_adopted'])
        if names_setting is not None:
            features_to_scrape_df['is_adopted'] = features_to_scrape_df['Name'].map(names_setting).fillna(features_to_scrape_df['is_adopted'])
        features_to_scrape_df['is_adopted'] = features_to_scrape_df['is_adopted'].replace({'TRUE': True, 'FALSE': False}).astype(bool)
        return features_to_scrape_df

    @staticmethod
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
                a_feature_df = FeaturesCalculator._calculate_1day_return_commodity_JPY(row, USDJPY_path=USDJPY_path)
            else:
                a_feature_df = FeaturesCalculator._calculate_1day_return(row)
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
                features_indices_df = FeaturesCalculator._process_currency(features_indices_df)
        #債券特徴量の処理
        bond_is_adopted = features_to_scrape_df.loc[features_to_scrape_df['Group'] == 'bond', 'is_adopted'].any()
        if bond_is_adopted:
            features_indices_df = FeaturesCalculator._process_bond(features_indices_df)
            
        print('各特徴量の1day_returnを算出しました。')
        return features_indices_df

    @staticmethod
    def _calculate_1day_return(row:pd.Series) -> pd.DataFrame:
        a_feature_df = None
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
        return a_feature_df

    @staticmethod
    def _calculate_1day_return_commodity_JPY(row:pd.Series, USDJPY_path:str) -> pd.DataFrame:
        a_feature_df = None
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
        return a_feature_df

    @staticmethod
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
            stock_price_cap = SectorIndexCalculator.calc_marketcap(stock_dfs_dict['price'], stock_dfs_dict['fin'])
            stock_price_cap = stock_price_cap[stock_price_cap['Code'].isin(new_sector_list['Code'])]
            stock_price_cap = pd.merge(stock_price_cap, new_sector_list[['Code', 'Sector']], on='Code', how='left')
            stock_price_cap = stock_price_cap[['Date', 'Code', 'Sector', 'MarketCapClose']]
            stock_price_cap = stock_price_cap.groupby(['Date', 'Sector'])[['MarketCapClose']].mean()
            features_price_df['MarketCapAtClose'] = stock_price_cap['MarketCapClose']
            if add_rank:
                features_price_df['MarketCap_rank'] = features_price_df['MarketCapAtClose'].groupby('Date').rank(ascending=False)

        # EPSファクター
        if adopt_eps_factor:
            eps_df = stock_dfs_dict['fin'][[
                'Code', 'Date', 'EarningsPerShare', 'ForecastEarningsPerShare', 'NextYearForecastEarningsPerShare',
                ]].copy()
            eps_df.loc[
                (eps_df['ForecastEarningsPerShare'].notnull()) & (eps_df['NextYearForecastEarningsPerShare'].notnull()), 
                'NextYearForecastEarningsPerShare'] = 0
            eps_df[['ForecastEarningsPerShare', 'NextYearForecastEarningsPerShare']] = \
                eps_df[['ForecastEarningsPerShare', 'NextYearForecastEarningsPerShare']].fillna(0)
            eps_df['ForecastEPS'] = eps_df['ForecastEarningsPerShare'].values + eps_df['NextYearForecastEarningsPerShare'].values
            eps_df = eps_df[['Code', 'Date', 'ForecastEPS']]
            eps_df = pd.merge(stock_dfs_dict['price'][['Date', 'Code']], eps_df, how='outer', on=['Date', 'Code'])
            eps_df = pd.merge(new_sector_list[['Code', 'Sector']], eps_df, on='Code', how='right')
            eps_df['ForecastEPS'] = eps_df.groupby('Code')['ForecastEPS'].ffill()
            eps_df['ForecastEPS'] = eps_df.groupby('Code')['ForecastEPS'].bfill()
            eps_df = pd.merge(stock_dfs_dict['price'][['Date', 'Code']], eps_df, how='left', on=['Date', 'Code'])
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

    @staticmethod
    def merge_features(features_indices_df:pd.DataFrame, features_price_df:pd.DataFrame) -> pd.DataFrame:
        '''features_indicesとfeatures_priceを結合'''
        features_df = pd.merge(features_indices_df.reset_index(), features_price_df.reset_index(), on=['Date'], how='outer')
        features_df = features_df.dropna(subset=['Sector'])
        features_sector = features_df['Sector'].values #groupbyすると'Sector'列が消えるため、復元のためにあらかじめ値を取得
        features_df = features_df.groupby('Sector').ffill()
        features_df['Sector'] = features_sector

        features_df = features_df.set_index(['Date', 'Sector'], drop=True)

        return features_df



#%% デバッグ
if __name__ == '__main__':
    from utils.paths import Paths
    setting_df = pd.read_csv(Paths.FEATURES_TO_SCRAPE_CSV)
    ifc = IndexFeatureCalculator()
    for _, row in setting_df.iterrows():
        fmd = FeatureMetadata(row['Name'], row['Group'], f"{Paths.SCRAPED_DATA_FOLDER}/{row['Group']}/{row['Path']}", row['URL'], row['is_adopted'])
        ifc.calculate_return(feature_metadata=fmd, days=1)
    df = ifc.finalize()
    print(df)