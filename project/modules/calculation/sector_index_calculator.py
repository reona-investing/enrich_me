#%% モジュールのインポート
from utils.paths import Paths
from utils import yaml_utils
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Any



def _column_name_getter(yaml_info: dict[str | dict[str | Any]] | list[dict[str | Any]], raw_name: str) -> str:
    """
    指定したカラム名の変換後の名称を取得。

    Args:
        yaml_info (dict[str | dict[str | any]] | list[dict[str | any]]): yamlから取得した設定情報
        raw_name (str): 変換前のカラム名

    Returns:
        str: 変換後のカラム名
    """ 
    if isinstance(yaml_info, dict) or isinstance(yaml_info, list):
        return yaml_utils.column_name_getter(yaml_info, {'name': raw_name}, 'fixed_name')
    raise TypeError('yaml_infoには辞書またはリストを指定してください。')


class SectorIndexCalculator:
    _col_names = None
    
    @staticmethod
    def _get_column_names():
        if SectorIndexCalculator._col_names is None:
            _fin_original_yaml = yaml_utils.including_columns_loader(Paths.STOCK_FIN_COLUMNS_YAML, 'original_columns')
            _fin_calculated_yaml = yaml_utils.including_columns_loader(Paths.STOCK_FIN_COLUMNS_YAML, 'calculated_columns')
            _price_original_yaml = yaml_utils.including_columns_loader(Paths.STOCK_PRICE_COLUMNS_YAML, 'original_columns')
            _sector_calculated_yaml = yaml_utils.including_columns_loader(Paths.SECTOR_INDEX_COLUMNS_YAML, 'calculated_columns')

            col = {
                'fin_code': _column_name_getter(_fin_original_yaml, 'LocalCode'),
                'fin_date': _column_name_getter(_fin_original_yaml, 'DisclosedDate'),
                'period_end_date': _column_name_getter(_fin_original_yaml, 'CurrentPeriodEndDate'),
                'fin_outstanding_shares': _column_name_getter(_fin_calculated_yaml, 'OUTSTANDING_SHARES'),
                
                'price_code': _column_name_getter(_price_original_yaml, 'Code'),
                'price_date': _column_name_getter(_price_original_yaml, 'Date'),
                'adjustment_factor': _column_name_getter(_price_original_yaml, 'AdjustmentFactor'),
                'price_open': _column_name_getter(_price_original_yaml, 'Open'),
                'price_close': _column_name_getter(_price_original_yaml, 'Close'),
                'price_high': _column_name_getter(_price_original_yaml, 'High'),
                'price_low': _column_name_getter(_price_original_yaml, 'Low'),
                'volume': _column_name_getter(_price_original_yaml, 'Volume'),
                
                'sector': _column_name_getter(_sector_calculated_yaml, 'SECTOR'),
                'sector_code': _column_name_getter(_sector_calculated_yaml, 'CODE'),
                'sector_date': _column_name_getter(_sector_calculated_yaml, 'DATE'),
                'market_cap_open': _column_name_getter(_sector_calculated_yaml, 'MARKET_CAP_OPEN'),
                'market_cap_close': _column_name_getter(_sector_calculated_yaml, 'MARKET_CAP_CLOSE'),
                'market_cap_high': _column_name_getter(_sector_calculated_yaml, 'MARKET_CAP_HIGH'),
                'market_cap_low': _column_name_getter(_sector_calculated_yaml, 'MARKET_CAP_LOW'),
                'sector_outstanding_shares': _column_name_getter(_sector_calculated_yaml, 'OUTSTANDING_SHARES'),
                'correction_value': _column_name_getter(_sector_calculated_yaml, 'CORRECTION_VALUE'),
                '1d_return': _column_name_getter(_sector_calculated_yaml, '1D_RETURN'),
                '1d_rate': _column_name_getter(_sector_calculated_yaml, '1D_RATE'),
                'sector_open': _column_name_getter(_sector_calculated_yaml, 'OPEN'),
                'sector_close': _column_name_getter(_sector_calculated_yaml, 'CLOSE'),
                'sector_high': _column_name_getter(_sector_calculated_yaml, 'HIGH'),
                'sector_low': _column_name_getter(_sector_calculated_yaml, 'LOW'),
                     }
            return col


    @staticmethod
    def calc_new_sector_price(stock_dfs_dict:dict, SECTOR_REDEFINITIONS_CSV:str, SECTOR_INDEX_PARQUET:str) -> tuple[pd.DataFrame, pd.DataFrame]:
        '''
        セクターインデックスを算出します。
        Args:
            stock_dfs_dict (dict): 'list', 'fin', 'price'が定義されたデータフレーム
            SECTOR_REDEFINITIONS_CSV (str): セクター定義の設定ファイルのパス
            SECTOR_INDEX_PARQUET (str): セクターインデックスを出力するparquetファイルのパス
        Returns:
            pd.DataFrame: セクターインデックスを格納
            pd.DataFrame: 発注用に、個別銘柄の終値と時価総額を格納
        '''
        col = SectorIndexCalculator._get_column_names()
        stock_price = stock_dfs_dict['price']
        stock_fin = stock_dfs_dict['fin']
        #価格情報に発行済み株式数の情報を結合
        stock_price_for_order = SectorIndexCalculator.calc_marketcap(stock_price, stock_fin)
        #インデックス値の算出
        new_sector_price = SectorIndexCalculator._calc_index_value(stock_price_for_order, SECTOR_REDEFINITIONS_CSV)
        #データフレームを保存して、インデックスを設定
        new_sector_price = new_sector_price.reset_index()
        new_sector_price.to_parquet(SECTOR_INDEX_PARQUET)
        new_sector_price = new_sector_price.set_index([col['sector_date'], col['sector']])

        stock_price_for_order = \
            stock_price_for_order[[col['sector_date'], col['sector_code'], col['market_cap_close'], col['sector_close'], col['volume']]]
        print('セクターのインデックス値の算出が完了しました。')
        #TODO 2つのデータフレームを返す関数は分けたほうがよさそう。

        return new_sector_price, stock_price_for_order

    @staticmethod
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
        stock_price_with_shares = SectorIndexCalculator._merge_stock_price_and_shares(stock_price, stock_fin)
        # 発行済み株式数の補正係数を算出
        stock_price_cap = SectorIndexCalculator._calc_adjustment_factor(stock_price_with_shares, stock_price)
        stock_price_cap = SectorIndexCalculator._adjust_shares(stock_price_cap)
        # 時価総額と指数計算用の補正値を算出
        stock_price_cap = SectorIndexCalculator._calc_marketcap(stock_price_cap)
        stock_price_cap = SectorIndexCalculator._calc_correction_value(stock_price_cap)
        
        return stock_price_cap

    @staticmethod
    def _merge_stock_price_and_shares(stock_price: pd.DataFrame, stock_fin: pd.DataFrame) -> pd.DataFrame:
        """
        期末日以降最初の営業日時点での発行済株式数を結合。
        Args:
            stock_price (pd.DataFrame): 価格情報
            stock_fin (pd.DataFrame): 財務情報
        Returns:
            pd.DataFrame: 価格情報に発行済株式数を付記
        """
        col = SectorIndexCalculator._get_column_names()
        business_days = stock_price[col['price_date']].unique()
        shares_df = SectorIndexCalculator._calc_shares_at_end_period(stock_fin)
        shares_df = SectorIndexCalculator._append_next_period_start_date(shares_df, business_days)
        merged_df = SectorIndexCalculator._merge_with_stock_price(stock_price, shares_df)
        return merged_df

    @staticmethod
    def _calc_shares_at_end_period(stock_fin: pd.DataFrame) -> pd.DataFrame:
        """
        期末日時点での発行済株式数を計算する。
        Args:
            stock_fin (pd.DataFrame): 財務情報
        Returns:
            pd.DataFrame: 財務情報に発行済株式数を付記
        """
        col = SectorIndexCalculator._get_column_names()
        shares_df = stock_fin[[col['fin_code'], col['fin_date'], col['fin_outstanding_shares'], col['period_end_date']]].copy()
        shares_df = shares_df.sort_values(col['fin_date']).drop(col['fin_date'], axis=1)
        shares_df = shares_df.drop_duplicates(subset=[col['period_end_date'], col['fin_code']], keep='last')
        shares_df['NextPeriodStartDate'] = pd.to_datetime(shares_df[col['period_end_date']]) + timedelta(days=1)
        shares_df['isSettlementDay'] = True #あとで価格情報から抽出するために決算日フラグを設定しておく。
        return shares_df

    @staticmethod
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
            SectorIndexCalculator._find_next_business_day, business_days=business_days
        )
        return shares_df

    @staticmethod
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

    @staticmethod
    def _merge_with_stock_price(stock_price: pd.DataFrame, shares_df: pd.DataFrame) -> pd.DataFrame:
        """
        価格データに発行済株式数情報を結合する。
        Args:
            stock_price (pd.DataFrame): 価格情報データフレーム
            shares_df (pd.DataFrame): 発行済株式数を含むデータフレーム
        Returns:
            pd.DataFrame: 結合されたデータフレーム
        """
        col = SectorIndexCalculator._get_column_names()
        stock_price = stock_price.rename(columns={col['price_code']: col['sector_code'], col['price_date']: col['sector_date']})
        shares_df = shares_df.rename(columns={col['fin_code']: col['sector_code'], 'NextPeriodStartDate': col['sector_date']})

        merged_df = pd.merge(stock_price, 
                             shares_df[[col['sector_date'], col['sector_code'], col['fin_outstanding_shares'], 'isSettlementDay']],
                            on=[col['sector_date'], col['sector_code']],
                            how='left'
                            )
        merged_df.rename(columns={col['fin_outstanding_shares']: col['sector_outstanding_shares'],
                                  col['price_open']: col['sector_open'],
                                  col['price_close']: col['sector_close'],
                                  col['price_high']: col['sector_high'],
                                  col['price_low']: col['sector_low']})
        merged_df['isSettlementDay'] = merged_df['isSettlementDay'].astype(bool).fillna(False)
        return merged_df

    @staticmethod
    def _calc_adjustment_factor(stock_price_with_shares: pd.DataFrame, stock_price: pd.DataFrame) -> pd.DataFrame:
        """株式分割・併合による発行済み株式数の変化を調整します。
        Args:
            stock_price_with_shares (pd.DataFrame): 価格情報に発行済み株式数を併記
            stock_price (pd.DataFrame): 価格情報
        Returns:
            pd.DataFrame: 調整後の価格情報データフレーム
        """
        stock_price_to_adjust = SectorIndexCalculator._extract_rows_to_adjust(stock_price_with_shares)
        stock_price_to_adjust = SectorIndexCalculator._calc_shares_rate(stock_price_to_adjust)
        adjusted_stock_price = SectorIndexCalculator._correct_shares_rate_for_non_adjustment(stock_price_to_adjust)
        stock_price = SectorIndexCalculator._merge_shares_rate(stock_price, adjusted_stock_price)
        stock_price = SectorIndexCalculator._handle_special_cases(stock_price)
        return SectorIndexCalculator._calc_cumulative_shares_rate(stock_price)

    @staticmethod
    def _extract_rows_to_adjust(stock_price_with_shares_df: pd.DataFrame) -> pd.DataFrame:
        """
        株式分割・併合の対象行を抽出します。
        Args:
            stock_price_with_shares_df (pd.DataFrame): 価格情報に発行済み株式数を併記
        Returns:
            pd.DataFrame: 引数のdfから株式分割・併合対象行のみを抽出
        """
        col = SectorIndexCalculator._get_column_names()
        condition = (stock_price_with_shares_df[col['sector_outstanding_shares']].notnull() | (stock_price_with_shares_df[col['adjustment_factor']] != 1))
        return stock_price_with_shares_df.loc[condition].copy()

    @staticmethod
    def _calc_shares_rate(df: pd.DataFrame) -> pd.DataFrame:
        """
        株式分割・併合による発行済み株式数の変化率を計算します。
        Args:
            df (pd.DataFrame): 株式分割・併合対象行のデータフレーム
        Returns:
            pd.DataFrame: 発行済み株式数の比率を計算したデータフレーム
        """
        col = SectorIndexCalculator._get_column_names()
        df[col['sector_outstanding_shares']] = df.groupby(col['sector_code'])[col['sector_outstanding_shares']].bfill()
        df['SharesRate'] = (df.groupby(col['sector_code'])[col['sector_outstanding_shares']].shift(-1) / df[col['sector_outstanding_shares']]).round(1)
        return df

    @staticmethod
    def _correct_shares_rate_for_non_adjustment(df: pd.DataFrame) -> pd.DataFrame:
        """
        株式分割・併合由来でない発行済み株式数変更（株価補正なし）について、補正比率を1に修正します。
        Args:
            df (pd.DataFrame): 株式分割・併合対象行のデータフレーム
        Returns:
            pd.DataFrame: 調整後のデータフレーム
        """
        col = SectorIndexCalculator._get_column_names()
        # 補正係数の出るタイミングと実際に補正の必要なタイミングが±2データぶんずれる場合がある。
        shift_days = [1, 2, -1, -2]
        shift_columns = [f'Shift_AdjustmentFactor{i}' for i in shift_days]
        for shift_column, i in zip(shift_columns, shift_days):
            df[shift_column] = df.groupby(col['sector_code'])[col['adjustment_factor']].shift(i).fillna(1)
        df.loc[((df[shift_columns] == 1).all(axis=1) | (df['SharesRate'] == 1)), 'SharesRate'] = 1
        return df

    @staticmethod
    def _merge_shares_rate(stock_price: pd.DataFrame, df_to_calc_shares_rate: pd.DataFrame) -> pd.DataFrame:
        """
        株価調整用の発行済株式数比率を元の価格情報データフレームに結合。
        Args:
            stock_price (pd.DataFrame): 価格情報
            df_to_calc_shares_rate (pd.DataFrame): 発行済株式数比率を含むデータフレーム
        Returns:
            pd.DataFrame: 価格情報に発行済株式数比率を併記
        """
        col = SectorIndexCalculator._get_column_names()
        df_to_calc_shares_rate = df_to_calc_shares_rate[df_to_calc_shares_rate['isSettlementDay']]
        df_to_calc_shares_rate['SharesRate'] = df_to_calc_shares_rate.groupby(col['sector_code'])['SharesRate'].shift(1)
        stock_price = pd.merge(
            stock_price,
            df_to_calc_shares_rate[[col['sector_date'], col['sector_code'], col['sector_outstanding_shares'], 'SharesRate']],
            how='left',
            on=[col['sector_date'], col['sector_code']]
        )
        stock_price['SharesRate'] = stock_price.groupby(col['sector_code'])['SharesRate'].shift(-1)
        stock_price['SharesRate'] = stock_price['SharesRate'].fillna(1)
        return stock_price

    @staticmethod
    def _handle_special_cases(stock_price: pd.DataFrame) -> pd.DataFrame:
        """
        最初の決算発表データ前の発行済株式数比率を手動で補正します。
        Args:
            stock_price (pd.DataFrame): 価格情報データフレーム
        Returns:
            pd.DataFrame: 補正後のデータフレーム
        """
        col = SectorIndexCalculator._get_column_names()
        stock_price.loc[(stock_price[col['sector_code']] == '3064') & (stock_price[col['sector_date']] <= datetime(2013, 7, 25)), 'SharesRate'] = 1
        stock_price.loc[(stock_price[col['sector_code']] == '6920') & (stock_price[col['sector_date']] <= datetime(2013, 8, 9)), 'SharesRate'] = 1
        return stock_price

    @staticmethod
    def _calc_cumulative_shares_rate(stock_price: pd.DataFrame) -> pd.DataFrame:
        """
        発行済み株式数比率の累積積を計算します。
        Args:
            stock_price (pd.DataFrame): 価格情報データフレーム
        Returns:
            pd.DataFrame: 累積積を計算したデータフレーム
        """
        col = SectorIndexCalculator._get_column_names()
        stock_price = stock_price.sort_values(col['sector_date'], ascending=False)
        stock_price['CumulativeSharesRate'] = stock_price.groupby(col['sector_code'])['SharesRate'].cumprod()
        stock_price = stock_price.sort_values(col['sector_date'], ascending=True)
        stock_price['CumulativeSharesRate'] = stock_price['CumulativeSharesRate'].fillna(1)
        return stock_price

    @staticmethod
    def _adjust_shares(df:pd.DataFrame) -> pd.DataFrame:
        '''
        発行済株式数を調整します。
        Args:
            df (pd.DataFrame): 価格情報に補正係数を併記
        Returns:
            pd.DataFrame: 引数のdfの発行済株式数を補正したもの
        '''
        col = SectorIndexCalculator._get_column_names()
        #決算発表時以外が欠測値なので、後埋めする。
        df[col['sector_outstanding_shares']] = df.groupby(col['sector_code'], as_index=False)[col['sector_outstanding_shares']].ffill() 
        #初回決算発表以前の分を前埋め。
        df[col['sector_outstanding_shares']] = df.groupby(col['sector_code'], as_index=False)[col['sector_outstanding_shares']].bfill() 
        df[col['sector_outstanding_shares']] = df[col['sector_outstanding_shares']] * df['CumulativeSharesRate'] #株式併合・分割以降、決算発表までの期間の発行済み株式数を調整
        #不要行の削除
        return df.drop(['SharesRate', 'CumulativeSharesRate'], axis=1)

    @staticmethod
    def _calc_marketcap(df: pd.DataFrame) -> pd.DataFrame:
        '''
        時価総額を算出します。
        Args:
            df (pd.DataFrame): 価格情報に発行済株式数（補正済）を併記
        Returns:
            pd.DataFrame: 引数のdfに時価総額のOHLCを追加したもの
        '''
        col = SectorIndexCalculator._get_column_names()
        df[col['market_cap_open']] = df[col['sector_open']] * df[col['sector_outstanding_shares']]
        df[col['market_cap_close']] = df[col['sector_close']] * df[col['sector_outstanding_shares']]
        df[col['market_cap_high']] = df[col['sector_high']] * df[col['sector_outstanding_shares']]
        df[col['market_cap_low']] = df[col['sector_low']] * df[col['sector_outstanding_shares']]
        return df

    @staticmethod
    def _calc_correction_value(df: pd.DataFrame) -> pd.DataFrame:
        '''
        指数算出補正値を算出します。
        Args:
            df (pd.DataFrame): 価格情報に時価総額を併記
        Returns:
            pd.DataFrame: 引数のdfに指数算出補正値を併記したもの
        '''
        col = SectorIndexCalculator._get_column_names()
        df['OutstandingShares_forCorrection'] = \
            df.groupby(col['sector_code'])[col['sector_outstanding_shares']].shift(1)
        df['OutstandingShares_forCorrection'] = df['OutstandingShares_forCorrection'].fillna(0)
        df['MarketCapClose_forCorrection'] = df[col['sector_close']] * df['OutstandingShares_forCorrection']
        df[col['correction_value']] = df[col['market_cap_close']] - df['MarketCapClose_forCorrection']
        return df

    @staticmethod
    def _calc_index_value(stock_price: pd.DataFrame, SECTOR_REDEFINITIONS_CSV:str) -> pd.DataFrame:
        '''
        指標を算出します。
        Args:
            stock_price (pd.DataFrame): 価格情報に時価総額と指数算出補正値を併記
            SECTOR_REDEFINITIONS_CSV (str): セクター定義の設定ファイルのパス
        Returns:
            pd.DataFrame: セクターインデックス
        '''
        col = SectorIndexCalculator._get_column_names()
        new_sector_list = pd.read_csv(SECTOR_REDEFINITIONS_CSV).dropna(how='any', axis=1)
        new_sector_list[col['sector_code']] = new_sector_list[col['sector_code']].astype(str)
        new_sector_price = pd.merge(new_sector_list, stock_price, how='right', on=col['sector_code'])
        # TODO SECTOR_REDEFINITIONS_CSVのyamlデータも作らないといけない。

        #必要列を抜き出したデータフレームを作る。
        new_sector_price = new_sector_price.groupby([col['sector_date'], col['sector']])\
        [[col['market_cap_open'], col['market_cap_close'],col['market_cap_high'], col['market_cap_low'], col['sector_outstanding_shares'], col['correction_value']]].sum()
        new_sector_price[col['1d_return']] = new_sector_price[col['market_cap_close']] \
                                        / (new_sector_price.groupby(col['sector'])[col['market_cap_close']].shift(1) \
                                        + new_sector_price[col['correction_value']]) - 1
        new_sector_price[col['1d_rate']] = 1 + new_sector_price[col['1d_return']]

        #初日の終値を1とすると、各日の終値は1d_rateのcumprodで求められる。→OHLは、Cとの比率で求められる。
        new_sector_price[col['sector_close']] = new_sector_price.groupby(col['sector'])[col['1d_rate']].cumprod()
        new_sector_price[col['sector_open']] = \
            new_sector_price[col['sector_close']]  * new_sector_price[col['market_cap_open']] / new_sector_price[col['market_cap_close']]
        new_sector_price[col['sector_high']] = \
            new_sector_price[col['sector_close']]  * new_sector_price[col['market_cap_high']] / new_sector_price[col['market_cap_close']]
        new_sector_price[col['sector_low']] = \
            new_sector_price[col['sector_close']]  * new_sector_price[col['market_cap_low']] / new_sector_price[col['market_cap_close']]

        return new_sector_price
        

if __name__ == '__main__':
    from facades.stock_acquisition_facade import StockAcquisitionFacade
    acq = StockAcquisitionFacade(filter = "(Listing==1)&((ScaleCategory=='TOPIX Core30')|(ScaleCategory=='TOPIX Large70')|(ScaleCategory=='TOPIX Mid400'))")
    stock_dfs = acq.get_stock_data_dict()

    sic = SectorIndexCalculator()
    sector_price_df, order_price_df = sic.calc_new_sector_price(stock_dfs, 
                                            f'{Paths.SECTOR_REDEFINITIONS_FOLDER}/48sectors_2024-2025.csv', 
                                            f'{Paths.SECTOR_REDEFINITIONS_FOLDER}/New48sectors_price_test.parquet')
    print(sector_price_df)