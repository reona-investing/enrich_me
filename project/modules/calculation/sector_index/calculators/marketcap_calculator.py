from datetime import datetime, timedelta
from typing import Callable

import numpy as np
import pandas as pd


class MarketCapCalculator:
    """株価データに発行済み株式数を付与し時価総額を計算する補助クラス。"""

    @staticmethod
    def merge_stock_price_and_shares(stock_price: pd.DataFrame, stock_fin: pd.DataFrame, col_getter: Callable):
        """株価データに発行済み株式数情報を結合する。"""

        _, price_col, _ = col_getter()
        business_days = stock_price[price_col['日付']].unique()
        shares_df = MarketCapCalculator._calc_shares_at_end_period(stock_fin, col_getter)
        shares_df = MarketCapCalculator._append_next_period_start_date(shares_df, business_days)
        return MarketCapCalculator._merge_with_stock_price(stock_price, shares_df, col_getter)

    @staticmethod
    def _calc_shares_at_end_period(stock_fin: pd.DataFrame, col_getter: Callable) -> pd.DataFrame:
        """決算期末時点の発行済み株式数を抽出する。"""

        fin_col, _, _ = col_getter()
        shares_df = stock_fin[[fin_col['銘柄コード'], fin_col['日付'], fin_col['発行済み株式数'], fin_col['当会計期間終了日']]].copy()
        shares_df = shares_df.sort_values(fin_col['日付']).drop(fin_col['日付'], axis=1)
        shares_df = shares_df.drop_duplicates(subset=[fin_col['当会計期間終了日'], fin_col['銘柄コード']], keep='last')
        shares_df['NextPeriodStartDate'] = pd.to_datetime(shares_df[fin_col['当会計期間終了日']]) + timedelta(days=1)
        shares_df['isSettlementDay'] = True
        return shares_df

    @staticmethod
    def _append_next_period_start_date(shares_df: pd.DataFrame, business_days: np.ndarray) -> pd.DataFrame:
        """次会計期間の開始日を営業日に丸める。"""

        shares_df['NextPeriodStartDate'] = shares_df['NextPeriodStartDate'].apply(
            MarketCapCalculator._find_next_business_day, business_days=business_days
        )
        return shares_df

    @staticmethod
    def _find_next_business_day(date: pd.Timestamp, business_days: np.ndarray) -> pd.Timestamp:
        """与えられた日付以降で最初の営業日を返す。"""
        if pd.isna(date):
            return date
        while date not in business_days:
            date += np.timedelta64(1, 'D')
        return date

    @staticmethod
    def _merge_with_stock_price(stock_price: pd.DataFrame, shares_df: pd.DataFrame, col_getter: Callable) -> pd.DataFrame:
        """株価データと株式数情報をマージする。"""

        fin_col, price_col, sector_col = col_getter()
        stock_price = stock_price.rename(columns={price_col['銘柄コード']: sector_col['銘柄コード'], price_col['日付']: sector_col['日付']})
        shares_df = shares_df.rename(columns={fin_col['銘柄コード']: sector_col['銘柄コード'], 'NextPeriodStartDate': sector_col['日付']})
        merged_df = pd.merge(
            stock_price,
            shares_df[[sector_col['日付'], sector_col['銘柄コード'], fin_col['発行済み株式数'], 'isSettlementDay']],
            on=[sector_col['日付'], sector_col['銘柄コード']],
            how='left'
        )
        merged_df.rename(columns={fin_col['発行済み株式数']: sector_col['発行済み株式数'],
                                  price_col['始値']: sector_col['始値'],
                                  price_col['終値']: sector_col['終値'],
                                  price_col['高値']: sector_col['高値'],
                                  price_col['安値']: sector_col['安値']})
        merged_df['isSettlementDay'] = merged_df['isSettlementDay'].astype(bool).fillna(False)
        return merged_df

    @staticmethod
    def calc_adjustment_factor(stock_price_with_shares: pd.DataFrame, stock_price: pd.DataFrame, col_getter: Callable) -> pd.DataFrame:
        """株式分割などの調整係数を計算する。"""

        stock_price_to_adjust = MarketCapCalculator._extract_rows_to_adjust(stock_price_with_shares, col_getter)
        stock_price_to_adjust = MarketCapCalculator._calc_shares_rate(stock_price_to_adjust, col_getter)
        adjusted_stock_price = MarketCapCalculator._correct_shares_rate_for_non_adjustment(stock_price_to_adjust, col_getter)
        stock_price = MarketCapCalculator._merge_shares_rate(stock_price, adjusted_stock_price, col_getter)
        stock_price = MarketCapCalculator._handle_special_cases(stock_price, col_getter)
        return MarketCapCalculator._calc_cumulative_shares_rate(stock_price, col_getter)

    @staticmethod
    def _extract_rows_to_adjust(stock_price_with_shares_df: pd.DataFrame, col_getter: Callable) -> pd.DataFrame:
        """分割調整が必要な行を抽出する。"""

        _, price_col, sector_col = col_getter()
        condition = (
            stock_price_with_shares_df[sector_col['発行済み株式数']].notnull() |
            (stock_price_with_shares_df[price_col['調整係数']] != 1)
        )
        return stock_price_with_shares_df.loc[condition].copy()

    @staticmethod
    def _calc_shares_rate(df: pd.DataFrame, col_getter: Callable) -> pd.DataFrame:
        """株式数変化率を計算する。"""

        _, _, sector_col = col_getter()
        df[sector_col['発行済み株式数']] = df.groupby(sector_col['銘柄コード'])[sector_col['発行済み株式数']].bfill()
        df['SharesRate'] = (
            df.groupby(sector_col['銘柄コード'])[sector_col['発行済み株式数']].shift(-1) / df[sector_col['発行済み株式数']]
        ).round(1)
        return df

    @staticmethod
    def _correct_shares_rate_for_non_adjustment(df: pd.DataFrame, col_getter: Callable) -> pd.DataFrame:
        """調整不要なケースを考慮して ``SharesRate`` を補正する。"""

        _, price_col, sector_col = col_getter()
        shift_days = [1, 2, -1, -2]
        shift_columns = [f'Shift_AdjustmentFactor{i}' for i in shift_days]
        for shift_column, i in zip(shift_columns, shift_days):
            df[shift_column] = df.groupby(sector_col['銘柄コード'])[price_col['調整係数']].shift(i).fillna(1)
        df.loc[((df[shift_columns] == 1).all(axis=1) | (df['SharesRate'] == 1)), 'SharesRate'] = 1
        return df

    @staticmethod
    def _merge_shares_rate(stock_price: pd.DataFrame, df_to_calc_shares_rate: pd.DataFrame, col_getter: Callable) -> pd.DataFrame:
        """株数変化率を株価データにマージする。"""

        _, _, sector_col = col_getter()
        df_to_calc_shares_rate = df_to_calc_shares_rate[df_to_calc_shares_rate['isSettlementDay']]
        df_to_calc_shares_rate['SharesRate'] = df_to_calc_shares_rate.groupby(sector_col['銘柄コード'])['SharesRate'].shift(1)
        stock_price = pd.merge(
            stock_price,
            df_to_calc_shares_rate[[sector_col['日付'], sector_col['銘柄コード'], sector_col['発行済み株式数'], 'SharesRate']],
            how='left',
            on=[sector_col['日付'], sector_col['銘柄コード']]
        )
        stock_price['SharesRate'] = stock_price.groupby(sector_col['銘柄コード'])['SharesRate'].shift(-1)
        stock_price['SharesRate'] = stock_price['SharesRate'].fillna(1)
        return stock_price

    @staticmethod
    def _handle_special_cases(stock_price: pd.DataFrame, col_getter: Callable) -> pd.DataFrame:
        """特殊な銘柄の ``SharesRate`` を補正する。"""

        _, _, sector_col = col_getter()
        stock_price.loc[(stock_price[sector_col['銘柄コード']] == '3064') & (stock_price[sector_col['日付']] <= datetime(2013, 7, 25)), 'SharesRate'] = 1
        stock_price.loc[(stock_price[sector_col['銘柄コード']] == '6920') & (stock_price[sector_col['日付']] <= datetime(2013, 8, 9)), 'SharesRate'] = 1
        return stock_price

    @staticmethod
    def _calc_cumulative_shares_rate(stock_price: pd.DataFrame, col_getter: Callable) -> pd.DataFrame:
        """変化率を累積して連続的な補正率を算出する。"""

        _, _, sector_col = col_getter()
        stock_price = stock_price.sort_values(sector_col['日付'], ascending=False)
        stock_price['CumulativeSharesRate'] = stock_price.groupby(sector_col['銘柄コード'])['SharesRate'].cumprod()
        stock_price = stock_price.sort_values(sector_col['日付'], ascending=True)
        stock_price['CumulativeSharesRate'] = stock_price['CumulativeSharesRate'].fillna(1)
        return stock_price

    @staticmethod
    def adjust_shares(df: pd.DataFrame, col_getter: Callable) -> pd.DataFrame:
        """株式数を ``CumulativeSharesRate`` で補正する。"""

        _, _, sector_col = col_getter()
        df[sector_col['発行済み株式数']] = df.groupby(sector_col['銘柄コード'], as_index=False)[sector_col['発行済み株式数']].ffill()
        df[sector_col['発行済み株式数']] = df.groupby(sector_col['銘柄コード'], as_index=False)[sector_col['発行済み株式数']].bfill()
        df[sector_col['発行済み株式数']] = df[sector_col['発行済み株式数']] * df['CumulativeSharesRate']
        return df.drop(['SharesRate', 'CumulativeSharesRate'], axis=1)

    @staticmethod
    def calc_marketcap(df: pd.DataFrame, col_getter: Callable) -> pd.DataFrame:
        """OHLC 各値の時価総額を計算する。"""

        _, _, sector_col = col_getter()
        df[sector_col['始値時価総額']] = df[sector_col['始値']] * df[sector_col['発行済み株式数']]
        df[sector_col['終値時価総額']] = df[sector_col['終値']] * df[sector_col['発行済み株式数']]
        df[sector_col['高値時価総額']] = df[sector_col['高値']] * df[sector_col['発行済み株式数']]
        df[sector_col['安値時価総額']] = df[sector_col['安値']] * df[sector_col['発行済み株式数']]
        return df

    @staticmethod
    def calc_correction_value(df: pd.DataFrame, col_getter: Callable) -> pd.DataFrame:
        """指数算出用の補正値を計算する。"""

        _, _, sector_col = col_getter()
        df['OutstandingShares_forCorrection'] = df.groupby(sector_col['銘柄コード'])[sector_col['発行済み株式数']].shift(1)
        df['OutstandingShares_forCorrection'] = df['OutstandingShares_forCorrection'].fillna(0)
        df['MarketCapClose_forCorrection'] = df[sector_col['終値']] * df['OutstandingShares_forCorrection']
        df[sector_col['指数算出用の補正値']] = df[sector_col['終値時価総額']] - df['MarketCapClose_forCorrection']
        return df
