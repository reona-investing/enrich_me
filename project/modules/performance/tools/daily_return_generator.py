from __future__ import annotations

import pandas as pd

class DailyReturnGenerator:
    """銘柄ごとのリターン実績から、日次リターンを算出する。"""

    def __init__(self, 
                 date_series: pd.Series, 
                 acquisition_price_series: pd.Series, 
                 settlement_price_series: pd.Series,
                 long_or_short_series: pd.Series,
                 short_keyphrase: str | bool) -> None:
        self._date_series = pd.to_datetime(date_series)
        self._acquisition_price_series = pd.Series(acquisition_price_series).astype(float)
        self._settlement_price_series = pd.Series(settlement_price_series).astype(float)
        self._long_or_short_series = pd.Series(long_or_short_series)
        self._short_keyphrase = short_keyphrase
        
        self._daily_return_df = self._generate()

    def get(self) -> pd.DataFrame:
        return self._daily_return_df

    def _generate(self) -> pd.DataFrame:
        df = pd.DataFrame({'Date': self._date_series, 
                           'Acquisition': self._acquisition_price_series,
                           'Settlement': self._settlement_price_series,
                           'LongOrShort': self._long_or_short_series})
        df['Coefficient'] = 1
        df.loc[df['LongOrShort'] == self._short_keyphrase, 'Coefficient'] = -1
        df['Profit'] = (df['Settlement'] - df['Acquisition']) * df['Coefficient']
        daily_return_df = df.groupby('Date')[['Profit', 'Acquisition']].sum()
        daily_return_df['Return'] = daily_return_df['Profit'] / daily_return_df['Acquisition']
        return daily_return_df[['Return']].sort_index()
