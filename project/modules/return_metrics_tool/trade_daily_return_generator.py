from __future__ import annotations

import pandas as pd
from typing import Optional

class TradeDailyReturnGenerator:
    """Calculate daily returns from trade data."""

    def __init__(
        self,
        date_series: pd.Series,
        settlement_price_series: pd.Series,
        acquisition_price_series: pd.Series,
        side_series: pd.Series,
        short_keyword: str,
        sector_series: Optional[pd.Series] = None,
    ) -> None:
        self.date_series = pd.to_datetime(date_series)
        self.settlement_series = pd.Series(settlement_price_series).astype(float)
        self.acquisition_series = pd.Series(acquisition_price_series).astype(float)
        self.side_series = pd.Series(side_series)
        self.sector_series = pd.Series(sector_series) if sector_series is not None else None
        self.short_keyword = short_keyword

    def generate(self) -> pd.Series:
        df = pd.DataFrame({
            'Date': self.date_series,
            'Settlement': self.settlement_series,
            'Acquisition': self.acquisition_series,
            'Side': self.side_series,
        })
        if self.sector_series is not None:
            df['Sector'] = self.sector_series.values

        diff = df['Settlement'] - df['Acquisition']
        diff[df['Side'] == self.short_keyword] *= -1

        if self.sector_series is not None:
            num = diff.groupby([df['Date'], df['Sector']]).sum()
            den = df.groupby([df['Date'], df['Sector']])['Acquisition'].sum()
            daily = num / den
            daily.index.names = ['Date', 'Sector']
        else:
            num = diff.groupby(df['Date']).sum()
            den = df.groupby(df['Date'])['Acquisition'].sum()
            daily = num / den
            daily.index.name = 'Date'
        return daily.sort_index()

