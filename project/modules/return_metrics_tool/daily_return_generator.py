from __future__ import annotations

import pandas as pd
from typing import Optional

class DailyReturnGenerator:
    """Create daily return series from raw returns."""

    def __init__(self, date_series: pd.Series, return_series: pd.Series, sector_series: Optional[pd.Series] = None) -> None:
        self.date_series = pd.to_datetime(date_series)
        self.return_series = pd.Series(return_series).astype(float)
        self.sector_series = pd.Series(sector_series) if sector_series is not None else None

    def generate(self) -> pd.Series:
        df = pd.DataFrame({'Date': self.date_series, 'Return': self.return_series})
        if self.sector_series is not None:
            df['Sector'] = self.sector_series.values
            daily = df.groupby('Date')['Return'].mean()
        else:
            daily = df.set_index('Date')['Return']
        return daily.sort_index()

    def generate_label_series(self, label_series: pd.Series) -> pd.Series:
        df = pd.DataFrame({'Date': self.date_series, 'Label': label_series})
        if self.sector_series is not None:
            df['Sector'] = self.sector_series.values
            daily = df.groupby('Date')['Label'].mean()
        else:
            daily = df.set_index('Date')['Label']
        return daily.sort_index()
