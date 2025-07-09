from abc import ABC, abstractmethod
import pandas as pd

class CalculationMethodBase(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def calculate(self, return_timeseries_df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        pass