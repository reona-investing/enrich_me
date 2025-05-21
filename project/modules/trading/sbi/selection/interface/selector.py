from abc import ABC, abstractmethod
from typing import Optional, Tuple
import pandas as pd

class IStockSelector(ABC):
    """銘柄選択インターフェース"""
    @abstractmethod
    async def select_stocks(self, margin_power: Optional[float] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """取引銘柄を選択"""
        pass