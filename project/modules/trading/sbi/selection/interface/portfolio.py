from abc import ABC, abstractmethod
from typing import List, Literal
import pandas as pd
from .dataclasses import StockWeight, OrderUnit

class IOrderAllocator(ABC):
    """注文配分インターフェース"""
    @abstractmethod
    def allocate_orders(self, 
                       weights: List[StockWeight], 
                       target_sectors: List[str], 
                       tradable_symbols: pd.DataFrame, 
                       margin_power: float, 
                       direction: Literal['Long', 'Short']) -> List[OrderUnit]:
        """注文を配分"""
        pass