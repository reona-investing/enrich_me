from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Literal
import pandas as pd
from .dataclasses import SectorPrediction, StockWeight

class ISectorAnalyzer(ABC):
    """セクター分析インターフェース"""
    @abstractmethod
    def analyze_sector_predictions(self, predictions_df: pd.DataFrame) -> List[SectorPrediction]:
        """セクター予測を分析"""
        pass
    
    @abstractmethod
    def select_sectors_to_trade(self, 
                              sector_predictions: List[SectorPrediction], 
                              num_sectors: int, 
                              num_candidates: int) -> Tuple[List[str], List[str]]:
        """取引対象セクターを選択"""
        pass

class IWeightCalculator(ABC):
    """ウェイト計算インターフェース"""
    @abstractmethod
    def calculate_weights(self, 
                         sector_definitions: pd.DataFrame, 
                         price_data: pd.DataFrame) -> List[StockWeight]:
        """銘柄ウェイトを計算"""
        pass
