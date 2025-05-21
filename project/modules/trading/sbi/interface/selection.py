from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Literal
import pandas as pd

@dataclass
class SectorPrediction:
    """セクター予測結果"""
    Sector: str
    PredictionValue: float
    Rank: int
    isBuyable: bool = False
    isSellable: bool = False

@dataclass
class StockWeight:
    """銘柄ウェイト情報"""
    Code: str
    Sector: str
    CompanyName: str
    Weight: float
    EstimatedCost: float
    
@dataclass
class OrderUnit:
    """注文単位情報"""
    Code: str
    CompanyName: str
    Sector: str
    Unit: int
    EstimatedCost: float
    TotalCost: float
    UpperLimitCost: float
    UpperLimitTotal: float
    Direction: Literal['Long', 'Short']
    isBorrowingStock: bool = False

class ISectorProvider(ABC):
    """セクター情報提供インターフェース"""
    @abstractmethod
    def get_sector_definitions(self) -> pd.DataFrame:
        """セクター定義情報を取得"""
        pass

class IPriceProvider(ABC):
    """価格情報提供インターフェース"""
    @abstractmethod
    def get_price_data(self) -> pd.DataFrame:
        """価格データを取得"""
        pass
    
    @abstractmethod
    def get_etf_price(self, symbol_code: str) -> float:
        """指定したETFの価格を取得"""
        pass

class ITradeLimitProvider(ABC):
    """取引制限情報提供インターフェース"""
    @abstractmethod
    async def get_buyable_symbols(self) -> pd.DataFrame:
        """買い可能な銘柄リストを取得"""
        pass
    
    @abstractmethod
    async def get_sellable_symbols(self) -> pd.DataFrame:
        """売り可能な銘柄リストを取得"""
        pass
    
    @abstractmethod
    async def get_margin_power(self) -> float:
        """信用建余力を取得"""
        pass

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

class IStockSelector(ABC):
    """銘柄選択インターフェース"""
    @abstractmethod
    async def select_stocks(self, margin_power: Optional[float] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """取引銘柄を選択"""
        pass