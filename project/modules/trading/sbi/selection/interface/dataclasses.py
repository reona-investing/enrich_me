from dataclasses import dataclass
from typing import Literal

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
    isBorrowingStock: bool = True