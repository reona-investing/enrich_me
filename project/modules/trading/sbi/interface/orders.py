from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional, List, Dict, Any, Tuple, Union
import pandas as pd

@dataclass
class OrderRequest:
    """注文リクエストのデータコンテナ"""
    symbol_code: str
    unit: int  # 単位数（100株単位）
    direction: Literal['Long', 'Short']
    estimated_price: float
    is_borrowing_stock: bool = False
    order_type: Literal["指値", "成行", "逆指値"] = "成行"
    order_type_value: Optional[Literal["寄指", "引指", "不成", "IOC指", "寄成", "引成", "IOC成"]] = None
    limit_price: Optional[float] = None
    trigger_price: Optional[float] = None


@dataclass
class OrderResult:
    """注文結果のデータコンテナ"""
    success: bool
    order_id: Optional[str] = None
    message: str = ""
    error_code: Optional[str] = None


class IOrderExecutor(ABC):
    """注文実行のインターフェース"""
    
    @abstractmethod
    async def place_order(self, order_request: OrderRequest) -> OrderResult:
        """注文を発注する"""
        pass
    
    @abstractmethod
    async def cancel_all_orders(self, order_id: str) -> OrderResult:
        """注文をキャンセルする"""
        pass
    
    @abstractmethod
    async def settle_position(self, symbol_code: str, unit: Optional[int] = None) -> OrderResult:
        """ポジションを決済する"""
        pass
    
    @abstractmethod
    async def get_active_orders(self) -> pd.DataFrame:
        """有効な注文一覧を取得する"""
        pass
    
    @abstractmethod
    async def get_positions(self) -> pd.DataFrame:
        """現在のポジション一覧を取得する"""
        pass


class IMarginProvider(ABC):
    """証拠金情報提供のインターフェース"""
    
    @abstractmethod
    async def get_available_margin(self) -> float:
        """利用可能な証拠金を取得する"""
        pass
    
    @abstractmethod
    async def refresh(self) -> None:
        """証拠金情報を更新する"""
        pass