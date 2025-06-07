from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional, List
import pandas as pd
from pydantic import BaseModel, Field


class OrderRequest(BaseModel):
    """注文リクエストのデータコンテナ"""
    direction: Literal['Long', 'Short']
    symbol_code: str = Field(..., description="銘柄コード")
    trade_type: Literal["現物買", "現物売", "信用新規買", "信用新規売"] = Field("信用新規買", description="取引タイプ")
    unit: int = Field(100, gt=0, description="取引単位（正の整数）")
    order_type: Literal["指値", "成行", "逆指値"] = Field("成行", description="注文タイプ")
    order_type_value: Literal["寄指", "引指", "不成", "IOC指", "寄成", "引成", "IOC成", None] = Field("寄成", description="注文詳細")
    limit_price: Optional[float] = Field(None, gt=0, description="指値注文の価格")
    trigger_price: Optional[float] = Field(None, gt=0, description="逆指値のトリガー価格")
    stop_order_type: Literal["指値", "成行"] = Field("成行", description="逆指値のタイプ")
    stop_order_price: Optional[float] = Field(None, gt=0, description="逆指値注文の価格")
    period_type: Literal["当日中", "今週中", "期間指定"] = Field("当日中", description="注文期間のタイプ")
    period_value: Optional[str] = Field(None, description="期間指定時の日付")
    period_index: Optional[int] = Field(None, description="期間指定時のインデックス")
    trade_section: Literal["特定預り", "一般預り", "NISA預り", "旧NISA預り"] = Field("特定預り", description="取引区分")
    margin_trade_section: Literal["制度", "一般", "日計り"] = Field("制度", description="信用取引区分")


@dataclass
class OrderResult:
    """注文結果のデータコンテナ"""
    success: bool
    order_id: Optional[str] = None
    message: str = ""
    error_code: Optional[str] = None


class IOrderPlacer(ABC):
    @abstractmethod
    async def place_order(self, order_request: OrderRequest) -> OrderResult:
        pass


class IOrderCanceller(ABC):
    @abstractmethod
    async def cancel_all_orders(self) -> List[OrderResult]:
        pass


class IPositionSettler(ABC):
    @abstractmethod
    async def settle_position(self, symbol_code: str, unit: Optional[int] = None) -> OrderResult:
        pass

    @abstractmethod
    async def get_positions(self) -> pd.DataFrame:
        pass


class IOrderInquiry(ABC):
    @abstractmethod
    async def get_active_orders(self) -> pd.DataFrame:
        pass


class IOrderExecutor(IOrderPlacer, IOrderCanceller, IPositionSettler, IOrderInquiry, ABC):
    """注文実行全体のインターフェース"""
    pass
