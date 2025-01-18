from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator

class TradeParameters(BaseModel):
    symbol_code: str = Field(..., description="銘柄コード")
    trade_type: Literal["現物買", "現物売", "信用新規買", "信用新規売"] = Field("信用新規買", description="取引タイプ")
    unit: int = Field(100, gt=0, description="取引単位（正の整数）")
    order_type: Literal["指値", "成行", "逆指値"] = Field("成行", description="注文タイプ")
    order_type_value: Literal["寄指", "引指", "不成", "IOC指", "寄成", "引成", "IOC成", None] = Field("寄成", description="注文詳細")
    limit_order_price: Optional[float] = Field(None, gt=0, description="指値注文の価格")
    stop_order_trigger_price: Optional[float] = Field(None, gt=0, description="逆指値のトリガー価格")
    stop_order_type: Literal["指値", "成行"] = Field("成行", description="逆指値のタイプ")
    stop_order_price: Optional[float] = Field(None, gt=0, description="逆指値注文の価格")
    period_type: Literal["当日中", "今週中", "期間指定"] = Field("当日中", description="注文期間のタイプ")
    period_value: Optional[str] = Field(None, description="期間指定時の日付")
    period_index: Optional[int] = Field(None, description="期間指定時のインデックス")
    trade_section: Literal["特定預り", "一般預り", "NISA預り", "旧NISA預り"] = Field("特定預り", description="取引区分")
    margin_trade_section: Literal["制度", "一般", "日計り"] = Field("制度", description="信用取引区分")

    @field_validator("limit_order_price", "stop_order_trigger_price", "stop_order_price")
    def validate_positive_price(cls, value):
        """価格が正の数であることを確認"""
        if value is not None and value <= 0:
            raise ValueError("価格は正の数でなければなりません。")
        return value

    @field_validator("period_value")
    def validate_period_value(cls, value, values):
        """期間指定時に日付が設定されているかを確認"""
        if values.data.get("period_type") == "期間指定" and not value:
            raise ValueError("期間指定タイプの場合、period_valueを設定してください。")
        return value

    @field_validator("order_type_value")
    def validate_order_type_value(cls, value, values):
        """order_typeに応じてorder_type_valueの選択肢を絞る"""
        order_type = values.data.get("order_type")
        if order_type == "指値" and value not in ["寄指", "引指", "不成", "IOC指", None]:
            raise ValueError(f"order_typeが'指値'の場合、order_type_valueは'寄指', '引指', '不成', 'IOC指', Noneのいずれかにしてください。")
        elif order_type == "成行" and value not in ["寄成", "引成", "IOC成", None]:
            raise ValueError(f"order_typeが'成行'の場合、order_type_valueは'寄成', '引成', 'IOC成', Noneのいずれかにしてください。")
        return value

    @field_validator("limit_order_price")
    def validate_limit_order_price(cls, value, values):
        """order_typeが指値のとき、limit_order_priceを必須入力とする"""
        if values.data.get("order_type") == "指値" and value is None:
            raise ValueError("order_typeが'指値'の場合、limit_order_priceを設定してください。")
        return value

    @field_validator("stop_order_trigger_price")
    def validate_stop_order_trigger_price(cls, value, values):
        """order_typeが逆指値の時、stop_order_trigger_priceを必須入力とする"""
        if values.data.get("order_type") == "逆指値" and value is None:
            raise ValueError("order_typeが'逆指値'の場合、stop_order_trigger_priceを設定してください。")
        return value

    @field_validator("stop_order_price")
    def validate_stop_order_price(cls, value, values):
        """order_typeが逆指値でstop_order_typeが指値のとき、stop_order_priceを必須入力とする"""
        if values.data.get("order_type") == "逆指値" and values.get("stop_order_type") == "指値" and value is None:
            raise ValueError("order_typeが'逆指値'でstop_order_typeが'指値'の場合、stop_order_priceを設定してください。")
        return value

    @field_validator("period_value", "period_index", mode="before")
    def validate_period_value_or_index(cls, value, values):
        """period_typeが期間指定のとき、period_valueまたはperiod_indexのいずれかを必ず入力する"""
        if values.data.get("period_type") == "期間指定" and not (values.get("period_value") or values.get("period_index")):
            raise ValueError("period_typeが'期間指定'の場合、period_valueまたはperiod_indexのいずれかを設定してください。")
        return value