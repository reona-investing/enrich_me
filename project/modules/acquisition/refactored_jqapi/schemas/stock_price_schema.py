from .base_schema import BaseSchema, ColumnDefinition
from datetime import datetime

class RawStockPriceSchema(BaseSchema):
    """生の株価データのスキーマ定義"""
    
    date = ColumnDefinition(
        key="日付",
        raw_name="日付",
        processed_name="Date",
        dtype=str,
        description="取引日"
    )
    
    code = ColumnDefinition(
        key="銘柄コード",
        raw_name="銘柄コード",
        processed_name="Code",
        dtype=str,
        description="銘柄コード"
    )
    
    open = ColumnDefinition(
        key="始値",
        raw_name="始値",
        processed_name="Open",
        dtype=float,
        description="始値"
    )
    
    high = ColumnDefinition(
        key="高値",
        raw_name="高値",
        processed_name="High",
        dtype=float,
        description="高値"
    )
    
    low = ColumnDefinition(
        key="安値",
        raw_name="安値",
        processed_name="Low",
        dtype=float,
        description="安値"
    )
    
    close = ColumnDefinition(
        key="終値",
        raw_name="終値",
        processed_name="Close",
        dtype=float,
        description="終値"
    )
    
    volume = ColumnDefinition(
        key="取引高",
        raw_name="取引高",
        processed_name="Volume",
        dtype=float,
        description="出来高"
    )
    
    turnover_value = ColumnDefinition(
        key="取引代金",
        raw_name="取引代金",
        processed_name="TurnoverValue",
        dtype=float,
        description="売買代金"
    )
    
    adjustment_factor = ColumnDefinition(
        key="調整係数",
        raw_name="調整係数",
        processed_name="AdjustmentFactor",
        dtype=float,
        description="調整係数"
    )

class StockPriceSchema(BaseSchema):
    """処理済み株価データのスキーマ定義"""
    
    date = ColumnDefinition(
        key="日付",
        raw_name="Date",
        processed_name="Date",
        dtype=datetime,
        description="取引日"
    )
    
    code = ColumnDefinition(
        key="銘柄コード",
        raw_name="Code",
        processed_name="Code",
        dtype=str,
        description="銘柄コード（4桁）"
    )
    
    open = ColumnDefinition(
        key="始値",
        raw_name="Open",
        processed_name="Open",
        dtype=float,
        description="始値（調整後）"
    )
    
    high = ColumnDefinition(
        key="高値",
        raw_name="High",
        processed_name="High",
        dtype=float,
        description="高値（調整後）"
    )
    
    low = ColumnDefinition(
        key="安値",
        raw_name="Low",
        processed_name="Low",
        dtype=float,
        description="安値（調整後）"
    )
    
    close = ColumnDefinition(
        key="終値",
        raw_name="Close",
        processed_name="Close",
        dtype=float,
        description="終値（調整後）"
    )
    
    volume = ColumnDefinition(
        key="取引高",
        raw_name="Volume",
        processed_name="Volume",
        dtype=float,
        description="出来高（調整後）"
    )
    
    turnover_value = ColumnDefinition(
        key="取引代金",
        raw_name="TurnoverValue",
        processed_name="TurnoverValue",
        dtype=float,
        description="売買代金"
    )
    
    adjustment_factor = ColumnDefinition(
        key="調整係数",
        raw_name="AdjustmentFactor",
        processed_name="AdjustmentFactor",
        dtype=float,
        description="調整係数"
    )
    
    # 追加計算カラム
    cumulative_adjustment_factor = ColumnDefinition(
        key="累積調整係数",
        raw_name="CumulativeAdjustmentFactor",  # 計算で生成
        processed_name="CumulativeAdjustmentFactor",
        dtype=float,
        required=False,
        description="累積調整係数"
    )