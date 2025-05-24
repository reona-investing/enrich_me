from .base_schema import BaseSchema, ColumnDefinition

class StockListSchema(BaseSchema):
    """銘柄リストのスキーマ定義"""
    
    code = ColumnDefinition(
        key="銘柄コード",
        raw_name="Code",
        processed_name="Code",
        dtype=str,
        description="銘柄コード（4桁）"
    )
    
    company_name = ColumnDefinition(
        key="会社名",
        raw_name="CompanyName",
        processed_name="CompanyName",
        dtype=str,
        description="会社名"
    )
    
    market_code_name = ColumnDefinition(
        key="市場区分名",
        raw_name="MarketCodeName",
        processed_name="MarketCodeName",
        dtype=str,
        description="市場区分名"
    )
    
    sector_17_code_name = ColumnDefinition(
        key="17業種区分名",
        raw_name="Sector17CodeName",
        processed_name="Sector17CodeName",
        dtype=str,
        description="17業種区分名"
    )
    
    sector_33_code_name = ColumnDefinition(
        key="33業種区分名",
        raw_name="Sector33CodeName",
        processed_name="Sector33CodeName",
        dtype=str,
        description="33業種区分名"
    )
    
    scale_category = ColumnDefinition(
        key="規模区分",
        raw_name="ScaleCategory",
        processed_name="ScaleCategory",
        dtype=str,
        description="規模区分"
    )
    
    listing = ColumnDefinition(
        key="上場フラグ",
        raw_name="Listing",
        processed_name="Listing",
        dtype=int,
        description="上場フラグ（1:上場中, 0:廃止）"
    )