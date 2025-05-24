from .base_schema import BaseSchema, ColumnDefinition
from datetime import datetime

class RawStockFinSchema(BaseSchema):
    """生の財務データのスキーマ定義"""
    
    disclosed_date = ColumnDefinition(
        key="日付",
        raw_name="日付",
        processed_name="DisclosedDate",
        dtype=str,
        description="開示日"
    )
    
    disclosed_time = ColumnDefinition(
        key="時刻",
        raw_name="時刻",
        processed_name="DisclosedTime",
        dtype=str,
        description="開示時刻"
    )
    
    local_code = ColumnDefinition(
        key="銘柄コード",
        raw_name="銘柄コード",
        processed_name="LocalCode",
        dtype=str,
        description="銘柄コード"
    )
    
    disclosure_number = ColumnDefinition(
        key="開示番号",
        raw_name="開示番号",
        processed_name="DisclosureNumber",
        dtype=str,
        description="開示番号"
    )
    
    type_of_document = ColumnDefinition(
        key="開示書類種別",
        raw_name="開示書類種別",
        processed_name="TypeOfDocument",
        dtype=str,
        description="開示書類種別"
    )
    
    type_of_current_period = ColumnDefinition(
        key="当会計期間の種類",
        raw_name="当会計期間の種類",
        processed_name="TypeOfCurrentPeriod",
        dtype=str,
        description="会計期間種類"
    )
    
    current_period_end_date = ColumnDefinition(
        key="当会計期間終了日",
        raw_name="当会計期間終了日",
        processed_name="CurrentPeriodEndDate",
        dtype=str,
        description="会計期間終了日"
    )
    
    current_fiscal_year_start_date = ColumnDefinition(
        key="当事業年度開始日",
        raw_name="当事業年度開始日",
        processed_name="CurrentFiscalYearStartDate",
        dtype=str,
        description="事業年度開始日"
    )
    
    current_fiscal_year_end_date = ColumnDefinition(
        key="当事業年度終了日",
        raw_name="当事業年度終了日",
        processed_name="CurrentFiscalYearEndDate",
        dtype=str,
        description="事業年度終了日"
    )
    
    net_sales = ColumnDefinition(
        key="売上高",
        raw_name="売上高",
        processed_name="NetSales",
        dtype=str,
        description="売上高"
    )
    
    operating_profit = ColumnDefinition(
        key="営業利益",
        raw_name="営業利益",
        processed_name="OperatingProfit",
        dtype=str,
        description="営業利益"
    )
    
    ordinary_profit = ColumnDefinition(
        key="経常利益",
        raw_name="経常利益",
        processed_name="OrdinaryProfit",
        dtype=str,
        description="経常利益"
    )
    
    profit = ColumnDefinition(
        key="当期純利益",
        raw_name="当期純利益",
        processed_name="Profit",
        dtype=str,
        description="当期純利益"
    )
    
    earnings_per_share = ColumnDefinition(
        key="EPS",
        raw_name="実績EPS",
        processed_name="EarningsPerShare",
        dtype=str,
        description="実績EPS"
    )
    
    total_assets = ColumnDefinition(
        key="総資産",
        raw_name="総資産",
        processed_name="TotalAssets",
        dtype=str,
        description="総資産"
    )
    
    equity = ColumnDefinition(
        key="純資産",
        raw_name="純資産",
        processed_name="Equity",
        dtype=str,
        description="純資産"
    )
    
    forecast_eps_current = ColumnDefinition(
        key="EPS_予想_期末",
        raw_name="EPS_予想_期末",
        processed_name="ForecastEarningsPerShare",
        dtype=str,
        required=False,
        description="予想EPS（期末）"
    )
    
    forecast_eps_next_year = ColumnDefinition(
        key="EPS_予想_翌事業年度期末",
        raw_name="EPS_予想_翌事業年度期末",
        processed_name="NextYearForecastEarningsPerShare",
        dtype=str,
        required=False,
        description="予想EPS（翌年）"
    )
    
    outstanding_shares_issued = ColumnDefinition(
        key="期末発行済株式数",
        raw_name="期末発行済株式数",
        processed_name="NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock",
        dtype=str,
        description="期末発行済株式数"
    )
    
    treasury_stock = ColumnDefinition(
        key="期末自己株式数",
        raw_name="期末自己株式数",
        processed_name="NumberOfTreasuryStockAtTheEndOfFiscalYear",
        dtype=str,
        required=False,
        description="期末自己株式数"
    )

class StockFinSchema(BaseSchema):
    """処理済み財務データのスキーマ定義"""
    
    date = ColumnDefinition(
        key="日付",
        raw_name="DisclosedDate",
        processed_name="Date",
        dtype=datetime,
        description="開示日"
    )
    
    code = ColumnDefinition(
        key="銘柄コード",
        raw_name="LocalCode",
        processed_name="Code",
        dtype=str,
        description="銘柄コード（4桁）"
    )
    
    type_of_document = ColumnDefinition(
        key="開示書類種別",
        raw_name="TypeOfDocument",
        processed_name="TypeOfDocument",
        dtype=str,
        description="開示書類種別"
    )
    
    type_of_current_period = ColumnDefinition(
        key="当会計期間の種類",
        raw_name="TypeOfCurrentPeriod",
        processed_name="TypeOfCurrentPeriod",
        dtype=str,
        description="会計期間種類（1Q, 2Q, 3Q, 4Q, FY）"
    )
    
    current_period_end_date = ColumnDefinition(
        key="当会計期間終了日",
        raw_name="CurrentPeriodEndDate",
        processed_name="CurrentPeriodEndDate",
        dtype=datetime,
        description="会計期間終了日"
    )
    
    current_fiscal_year_start_date = ColumnDefinition(
        key="当事業年度開始日",
        raw_name="CurrentFiscalYearStartDate",
        processed_name="CurrentFiscalYearStartDate",
        dtype=datetime,
        description="事業年度開始日"
    )
    
    current_fiscal_year_end_date = ColumnDefinition(
        key="当事業年度終了日",
        raw_name="CurrentFiscalYearEndDate",
        processed_name="CurrentFiscalYearEndDate",
        dtype=datetime,
        description="事業年度終了日"
    )
    
    net_sales = ColumnDefinition(
        key="売上高",
        raw_name="NetSales",
        processed_name="NetSales",
        dtype=float,
        plus_for_merger=True,
        description="売上高（百万円）"
    )
    
    operating_profit = ColumnDefinition(
        key="営業利益",
        raw_name="OperatingProfit",
        processed_name="OperatingProfit",
        dtype=float,
        plus_for_merger=True,
        description="営業利益（百万円）"
    )
    
    ordinary_profit = ColumnDefinition(
        key="経常利益",
        raw_name="OrdinaryProfit",
        processed_name="OrdinaryProfit",
        dtype=float,
        plus_for_merger=True,
        description="経常利益（百万円）"
    )
    
    profit = ColumnDefinition(
        key="当期純利益",
        raw_name="Profit",
        processed_name="Profit",
        dtype=float,
        plus_for_merger=True,
        description="当期純利益（百万円）"
    )
    
    earnings_per_share = ColumnDefinition(
        key="EPS",
        raw_name="EarningsPerShare",
        processed_name="EarningsPerShare",
        dtype=float,
        description="実績EPS（円）"
    )
    
    total_assets = ColumnDefinition(
        key="総資産",
        raw_name="TotalAssets",
        processed_name="TotalAssets",
        dtype=float,
        plus_for_merger=True,
        description="総資産（百万円）"
    )
    
    equity = ColumnDefinition(
        key="純資産",
        raw_name="Equity",
        processed_name="Equity",
        dtype=float,
        plus_for_merger=True,
        description="純資産（百万円）"
    )
    
    forecast_eps_current = ColumnDefinition(
        key="EPS_予想_期末",
        raw_name="ForecastEarningsPerShare",
        processed_name="ForecastEarningsPerShare",
        dtype=float,
        required=False,
        description="予想EPS（期末）"
    )
    
    forecast_eps_next_year = ColumnDefinition(
        key="EPS_予想_翌事業年度期末",
        raw_name="NextYearForecastEarningsPerShare",
        processed_name="NextYearForecastEarningsPerShare",
        dtype=float,
        required=False,
        description="予想EPS（翌年）"
    )
    
    outstanding_shares_issued = ColumnDefinition(
        key="期末発行済株式数",
        raw_name="NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock",
        processed_name="NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock",
        dtype=float,
        plus_for_merger=True,
        description="期末発行済株式数"
    )
    
    treasury_stock = ColumnDefinition(
        key="期末自己株式数",
        raw_name="NumberOfTreasuryStockAtTheEndOfFiscalYear",
        processed_name="NumberOfTreasuryStockAtTheEndOfFiscalYear",
        dtype=float,
        plus_for_merger=True,
        required=False,
        description="期末自己株式数"
    )
    
    # 追加計算カラム（生データには存在しない）
    forecast_eps = ColumnDefinition(
        key="予想EPS",
        raw_name="ForecastEPS",  # 計算で生成
        processed_name="ForecastEPS",
        dtype=float,
        required=False,
        description="予想EPS（円）"
    )
    
    outstanding_shares = ColumnDefinition(
        key="発行済み株式数",
        raw_name="OutstandingShares",  # 計算で生成
        processed_name="OutstandingShares",
        dtype=float,
        required=False,
        description="発行済み株式数（株）"
    )
    
    current_fiscal_year = ColumnDefinition(
        key="年度",
        raw_name="CurrentFiscalYear",  # 計算で生成
        processed_name="CurrentFiscalYear",
        dtype=int,
        required=False,
        description="決算年度"
    )
    
    forecast_fiscal_year_end_date = ColumnDefinition(
        key="予測対象の年度の終了日",
        raw_name="ForecastFiscalYearEndDate",  # 計算で生成
        processed_name="ForecastFiscalYearEndDate",
        dtype=str,
        required=False,
        description="予測対象年度終了日（YYYY/MM形式）"
    )
