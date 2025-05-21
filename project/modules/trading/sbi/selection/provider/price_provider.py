import pandas as pd
from utils.jquants_api_utils import cli
from trading.sbi.interface.selection import IPriceProvider

class PriceProvider(IPriceProvider):
    """価格情報提供クラス"""
    
    def __init__(self, order_price_df: pd.DataFrame):
        """
        Args:
            order_price_df: 注文価格データフレーム
        """
        self.order_price_df = order_price_df
    
    def get_price_data(self) -> pd.DataFrame:
        """価格データを取得"""
        return self.order_price_df.copy()
    
    def get_etf_price(self, symbol_code: str) -> float:
        """指定したETFの価格を取得"""
        etf_price = cli.get_prices_daily_quotes(code=symbol_code)
        return etf_price['Close'].iat[-1]