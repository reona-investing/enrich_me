from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime
from utils.browser.browser_manager import BrowserManager


class BaseLatestValueScraper(ABC):
    """最新値スクレイパーの抽象基底クラス"""
    
    def __init__(self, browser_manager: BrowserManager):
        self.browser_manager = browser_manager
    
    @abstractmethod
    async def scrape_latest_value(self, code: str) -> pd.DataFrame:
        """
        最新値を取得する抽象メソッド
        
        Args:
            code (str): 取得する銘柄のコード
            
        Returns:
            pd.DataFrame: 最新値データ ['Date', 'Open', 'Close', 'High', 'Low']
        """
        pass
    
    def _create_df_with_one_row(self, day: datetime, value: float) -> pd.DataFrame:
        """単一行のDataFrameを作成（OHLC全て同じ値）"""
        if day is None:
            return pd.DataFrame()
        return pd.DataFrame({
            'Date': day, 
            'Open': value, 
            'Close': value, 
            'High': value, 
            'Low': value
        }, index=[0])