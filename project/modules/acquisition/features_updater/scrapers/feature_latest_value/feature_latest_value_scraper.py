from utils.browser.browser_manager import BrowserManager
import pandas as pd
from typing import Literal
from acquisition.features_updater.scrapers.feature_latest_value.factory import ScraperFactory


class FeatureLatestValueScraper:
    """各種データソースから最新値を取得するクラス（リファクタリング後）"""
    
    def __init__(self, browser_manager: BrowserManager):
        self.browser_manager = browser_manager

    async def scrape_latest_value(self, 
                                  source: Literal['Baltic', 'Tradingview', 'ARCA'], 
                                  code: str) -> pd.DataFrame:
        """
        指定されたデータソースから最新値を取得
        
        Args:
            source (Literal): データソース ('Baltic', 'Tradingview', 'ARCA')
            code (str): 取得する銘柄のコード
            
        Returns:
            pd.DataFrame: 最新値データ ['Date', 'Open', 'Close', 'High', 'Low']
        """
        scraper = ScraperFactory.create_scraper(source, self.browser_manager)
        return await scraper.scrape_latest_value(code)