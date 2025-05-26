from utils.browser.browser_manager import BrowserManager
from datetime import datetime
import pandas as pd
from typing import Literal
from acquisition.features_updater.scrapers import FeatureScraper

class SingleFeatureScraper:
    """FeatureScraperを使用した実装（既存インターフェース維持）"""
    
    def __init__(self, browser_manager: BrowserManager):
        self.feature_scraper = FeatureScraper(browser_manager)
        self.browser_manager = browser_manager  # 既存コードとの互換性のため

    async def scrape_feature(self, 
                             investing_code: str, 
                             additional_scrape: Literal['None', 'Baltic', 'Tradingview', 'ARCA'] = 'None',
                             additional_code: str = 'None') -> pd.DataFrame:
        """
        既存のインターフェースを維持しつつ、内部実装はFeatureScraperに委譲
        
        Args:
            investing_code (str): investing.comの銘柄コード
            additional_scrape (Literal): 追加データソース
            additional_code (str): 追加データソースでの銘柄コード
            
        Returns:
            pd.DataFrame: 統合されたデータ ['Date', 'Open', 'Close', 'High', 'Low']
        """
        return await self.feature_scraper.scrape_feature(
            investing_code=investing_code,
            additional_scrape=additional_scrape,
            additional_code=additional_code
        )

    # 既存コードで使用されている可能性のあるメソッドを維持
    async def _scrape_from_investing(self, name: str, url: str) -> pd.DataFrame:
        """既存コードとの互換性のため、FeatureHistoricalScraperに委譲"""
        return await self.feature_scraper.historical_scraper._scrape_from_investing(name, url)
    
    def _format_investing_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """既存コードとの互換性のため、FeatureHistoricalScraperに委譲"""
        return self.feature_scraper.historical_scraper._format_investing_df(df)
    
    async def _additional_scrape_from_baltic_exchange(self, 
                                                      name: str = 'baltic_exchange', 
                                                      url: str = 'https://www.balticexchange.com/en/index.html') -> pd.DataFrame:
        """既存コードとの互換性のため、FeatureLatestValueScraperに委譲"""
        return await self.feature_scraper.latest_value_scraper._scrape_from_baltic_exchange(name, url)
    
    async def _additional_scrape_from_tradingview(self, 
                                                  name: str = 'Tradingview',
                                                  code: str = 'COMEX-TIO1!') -> pd.DataFrame:
        """既存コードとの互換性のため、FeatureLatestValueScraperに委譲"""
        return await self.feature_scraper.latest_value_scraper._scrape_from_tradingview(name, code)
    
    async def _additional_scrape_from_ARCA(self, name: str, code: str) -> pd.DataFrame:
        """既存コードとの互換性のため、FeatureLatestValueScraperに委譲"""
        return await self.feature_scraper.latest_value_scraper._scrape_from_ARCA(name, code)
    
    def _create_df_with_one_row(self, day: datetime, value: float) -> pd.DataFrame:
        """既存コードとの互換性のため、FeatureLatestValueScraperに委譲"""
        return self.feature_scraper.latest_value_scraper._create_df_with_one_row(day, value)