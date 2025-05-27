from typing import Literal
from utils.browser.browser_manager import BrowserManager
from acquisition.features_updater.scrapers.feature_latest_value.base import BaseLatestValueScraper
from acquisition.features_updater.scrapers.feature_latest_value.sources import (
    BalticScraper, TradingviewScraper, ArcaScraper
)


class ScraperFactory:
    """各データソースのスクレイパーを生成するファクトリークラス"""
    
    @staticmethod
    def create_scraper(source: Literal['Baltic', 'Tradingview', 'ARCA'], 
                      browser_manager: BrowserManager) -> BaseLatestValueScraper:
        """
        指定されたデータソースのスクレイパーを生成
        
        Args:
            source (Literal): データソース ('Baltic', 'Tradingview', 'ARCA')
            browser_manager (BrowserManager): ブラウザマネージャー
            
        Returns:
            BaseLatestValueScraper: 対応するスクレイパーインスタンス
            
        Raises:
            ValueError: サポートされていないデータソースが指定された場合
        """
        if source == 'Baltic':
            return BalticScraper(browser_manager)
        elif source == 'Tradingview':
            return TradingviewScraper(browser_manager)
        elif source == 'ARCA':
            return ArcaScraper(browser_manager)
        else:
            raise ValueError(f"Unsupported source: {source}")