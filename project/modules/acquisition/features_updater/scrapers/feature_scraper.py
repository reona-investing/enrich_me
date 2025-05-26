from utils.browser.browser_manager import BrowserManager
import pandas as pd
from typing import Literal

from acquisition.features_updater.scrapers._feature_historical_scraper import FeatureHistoricalScraper
from acquisition.features_updater.scrapers._feature_latest_value_scraper import FeatureLatestValueScraper

class FeatureScraper:
    """ヒストリカルデータと最新値データを統合するクラス"""
    
    def __init__(self, browser_manager: BrowserManager):
        self.historical_scraper = FeatureHistoricalScraper(browser_manager)
        self.latest_value_scraper = FeatureLatestValueScraper(browser_manager)

    async def scrape_feature(self, 
                             investing_code: str, 
                             additional_scrape: Literal['None', 'Baltic', 'Tradingview', 'ARCA'] = 'None',
                             additional_code: str = 'None') -> pd.DataFrame:
        """
        指定された銘柄の完全なデータ（ヒストリカル + 最新値）を取得
        
        Args:
            investing_code (str): investing.comの銘柄コード
            additional_scrape (Literal): 追加データソース
            additional_code (str): 追加データソースでの銘柄コード
            
        Returns:
            pd.DataFrame: 統合されたデータ ['Date', 'Open', 'Close', 'High', 'Low']
        """
        # ヒストリカルデータを取得
        df = await self.historical_scraper.scrape_historical_data(investing_code)
        
        # 追加の最新値データを取得
        df_to_add = pd.DataFrame()
        if additional_scrape != 'None':
            df_to_add = await self.latest_value_scraper.scrape_latest_value(
                source=additional_scrape, 
                code=additional_code
            )
        
        # データを統合
        if not df_to_add.empty:
            df = pd.concat([df, df_to_add], axis=0, ignore_index=True)
        
        return df.drop_duplicates(subset='Date', keep='last').reset_index(drop=True)


if __name__ == '__main__':
    async def main():
        bm = BrowserManager()
        fs = FeatureScraper(bm)
        df = await fs.scrape_feature(investing_code='commodities/iron-ore-62-cfr-futures', additional_scrape='Tradingview', additional_code='COMEX-TIO1!')
        print(df)
    
    import asyncio
    asyncio.get_event_loop().run_until_complete(main())