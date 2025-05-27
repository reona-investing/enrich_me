import pandas as pd
from datetime import datetime
import pytz
from bs4 import BeautifulSoup as soup
from acquisition.features_updater.scrapers.feature_latest_value.base import BaseLatestValueScraper


class TradingviewScraper(BaseLatestValueScraper):
    """TradingViewから最新のコモディティ価格を取得するクラス"""
    
    async def scrape_latest_value(self, code: str) -> pd.DataFrame:
        """
        TradingViewから最新のコモディティ価格を取得
        
        Args:
            code (str): 取得する銘柄のコード（例: 'COMEX-TIO1!'）
            
        Returns:
            pd.DataFrame: 最新値データ ['Date', 'Open', 'Close', 'High', 'Low']
        """
        return await self._scrape_from_tradingview(
            name='Tradingview',
            code=code
        )
    
    async def _scrape_from_tradingview(self, 
                                       name: str,
                                       code: str) -> pd.DataFrame:
        """TradingViewから最新のコモディティ価格を取得"""
        # コモディティはシカゴ時間で8時に更新（=シカゴ時間8時以降のみデータ取得する）
        latest_day = value = None
        chicago_time = datetime.now().astimezone(pytz.utc).astimezone(pytz.timezone('America/Chicago'))
        
        if chicago_time.hour >= 8:
            latest_day = datetime.combine(chicago_time.date(), datetime.min.time())

            url = f'https://jp.tradingview.com/symbols/{code}/'
            named_tab = await self.browser_manager.new_tab(name=name, url=url)
            await named_tab.tab.utils.wait(10)
            
            html = await named_tab.tab.utils.get_html_content()
            s = soup(html, 'html.parser')
            text_list = s.select('div.js-symbol-header-ticker')[1].select('span')
            value = float([x.get_text() for x in text_list][0])
            
            await self.browser_manager.close_tab(name=name)

        if latest_day is None or value is None:
            return pd.DataFrame()
        return self._create_df_with_one_row(latest_day, value)