import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup as soup
from acquisition.features_updater.scrapers.feature_latest_value.base import BaseLatestValueScraper


class ArcaScraper(BaseLatestValueScraper):
    """ARCAから最新の価格情報を取得するクラス"""
    
    async def scrape_latest_value(self, code: str) -> pd.DataFrame:
        """
        ARCAから最新の価格情報を取得
        
        Args:
            code (str): 取得する銘柄のコード
            
        Returns:
            pd.DataFrame: 最新値データ ['Date', 'Open', 'Close', 'High', 'Low']
        """
        return await self._scrape_from_arca(name=code, code=code)
    
    async def _scrape_from_arca(self, name: str, code: str) -> pd.DataFrame:
        """ARCAから最新の価格情報を取得"""
        url = f'https://www.nyse.com/quote/index/{code}'
        named_tab = await self.browser_manager.new_tab(name=name, url=url)
        html = await named_tab.tab.utils.get_html_content()
        await self.browser_manager.close_tab(name=name)

        s = soup(html, 'html.parser')
        
        # 価格
        value = s.find_all('span', class_='d-dquote-x3')[0].text
        
        # 日時
        s_time = s.select('div.d-dquote-time')[0]
        time_elem = s_time.select('span')[1].get_text()
        latest_day = time_elem[1:11]
        latest_day = datetime.strptime(latest_day, '%m/%d/%Y')

        if latest_day is None or value is None:
            return pd.DataFrame()
        return self._create_df_with_one_row(latest_day, value)