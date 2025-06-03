from nodriver import Element
import pandas as pd
from datetime import datetime
import pytz
from acquisition.features_updater.scrapers.feature_latest_value.base import BaseLatestValueScraper
from bs4 import BeautifulSoup


class BalticScraper(BaseLatestValueScraper):
    """バルチック海運取引所から最新のBDI値を取得するクラス"""
    
    async def scrape_latest_value(self, code: str = 'BDI') -> pd.DataFrame:
        """
        バルチック海運取引所から最新のBDI値を取得
        
        Args:
            code (str): 取得する銘柄のコード（デフォルト: 'BDI'）
            
        Returns:
            pd.DataFrame: 最新値データ ['Date', 'Open', 'Close', 'High', 'Low']
        """
        return await self._scrape_from_baltic_exchange(
            name='baltic_exchange', 
            url='https://www.balticexchange.com/en/index.html',
            ticker_code=code
        )
    
    async def _scrape_from_baltic_exchange(self, 
                                           name: str, 
                                           url: str,
                                           ticker_code: str) -> pd.DataFrame:
        """バルチック海運取引所から最新のBDI値を取得"""
        # 英国時間で13時に更新（=英国時間13時以降のみデータ取得する）
        latest_day = value = None
        UK_time = datetime.now().astimezone(pytz.utc).astimezone(pytz.timezone('Europe/London'))
        
        if UK_time.hour >= 13: 
            latest_day = datetime.combine(UK_time.date(), datetime.min.time())

            named_tab = await self.browser_manager.new_tab(name=name, url=url)
            ticker_rail_element = await named_tab.tab.utils.wait_for('div[class="ticker-rail"]', is_css=True)
            await named_tab.tab.utils.wait(5)
            ticker_rail_html = await ticker_rail_element.get_html()

            soup = BeautifulSoup(ticker_rail_html, 'html.parser')
            price_text = None
            for ticket in soup.find_all('div', class_='ticket'):
                index_span = ticket.find('span', class_='index')
                value_span = ticket.find('span', class_='value')
                if index_span and value_span and index_span.get_text(strip=True) == ticker_code:
                    price_text = value_span.get_text(strip=True)
                    break
            
            if price_text is None:
                raise ValueError(f'{ticker_code}の価格情報の要素が見つかりませんでした。')
            
            value = float(price_text.replace(',', ''))

            await self.browser_manager.close_tab(name=name)

        if latest_day is None or value is None:
            return pd.DataFrame()
        return self._create_df_with_one_row(latest_day, value)


if __name__ == '__main__':
    from utils.browser import BrowserManager
    import asyncio
    async def main():
        bs = BalticScraper(BrowserManager())
        a = await bs.scrape_latest_value()
        print(a)
    asyncio.get_event_loop().run_until_complete(main())
