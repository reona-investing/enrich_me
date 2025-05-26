from nodriver import Element
from utils.browser.browser_manager import BrowserManager
from bs4 import BeautifulSoup as soup
from datetime import datetime
import pandas as pd
import pytz
from typing import Literal

class FeatureLatestValueScraper:
    """各種データソースから最新値を取得するクラス"""
    
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
        if source == 'Baltic':
            return await self._scrape_from_baltic_exchange()
        elif source == 'Tradingview':
            return await self._scrape_from_tradingview(code=code)
        elif source == 'ARCA':
            return await self._scrape_from_ARCA(name=code, code=code)
        else:
            raise ValueError(f"Unsupported source: {source}")

    async def _scrape_from_baltic_exchange(self, 
                                           name: str = 'baltic_exchange', 
                                           url: str = 'https://www.balticexchange.com/en/index.html') -> pd.DataFrame:
        """バルチック海運取引所から最新のBDI値を取得"""
        # 英国時間で13時に更新（=英国時間13時以降のみデータ取得する）
        latest_day = value = None
        UK_time = datetime.now().astimezone(pytz.utc).astimezone(pytz.timezone('Europe/London'))
        if UK_time.hour >= 13: 
            latest_day = datetime.combine(UK_time.date(), datetime.min.time())

            named_tab = await self.browser_manager.new_tab(name=name, url=url)
            await named_tab.tab.utils.wait(3)
            ticker_name_element = await named_tab.tab.utils.wait_for('BDI')
            if ticker_name_element is None or ticker_name_element.parent is None:
                raise ValueError('Baltic Dryの価格情報の要素が見つかりませんでした。')
            element = ticker_name_element.parent.children[1]
            if type(element) == Element:
                value = float(element.text.replace(',', ''))
            elif type(element) == str:
                value = float(element.replace(',', ''))
            else:
                raise ValueError('Baltic Dryの価格情報を正しく取得できませんでした。')

            await self.browser_manager.close_tab(name=name)

        if latest_day is None or value is None:
            return pd.DataFrame()
        return self._create_df_with_one_row(latest_day, value)

    async def _scrape_from_tradingview(self, 
                                       name: str = 'Tradingview',
                                       code: str = 'COMEX-TIO1!') -> pd.DataFrame:
        """TradingViewから最新のコモディティ価格を取得"""
        # コモディティはシカゴ時間で8時に更新（=シカゴ時間8時以降のみデータ取得する）
        latest_day = value = None
        chicago_time = datetime.now().astimezone(pytz.utc).astimezone(pytz.timezone('America/Chicago'))
        if chicago_time.hour >= 8:
            latest_day = datetime.combine(chicago_time.date(), datetime.min.time())

            url = f'https://jp.tradingview.com/symbols/{code}/'
            named_tab = await self.browser_manager.new_tab(name=name, url=url)
            #await named_tab.tab.utils.wait_for('USD / TNE')
            #await named_tab.tab.utils.wait(1)
            await named_tab.tab.utils.wait(10)
            html = await named_tab.tab.utils.get_html_content()
            s = soup(html, 'html.parser')
            text_list = s.select('div.js-symbol-header-ticker')[1].select('span')
            value = float([x.get_text() for x in text_list][0])
            await self.browser_manager.close_tab(name=name)

        if latest_day is None or value is None:
            return pd.DataFrame()
        return self._create_df_with_one_row(latest_day, value)

    async def _scrape_from_ARCA(self, name: str, code: str) -> pd.DataFrame:
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