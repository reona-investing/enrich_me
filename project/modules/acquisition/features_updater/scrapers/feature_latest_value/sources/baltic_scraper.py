from nodriver import Element
import pandas as pd
from datetime import datetime
import pytz
from acquisition.features_updater.scrapers.feature_latest_value.base import BaseLatestValueScraper


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
            await named_tab.tab.utils.wait(3)
            ticker_name_element = await named_tab.tab.utils.wait_for(ticker_code)
            
            if ticker_name_element is None or ticker_name_element.parent is None:
                raise ValueError(f'{ticker_code}の価格情報の要素が見つかりませんでした。')
            
            element = ticker_name_element.parent.children[1]
            if type(element) == Element:
                value = float(element.text.replace(',', ''))
            elif type(element) == str:
                value = float(element.replace(',', ''))
            else:
                raise ValueError(f'{ticker_code}の価格情報を正しく取得できませんでした。')

            await self.browser_manager.close_tab(name=name)

        if latest_day is None or value is None:
            return pd.DataFrame()
        return self._create_df_with_one_row(latest_day, value)