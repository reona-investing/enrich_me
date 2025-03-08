from utils.paths import Paths
from utils.browser import BrowserUtils
import asyncio
from bs4 import BeautifulSoup as soup
from datetime import datetime, timedelta
import pandas as pd
import pytz
from typing import Tuple, Literal
import time

from utils.timekeeper import timekeeper


class SingleFeatureScraper:
    def __init__(self, browser_utils:BrowserUtils):
        self.browser_utils = browser_utils

    async def scrape_feature(self, 
                             investing_code: str, 
                             additional_scrape: Literal['None', 'Baltic', 'Tradingview', 'ARCA'] = 'None',
                             additional_code: str = 'None'):
        investing_url = 'https://jp.investing.com/' + investing_code + '-historical-data'
        df = await self._scrape_from_investing(investing_url)
        if additional_scrape == 'Baltic':
            df_to_add = await self._additional_scrape_from_baltic_exchange()
        if additional_scrape == 'Tradingview':
            df_to_add = await self._additional_scrape_from_tradingview(additional_code)
        if additional_scrape == 'ARCA':
            df_to_add = await self._additional_scrape_from_ARCA(additional_code)
        if pd.notna(additional_scrape):
            df = pd.concat([df, df_to_add], axis = 0, ignore_index=True)
        return df.drop_duplicates(subset = 'Date', keep = 'last').reset_index(drop = True)



    async def _scrape_from_investing(self, url:str) -> pd.DataFrame:
        '''investingからのスクレイピング'''
        max_retry = 10
        for i in range(max_retry):
            try:
                if i == 0:
                    await self.browser_utils.open_url(url)
                else:
                    print('reloading...')
                    await self.browser_utils.reload()
                    
                # テーブルの要素を取得し、BeautifulSoupで処理
                table_header_initial = await self.browser_utils.wait_for('日付け')
                table_selector = table_header_initial
                while table_selector.tag_name != "table":
                    table_selector = table_selector.parent
                html = await table_selector.get_html()
                souped = soup(html, "html.parser")

                # ヘッダーの取得
                headers = [th.text.strip() for th in souped.select("thead th")]
                # データの取得
                rows = []
                for tr in souped.select("tbody tr"):
                    cells = [td.text.strip() for td in tr.find_all("td")]
                    rows.append(cells)         

                # DataFrame に変換
                df_to_add = pd.DataFrame(rows, columns=headers)
                break
            except:
                continue
        else:
            raise ValueError(f'DataFrame is not found: {url}')
        
        return self._format_investing_df(df_to_add)


    def _format_investing_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.iloc[:, :5].copy()
        df.columns = ['Date', 'Close', 'Open', 'High', 'Low']
        
        try:
            df['Date'] = pd.to_datetime(df['Date'], format='%Y年%m月%d日')
        except:
            df['Date'] = pd.to_datetime(df['Date'], format='%m月 %d, %Y')  # datetime型に変換

        return df[['Date', 'Open', 'Close', 'High', 'Low']].sort_values(by = 'Date', ascending = True)


    async def _additional_scrape_from_baltic_exchange(self, url:str = 'https://www.balticexchange.com/en/index.html') -> Tuple[str, datetime]:
        # 英国時間で13時に更新（=英国時間13時以降のみデータ取得する）
        latest_day = value = None
        UK_time = datetime.now().astimezone(pytz.utc).astimezone(pytz.timezone('Europe/London')) #現在のイギリス時間を取得
        if UK_time.hour >= 13: 
            latest_day = UK_time.date() #当日の日付

            await self.browser_utils.open_url(url)
            await self.browser_utils.wait(3)
            element = await self.browser_utils.wait_for('#ticker > div > div > div:nth-child(1) > span.value', is_css=True)
            value = float(element.text.replace(',', ''))

        return self._create_df_with_one_row(latest_day, value)
        

    def _create_df_with_one_row(self, day: datetime, value: float):
        if day is None:
            return pd.DataFrame()
        return pd.DataFrame({'Date':day, 'Open':value, 'Close':value, 'High':value, 'Low':value}, index=[0])


    async def _additional_scrape_from_tradingview(self, code: str = 'COMEX-TIO1!') -> Tuple[str, datetime]:
        # コモディティはシカゴ時間で8時に更新（=シカゴ時間8時以降のみデータ取得する）
        latest_day = value = None
        chicago_time = datetime.now().astimezone(pytz.utc).astimezone(pytz.timezone('America/Chicago'))
        if chicago_time.hour >= 8:
            latest_day = chicago_time.date() #当日の日付

            url = f'https://www.tradingview.com/symbols/{code}/'
            await self.browser_utils.open_url(url)
            await self.browser_utils.wait(10)
            element = await self.browser_utils.wait_for(
                '#js-category-content > div.tv-react-category-header > div.js-symbol-page-header-root > div > div > div > div.quotesRow-pAUXADuj > div:nth-child(1) > div > div.lastContainer-JWoJqCpY > span.last-JWoJqCpY.js-symbol-last > span',
                is_css = True)
            value = float(element.text)

        return self._create_df_with_one_row(latest_day, value)


    async def _additional_scrape_from_ARCA(self, code:str) -> Tuple[str, datetime]:
        '''ARCAからのスクレイピング'''
        url = f'https://www.nyse.com/quote/index/{code}'
        await self.browser_utils.open_url(url)
        await self.browser_utils.wait(10)
        html = await self.browser_utils.get_content()

        s = soup(html, 'html.parser')
        #価格
        value = s.find_all('span', class_='d-dquote-x3')[0].text
        #日時
        s_time = s.find_all('div', class_='d-dquote-time')[0]
        time_elem = s_time.find_all('span')[1].text
        latest_day = time_elem[1:11]
        latest_day = datetime.strptime(latest_day, '%m/%d/%Y')

        return self._create_df_with_one_row(latest_day, value)


if __name__ == '__main__':
    browser_utils = BrowserUtils()
    single_feature_scraper = SingleFeatureScraper(browser_utils)
    df = asyncio.get_event_loop().run_until_complete(
        single_feature_scraper.scrape_feature(feature_name = 'USDJPY', 
                                              path = 'raw_USDJPY_price.parquet', 
                                              investing_code = 'currencies/usd-jpy')
        )
    df.to_csv('mock.csv', index=False)
    print(df)
        