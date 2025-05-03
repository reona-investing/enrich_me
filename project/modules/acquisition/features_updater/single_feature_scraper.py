from nodriver import Element
from utils.browser.browser_manager import BrowserManager
import asyncio
from bs4 import BeautifulSoup as soup
from datetime import datetime
import pandas as pd
import pytz
from typing import Tuple, Literal
from utils.timekeeper import timekeeper


class SingleFeatureScraper:
    def __init__(self, browser_manager:BrowserManager):
        self.browser_manager = browser_manager


    async def scrape_feature(self, 
                             investing_code: str, 
                             additional_scrape: Literal['None', 'Baltic', 'Tradingview', 'ARCA'] = 'None',
                             additional_code: str = 'None'):
        investing_url = 'https://jp.investing.com/' + investing_code + '-historical-data'
        df = await self._scrape_from_investing(name=investing_code,url=investing_url)
        df_to_add = pd.DataFrame()  # Initialize as empty DataFrame
        if additional_scrape == 'Baltic':
            df_to_add = await self._additional_scrape_from_baltic_exchange()
        if additional_scrape == 'Tradingview':
            df_to_add = await self._additional_scrape_from_tradingview(code=additional_code)
        if additional_scrape == 'ARCA':
            df_to_add = await self._additional_scrape_from_ARCA(name=additional_code, code=additional_code)
        if additional_scrape != 'None':
            df = pd.concat([df, df_to_add], axis=0, ignore_index=True)
        return df.drop_duplicates(subset = 'Date', keep = 'last').reset_index(drop = True)


    async def _scrape_from_investing(self, name: str, url: str) -> pd.DataFrame:
        '''
        investingからのスクレイピング
        
        Args:
            name (str): タブの名前
            url (str): タブで開くURL

        Returns:
            pd.DataFrame: スクレイピングで取得した価格情報df
        '''
        named_tab = None
        max_retry = 10
        for i in range(max_retry):
            try:
                if named_tab:
                    print('reloading...')
                    await named_tab.tab.utils.reload()
                else:
                    named_tab = await self.browser_manager.new_tab(name=name, url=url)
                    
                # テーブルの要素を取得し、BeautifulSoupで処理
                table_header_initial = await named_tab.tab.utils.wait_for('日付け')
                table_selector = table_header_initial
                if table_selector is None:
                    raise ValueError(f'Table header not found: {url}')
                while table_selector.tag_name != "table":
                    table_selector = table_selector.parent
                    if table_selector is None:
                        raise ValueError(f'Table element not found: {url}')
                html = await table_selector.get_html()
                souped = soup(html, "html.parser")
                await self.browser_manager.close_tab(name=name)

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
        
        for fmt in ['%Y年%m月%d日', '%m月 %d, %Y', '%b %d, %Y']:
            try:
                df['Date'] = pd.to_datetime(df['Date'], format=fmt)
                break
            except:
                continue

        return df[['Date', 'Open', 'Close', 'High', 'Low']].sort_values(by = 'Date', ascending = True)


    async def _additional_scrape_from_baltic_exchange(self, 
                                                      name:str = 'baltic_exchange', 
                                                      url:str = 'https://www.balticexchange.com/en/index.html') -> pd.DataFrame:
        # 英国時間で13時に更新（=英国時間13時以降のみデータ取得する）
        latest_day = value = None
        UK_time = datetime.now().astimezone(pytz.utc).astimezone(pytz.timezone('Europe/London')) #現在のイギリス時間を取得
        if UK_time.hour >= 13: 
            latest_day = datetime.combine(UK_time.date(), datetime.min.time())  # Convert date to datetime

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
        

    def _create_df_with_one_row(self, day: datetime, value: float):
        if day is None:
            return pd.DataFrame()
        return pd.DataFrame({'Date':day, 'Open':value, 'Close':value, 'High':value, 'Low':value}, index=[0])


    async def _additional_scrape_from_tradingview(self, 
                                                  name: str = 'Tradingview',
                                                  code: str = 'COMEX-TIO1!') -> pd.DataFrame:
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


    async def _additional_scrape_from_ARCA(self, 
                                           name: str,
                                           code: str) -> pd.DataFrame:
        '''ARCAからのスクレイピング'''
        url = f'https://www.nyse.com/quote/index/{code}'
        named_tab = await self.browser_manager.new_tab(name=name, url=url)
        html = await named_tab.tab.utils.get_html_content()
        await self.browser_manager.close_tab(name=name)

        s = soup(html, 'html.parser')
        #価格
        value = s.find_all('span', class_='d-dquote-x3')[0].text
        #日時
        s_time = s.select('div.d-dquote-time')[0]
        time_elem = s_time.select('span')[1].get_text()
        latest_day = time_elem[1:11]
        latest_day = datetime.strptime(latest_day, '%m/%d/%Y')

        if latest_day is None or value is None:
            return pd.DataFrame()
        return self._create_df_with_one_row(latest_day, value)


if __name__ == '__main__':
    @timekeeper
    async def main():
        import numpy as np
        browser_manager = BrowserManager()
        single_feature_scraper = SingleFeatureScraper(browser_manager)
        #df1 = await single_feature_scraper.scrape_feature(investing_code = 'currencies/usd-jpy', additional_scrape = np.nan, additional_code = np.nan)
        #df2 = await single_feature_scraper.scrape_feature(investing_code = 'currencies/eur-jpy', additional_scrape = np.nan, additional_code = np.nan)
        #df3 = await single_feature_scraper.scrape_feature(investing_code = 'currencies/aud-jpy', additional_scrape = np.nan, additional_code = np.nan)
        #dfs = [df1, df2, df3]
        df = await single_feature_scraper._additional_scrape_from_baltic_exchange()
        print(df)
        dfs = await single_feature_scraper.scrape_feature(investing_code = 'commodities/iron-ore-62-cfr-futures', 
                                                          additional_scrape = 'Tradingview', additional_code = 'COMEX-TIO1!')
        return dfs
    
    @timekeeper
    async def main_async():
        import numpy as np
        browser_manager = BrowserManager()
        scraper = SingleFeatureScraper(browser_manager)
        tasks = []
        investing_codes = ['currencies/usd-jpy', 'currencies/eur-jpy', 'currencies/aud-jpy']
        for investing_code in investing_codes:
            tasks.append(scraper.scrape_feature(investing_code = investing_code, additional_scrape = 'None', additional_code = 'None'))

        dfs = await asyncio.gather(*tasks)
        return dfs

    dfs = asyncio.get_event_loop().run_until_complete(main())
    #dfs = asyncio.get_event_loop().run_until_complete(main_async())
    print(dfs)