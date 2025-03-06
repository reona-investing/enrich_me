import pandas as pd
from utils.browser import BrowserUtils
import asyncio

# 1. リンクを取得するクラス (モジュール内部でのみ使用)
class _LoanMarginsFetcher:
    def __init__(self, browser_utils: BrowserUtils):
        self.browser_utils = browser_utils

    async def fetch_url(self) -> str:
        HOME_PAGE = 'https://www.jpx.co.jp'
        LINKED_PAGE = f'{HOME_PAGE}/listing/others/margin/index.html'
        
        await self.browser_utils.open_url(LINKED_PAGE)
        file_block = await self.browser_utils.select_element('div[class*=component-file]', is_css=True)
        file_element = await file_block.query_selector('a[href]')
        
        return f'{HOME_PAGE}{file_element.attrs["href"]}'

# 2. データフレームを処理するクラス (モジュール内部でのみ使用)
class _LoanMarginsProcessor:
    @staticmethod
    def process_data(file_url: str) -> pd.DataFrame:
        return pd.read_excel(file_url, header=1)

# 3. 外部に公開するクラス (インターフェース)
class LoanMarginsListGetter:
    def __init__(self, browser_utils: BrowserUtils):
        self._fetcher = _LoanMarginsFetcher(browser_utils)
        self._processor = _LoanMarginsProcessor()
    
    async def get(self) -> pd.DataFrame:
        file_url = await self._fetcher.fetch_url()
        return self._processor.process_data(file_url)

# 4. 実行コード
if __name__ == '__main__':
    async def main():
        browser_utils = BrowserUtils()
        getter = LoanMarginsListGetter(browser_utils)
        return await getter.get()

    loan_margins_list = asyncio.get_event_loop().run_until_complete(main())
    print(loan_margins_list)