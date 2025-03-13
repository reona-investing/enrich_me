import pandas as pd
from utils.browser.browser_manager import BrowserManager
import asyncio

# 1. リンクを取得するクラス (モジュール内部でのみ使用)
class _LoanMarginsFetcher:
    def __init__(self, browser_manager: BrowserManager):
        self.browser_manager = browser_manager

    async def fetch_url(self) -> str:
        HOME_PAGE = 'https://www.jpx.co.jp'
        LINKED_PAGE = f'{HOME_PAGE}/listing/others/margin/index.html'

        named_tab = await self.browser_manager.new_tab(name='LoanMarginsFetcher', url=LINKED_PAGE)
        file_block = await named_tab.tab.utils.wait_for('div[class*=component-file]', is_css=True)
        file_element = await file_block.query_selector('a[href]')

        await named_tab.tab.close()
        
        return f'{HOME_PAGE}{file_element.attrs["href"]}'

# 2. データフレームを処理するクラス (モジュール内部でのみ使用)
class _LoanMarginsProcessor:
    @staticmethod
    def process_data(file_url: str) -> pd.DataFrame:
        return pd.read_excel(file_url, header=1)

# 3. 外部に公開するクラス (インターフェース)
class LoanMarginsListGetter:
    def __init__(self, browser_manager: BrowserManager):
        self._fetcher = _LoanMarginsFetcher(browser_manager)
        self._processor = _LoanMarginsProcessor()
    
    async def get(self) -> pd.DataFrame:
        file_url = await self._fetcher.fetch_url()
        return self._processor.process_data(file_url)

# 4. 実行コード
if __name__ == '__main__':
    async def main():
        browser_manager = BrowserManager()
        getter = LoanMarginsListGetter(browser_manager)
        return await getter.get()

    loan_margins_list = asyncio.get_event_loop().run_until_complete(main())
    print(loan_margins_list)