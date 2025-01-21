from trading.sbi.session import LoginHandler
from datetime import datetime
from pathlib import Path
from utils.paths import Paths
from trading.sbi.browser.browser_utils import BrowserUtils

class PageNavigator:
    def __init__(self, login_handler: LoginHandler):
        '''
        SBI証券のWebサイト上の各ページへの遷移を行います。
        login_handler (LoginHandler): SBI証券へのログイン状態を確認します。
        '''
        self.login_handler = login_handler
        self.tab = None
        self.should_return_to_home = False #特殊なレイアウトのページの場合、操作終了後ホームに戻りたい
 

    async def _set_tab(self):
        await self.login_handler.sign_in()  # LoginHandlerを使ってログイン
        self.tab = self.login_handler.session.tab
        if self.should_return_to_home:
            await self.home()
            self.should_return_to_home = False


    async def home(self):
        '''
        トップページに戻ります。
        '''
        await self._set_tab()
        await self.tab.get('https://site2.sbisec.co.jp/ETGate/')


    async def domestic_margin(self):
        '''
        国内株式（信用）のページに遷移します。
        '''
        await self._set_tab()
        button = await self.tab.select('img[title=口座管理]')
        await button.click()
        await self.tab.wait(1)
        button = await self.tab.find('当日約定一覧')
        await button.click()
        await self.tab.wait(1)
        button = await self.tab.find('国内株式(信用)')
        await button.click()
        await self.tab.wait(1)    


    async def domestic_stock(self):
        '''
        国内株式（現物）のページに遷移します。
        '''
        await self._set_tab()
        button = await self.tab.select('img[title=口座管理]')
        await button.click()
        await self.tab.wait(1)
        button = await self.tab.find('当日約定一覧')
        await button.click()
        await self.tab.wait(1)
        button = await self.tab.find('国内株式(現物)')
        await button.click()
        await self.tab.wait(1)  


    async def cashflow_transactions(self):
        '''
        入出金明細のページに遷移します。
        '''
        await self._set_tab()
        button = await self.tab.find('入出金明細')
        await button.click()
        await self.tab.wait(1)
        self.should_return_to_home = True


    async def fetch_past_margin_trades_csv(self, mydate: datetime):
        '''
        指定した日付の取引履歴csvをダウンロードします。
        Args:
            mydate (datetime): 取引履歴を取得する日付
        '''
        await self._set_tab()
        button = await self.tab.find('取引履歴')
        await button.click()
        await self.tab.wait(1)
        button = await self.tab.select('#shinT')
        await button.click()
        element_num = {'from_yyyy':f'{mydate.year}', 'from_mm':f'{mydate.month:02}', 'from_dd':f'{mydate.day:02}',
                       'to_yyyy':f'{mydate.year}', 'to_mm':f'{mydate.month:02}', 'to_dd':f'{mydate.day:02}'}
        for key, value in element_num.items():
            pulldown_selector = f'select[name="ref_{key}"] option[value="{value}"]'
            await BrowserUtils.select_pulldown(self.tab, pulldown_selector)
        button = await self.tab.find('照会')
        await button.click()
        await self.tab.wait(1)
        await self.tab.set_download_path(Path(Paths.DOWNLOAD_FOLDER))
        button = await self.tab.find('CSVダウンロード')
        await button.click()
        await self.tab.wait(5)
    


if __name__ == '__main__':
    import asyncio

    async def main():
        login_handler = LoginHandler()
        page_navigator = PageNavigator(login_handler)
        await page_navigator.domestic_margin()
    
    asyncio.run(main())




