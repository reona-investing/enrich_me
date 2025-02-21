from trading.sbi.session import LoginHandler
from datetime import datetime
from pathlib import Path
from utils.paths import Paths
from trading.sbi.browser import BrowserUtils

class PageNavigator:
    def __init__(self, login_handler: LoginHandler):
        '''
        SBI証券のWebサイト上の各ページへの遷移を行います。
        login_handler (LoginHandler): SBI証券へのログイン状態を確認します。
        '''
        self.login_handler = login_handler
        self.browser_utils = BrowserUtils(login_handler)
        self.tab = None
        self.should_return_to_home = False #特殊なレイアウトのページの場合、操作終了後ホームに戻りたい
 

    async def _set_tab(self):
        self.tab = await self.login_handler.sign_in()  # LoginHandlerを使ってログイン
        if self.should_return_to_home:
            await self._home()
            self.should_return_to_home = False
    
    async def _click_and_wait(self, selector: str, is_css: bool = False):
        '''
        指定の要素をクリックして1秒待機します。
        Args:
            selector (str): 探す要素を文字列またはcssセレクタで指定
            is_css (bool): Trueならcssセレクタ、Falseなら文字列
        '''
        await self._set_tab()
        if is_css:
            button = await self.tab.select(selector)
        else:
            button = await self.tab.find(selector)
        await button.click()
        await self.tab.wait(1)

    async def _home(self):
        await self.tab.get('https://site2.sbisec.co.jp/ETGate/')


    async def home(self):
        '''
        トップページに戻ります。
        '''
        await self._set_tab()
        await self._home()


    async def domestic_top(self):
        '''
        国内株式トップページに遷移します。
        '''
        await self._set_tab()
        await self.tab.get('https://site0.sbisec.co.jp/marble/domestic/top.do?')


    async def trade(self):
        await self._set_tab()
        await self._click_and_wait('img[title="取引"]', is_css=True)
        

    async def account_management(self):
        # 口座管理ページに遷移
        await self._set_tab()
        await self._click_and_wait('img[title=口座管理]', is_css=True)


    async def cashflow_transactions(self):
        '''
        入出金明細のページに遷移します。
        '''
        await self._set_tab()
        await self._click_and_wait('入出金明細')
        self.should_return_to_home = True


    async def trade_history(self):
        await self._set_tab()
        button = await self._click_and_wait('取引履歴')
 

    async def credit_position(self):
        await self.account_management()
        button = await self._click_and_wait('area[title=信用建玉]')


    async def domestic_margin(self):
        '''
        国内株式（信用）のページに遷移します。
        '''
        await self.account_management()
        await self._click_and_wait('当日約定一覧')
        await self._click_and_wait('国内株式(信用)')


    async def domestic_stock(self):
        '''
        国内株式（現物）のページに遷移します。
        '''
        await self.account_management()
        await self._click_and_wait('当日約定一覧')
        await self._click_and_wait('国内株式(現物)')


    async def order_inquiry(self):
        """
        注文照会のページに遷移します。
        """
        await self.trade()
        await self._click_and_wait('注文照会')
        await self.browser_utils.wait_for('未約定注文一覧')


    async def order_cancel(self):
        """
        注文照会のページに遷移します。
        """
        await self.order_inquiry()
        await self._click_and_wait('取消')
        await self.browser_utils.wait_for('注文取消')


    async def fetch_past_margin_trades_csv(self, mydate: datetime):
        '''
        指定した日付の取引履歴csvをダウンロードします。
        Args:
            mydate (datetime): 取引履歴を取得する日付
        '''
        await self.trade_history()
        button = await self.tab.select('#shinT')
        await button.click()
        element_num = {'from_yyyy':f'{mydate.year}', 'from_mm':f'{mydate.month:02}', 'from_dd':f'{mydate.day:02}',
                       'to_yyyy':f'{mydate.year}', 'to_mm':f'{mydate.month:02}', 'to_dd':f'{mydate.day:02}'}
        for key, value in element_num.items():
            pulldown_selector = f'select[name="ref_{key}"] option[value="{value}"]'
            await self.browser_utils.select_pulldown(pulldown_selector)
        await self._click_and_wait('照会')
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




