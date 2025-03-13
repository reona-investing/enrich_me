from datetime import datetime
from pathlib import Path
from utils.paths import Paths
from trading.sbi.browser import SBIBrowserManager
from utils.browser.named_tab import NamedTab

class PageNavigator:
    def __init__(self, browser_manager: SBIBrowserManager):
        '''
        SBI証券のWebサイト上の各ページへの遷移を行います。
        browser_manager (SBIBrowserManager): ブラウザ及びタブの管理・操作を司ります。
        '''
        self.browser_manager = browser_manager
        self.should_return_to_home = False #特殊なレイアウトのページの場合、操作終了後ホームに戻りたい
 

    async def _set_tab(self) -> NamedTab:
        named_tab = await self.browser_manager.launch()
        if self.should_return_to_home:
            await self._home(named_tab)
            self.should_return_to_home = False
        return named_tab


    async def _home(self, named_tab: NamedTab):
        await named_tab.tab.utils.open_url('https://site2.sbisec.co.jp/ETGate/')


    async def home(self) -> NamedTab:
        '''
        トップページに戻ります。
        '''
        named_tab = await self._set_tab()
        await self._home(named_tab)
        return named_tab


    async def domestic_top(self) -> NamedTab:
        '''
        国内株式トップページに遷移します。
        '''
        named_tab = await self._set_tab()
        await named_tab.tab.utils.open_url('https://site0.sbisec.co.jp/marble/domestic/top.do?')
        return named_tab


    async def trade(self) -> NamedTab:
        named_tab = await self._set_tab()
        await named_tab.tab.utils.click_element('img[title="取引"]', is_css=True)
        await named_tab.tab.utils.wait_for('注文入力')
        return named_tab
        

    async def account_management(self) -> NamedTab:
        # 口座管理ページに遷移
        named_tab = await self._set_tab()
        await named_tab.tab.utils.click_element('img[title=口座管理]', is_css=True)
        await named_tab.tab.utils.wait_for('口座サマリー')
        return named_tab


    async def cashflow_transactions(self) -> NamedTab:
        '''
        入出金明細のページに遷移します。
        '''
        named_tab = await self._set_tab()
        await named_tab.tab.utils.click_element('入出金明細')
        await named_tab.tab.utils.wait_for('入出金･振替')
        self.should_return_to_home = True
        return named_tab


    async def trade_history(self) -> NamedTab:
        named_tab = await self._set_tab()
        await named_tab.tab.utils.click_element('取引履歴')
        await named_tab.tab.utils.wait_for('条件選択')
        return named_tab
 

    async def credit_position(self) -> NamedTab:
        named_tab = await self.account_management()
        await named_tab.tab.utils.click_element('area[title=信用建玉]')
        await named_tab.tab.utils.wait_for('信用建玉一覧')
        return named_tab


    async def domestic_margin(self) -> NamedTab:
        '''
        国内株式（信用）のページに遷移します。
        '''
        named_tab = await self.account_management()
        await named_tab.tab.utils.click_element('当日約定一覧')
        await named_tab.tab.utils.click_element('国内株式(信用)')
        return named_tab


    async def domestic_stock(self) -> NamedTab:
        '''
        国内株式（現物）のページに遷移します。
        '''
        named_tab = await self.account_management()
        await named_tab.tab.utils.click_element('当日約定一覧')
        await named_tab.tab.utils.click_element('国内株式(現物)')
        return named_tab


    async def order_inquiry(self) -> NamedTab:
        """
        注文照会のページに遷移します。
        """
        named_tab = await self.trade()
        await named_tab.tab.utils.click_element('注文照会')
        await named_tab.tab.utils.wait_for('未約定注文一覧')
        return named_tab


    async def credit_position_close(self) -> NamedTab:
        named_tab = await self.trade()
        await named_tab.tab.utils.click_element('信用返済')
        await named_tab.tab.utils.wait_for('信用建玉一覧')
        return named_tab


    async def order_cancel(self) -> NamedTab:
        """
        注文照会のページに遷移します。
        """
        named_tab = await self.order_inquiry()
        key_element = await named_tab.tab.utils.wait_for('0ポイント')
        table_elements = key_element.parent.children
        target_element = [e for e in table_elements if '取消' in e.text]
        cancel_element= target_element[0].children[0]
        await cancel_element.click()
        await named_tab.tab.utils.wait_for('注文取消')
        return named_tab


    # TODO 本当はdfを返したい…
    async def fetch_past_margin_trades_csv(self, mydate: datetime) -> NamedTab:
        '''
        指定した日付の取引履歴csvをダウンロードします。
        Args:
            mydate (datetime): 取引履歴を取得する日付
        '''
        named_tab = await self.trade_history()
        button = await named_tab.tab.utils.wait_for('#shinT', is_css=True)
        await button.click()
        element_num = {'from_yyyy':f'{mydate.year}', 'from_mm':f'{mydate.month:02}', 'from_dd':f'{mydate.day:02}',
                       'to_yyyy':f'{mydate.year}', 'to_mm':f'{mydate.month:02}', 'to_dd':f'{mydate.day:02}'}
        for key, value in element_num.items():
            pulldown_selector = f'select[name="ref_{key}"] option[value="{value}"]'
            await named_tab.tab.utils.select_pulldown(pulldown_selector)
        await named_tab.tab.utils.click_element('照会')
        await named_tab.tab.utils.set_download_path(Path(Paths.DOWNLOAD_FOLDER))
        button = await named_tab.tab.utils.wait_for('CSVダウンロード')
        await button.click()
        await named_tab.tab.utils.wait(5)
        return named_tab
    

if __name__ == '__main__':
    import asyncio
    from datetime import datetime

    async def main():
        browser_manager = SBIBrowserManager()
        page_navigator = PageNavigator(browser_manager)
        '''
        await page_navigator.account_management()
        await asyncio.sleep(2)
        await page_navigator.cashflow_transactions()
        await asyncio.sleep(2)
        await page_navigator.credit_position()
        await asyncio.sleep(2)
        await page_navigator.credit_position_close()
        await asyncio.sleep(2)
        await page_navigator.domestic_margin()
        await asyncio.sleep(2)
        await page_navigator.domestic_stock()
        await asyncio.sleep(2)
        await page_navigator.domestic_top()
        await asyncio.sleep(2)
        await page_navigator.home()
        await asyncio.sleep(2)
        await page_navigator.fetch_past_margin_trades_csv(mydate=datetime(2025,3,10))
        await asyncio.sleep(2)
        #await page_navigator.order_cancel()
        #await asyncio.sleep(2)
        '''
        await page_navigator.order_inquiry()
        await asyncio.sleep(2)
        await page_navigator.trade()
        await asyncio.sleep(2)
        await page_navigator.trade_history()
        await asyncio.sleep(2)
    
    asyncio.get_event_loop().run_until_complete(main())




