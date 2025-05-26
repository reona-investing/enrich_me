from utils.browser.browser_manager import BrowserManager, retry_on_connection_error
from utils.browser.named_tab import NamedTab
from trading.gmail import auth_code_getter
import os


class SBIBrowserManager(BrowserManager):
    def __init__(self):
        """
        SBI証券向けのブラウザ操作を定義します。
        Args:
            browser_manager (BrowserManager): ブラウザおよびタブの操作と管理を司ります。
        """
        self.signed_in = False
        self.browser = None
        super().__init__()

    @retry_on_connection_error
    async def launch(self):
        """ SBI証券のWebサイトにサインインしてブラウザとタブを設定する """
        if not self.signed_in:
            START_URL = "https://www.sbisec.co.jp/ETGate"
            named_tab = await self.new_tab(name='SBI', url=START_URL)
            await self._sign_in(named_tab)
            self.signed_in = True
        else:
            named_tab = self.get_tab('SBI')
        return named_tab


    async def _sign_in(self, named_tab: NamedTab):
        await self._input_credentials(named_tab=named_tab)
        try:
            await named_tab.tab.utils.wait_for('ログアウト', timeout=10)
        except:
            device_code = await auth_code_getter()
            await named_tab.tab.utils.send_keys_to_element('input[name="device_code"]', is_css=False, keys=device_code)
            await named_tab.tab.utils.click_element('input[value="登録"]', is_css=True)
            await named_tab.tab.utils.wait_for('ログアウト', timeout=10)
            

        
    async def _input_credentials(self, named_tab: NamedTab):
        username_col = await named_tab.tab.utils.wait_for('input[name="user_id"]', is_css=True)
        username = os.getenv('SBI_USERNAME')
        if username is None:
            raise Exception('環境変数"SBI_USERNAME"が見つかりません。')
        else:
            await username_col.send_keys(username)
        
        password_col = await named_tab.tab.utils.wait_for('input[name="user_password"]', is_css=True)
        password = os.getenv('SBI_LOGINPASS')
        if password is None:
            raise Exception('環境変数"SBI_LOGINPASS"が見つかりません。')
        else:
            await password_col.send_keys(password)
        
        await named_tab.tab.utils.wait(1)
        login = await named_tab.tab.utils.wait_for('input[name="ACT_login"]')
        await login.click()



if __name__ == '__main__':
    import asyncio
    async def main():
        bu = SBIBrowserManager()
        await bu.launch()
        await asyncio.sleep(20)
        await bu.close_popup()
        await asyncio.sleep(10)
    
    asyncio.get_event_loop().run_until_complete(main())