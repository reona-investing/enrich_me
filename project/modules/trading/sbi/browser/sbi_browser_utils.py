from trading.sbi.session import LoginHandler
from utils.browser import BrowserUtils


class SBIBrowserUtils(BrowserUtils):
    def __init__(self, login_handler: LoginHandler):
        """
        SBI証券向けのブラウザ操作を定義します。
        Args:
            login_handler (LoginHandler): SBI証券へのログイン状態を管理
        """
        self.login_handler = login_handler
        self.browser = None
        self.tab = None
        super().__init__()

    async def _launch(self):
        """ SBI証券のWebサイトにサインインしてブラウザとタブを設定する """
        self.tab, self.browser = await self.login_handler.sign_in()

    async def close_popup(self):
        """
        SBI証券サイトで意図せず開いた別タブを閉じる
        """
        await self._launch()  # ブラウザが None にならないようにする
        for tab in self.browser.tabs:
            if self.tab != tab:
                await tab.close()
