from bs4 import BeautifulSoup as soup
from trading.sbi.session import LoginHandler
from trading.sbi.browser import BrowserUtils, PageNavigator
import re

class MarginManager:
    def __init__(self, login_handler: LoginHandler):
        """
        信用建余力・買付余力取得クラス
        """
        self.login_handler = login_handler
        self.page_navigator = PageNavigator(self.login_handler)
        self.browser_utils = BrowserUtils(self.login_handler)
        self.margin_power = None
        self.buying_power = None

    async def fetch(self):
        """信用建余力と買付余力を取得して返す"""
        # 口座管理ページに遷移
        await self.page_navigator.account_management()

        # ページのHTMLを取得
        html_content = await self.browser_utils.get_html_content()
        html = soup(html_content, "html.parser")

        # 信用建余力を取得
        div = html.find("div", string=re.compile("信用建余力"))
        if not div:
            raise ValueError("信用建余力の要素が見つかりません。")
        self.margin_power = int(div.find_next("div").getText().strip().replace(",", ""))

        # 買付余力（2営業日後）を取得
        div = html.find("div", string=re.compile("買付余力\\(2営業日後\\)"))
        if not div:
            raise ValueError("買付余力の要素が見つかりません。")
        self.buying_power = int(div.find_next("div").getText().strip().replace(",", ""))