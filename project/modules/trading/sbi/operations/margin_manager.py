from bs4 import BeautifulSoup as soup
from trading.sbi.browser import SBIBrowserManager, PageNavigator
import re

class MarginManager:
    def __init__(self, browser_manager: SBIBrowserManager):
        """
        信用建余力・買付余力取得クラス
        """
        self.browser_manager = browser_manager
        self.page_navigator = PageNavigator(self.browser_manager)
        self.margin_power = None
        self.buying_power = None

    async def fetch(self):
        """信用建余力と買付余力を取得して返す"""
        await self.browser_manager.launch()
        # 口座管理ページに遷移
        named_tab = await self.page_navigator.account_management()

        # ページのHTMLを取得
        html_content = await named_tab.tab.utils.get_html_content()
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


if __name__ == '__main__':
    import asyncio
    async def main():
        browser_manager = SBIBrowserManager()
        mm = MarginManager(browser_manager)
        await mm.fetch()
        print(mm.buying_power)
        print(mm.margin_power)
    
    asyncio.get_event_loop().run_until_complete(main())