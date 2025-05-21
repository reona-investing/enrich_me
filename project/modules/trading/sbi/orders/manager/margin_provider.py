from trading.sbi.interface.orders import IMarginProvider
from trading.sbi.browser import SBIBrowserManager, PageNavigator
from bs4 import BeautifulSoup as soup
import re


class SBIMarginProvider(IMarginProvider):
    """SBI証券から証拠金情報を提供するクラス"""
    
    def __init__(self, browser_manager: SBIBrowserManager):
        """コンストラクタ
        
        Args:
            browser_manager (SBIBrowserManager): SBI証券のブラウザセッションを管理するオブジェクト
        """
        self.browser_manager = browser_manager
        self.page_navigator = PageNavigator(self.browser_manager)
        self.margin_power = None  # 信用建余力
        self.buying_power = None  # 買付余力
        
    async def get_available_margin(self) -> float:
        """利用可能な証拠金（信用建余力）を取得する
        
        Returns:
            float: 利用可能な証拠金金額
        """
        if self.margin_power is None:
            await self.refresh()
        return self.margin_power
    
    async def refresh(self) -> None:
        """証拠金情報（信用建余力、買付余力）を最新の情報に更新する"""
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