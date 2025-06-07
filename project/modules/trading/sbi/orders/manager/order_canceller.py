import os
import re
import traceback
from typing import List
from bs4 import BeautifulSoup as soup

from trading.sbi.browser import SBIBrowserManager, PageNavigator
from trading.sbi.orders.interface import OrderResult
from .order_inquiry import SBIOrderInquiry


class SBIOrderCanceller(SBIOrderInquiry):
    """全注文のキャンセル処理を担当するクラス"""

    def __init__(self, browser_manager: SBIBrowserManager):
        super().__init__(browser_manager)
        self.page_navigator = PageNavigator(browser_manager)

    async def cancel_all_orders(self) -> List[OrderResult]:
        try:
            order_list: List[OrderResult] = []
            await self.browser_manager.launch()
            await self.page_navigator.order_inquiry()
            await self._extract_order_list()

            for _ in range(len(self.order_list_df)):
                await self.page_navigator.order_cancel()
                named_tab = self.browser_manager.get_tab('SBI')
                await named_tab.tab.utils.wait_for('取引パスワード')
                await self._input_trade_pass()
                await named_tab.tab.utils.click_element('input[value=注文取消]', is_css=True)

                try:
                    await named_tab.tab.utils.wait_for('ご注文を受け付けました。')
                    html_content = await named_tab.tab.utils.get_html_content()
                    html = soup(html_content, "html.parser")
                    code_element = html.find("b", string=re.compile("銘柄コード"))
                    code = code_element.find_parent("th").find_next_sibling("td").get_text(strip=True)
                    unit_element = html.find("b", string=re.compile("株数"))
                    unit = unit_element.find_parent("th").find_next_sibling("td").get_text(strip=True)
                    unit = int(unit[:-1].replace(',', ''))
                    order_type_element = html.find("b", string=re.compile("取引"))
                    order_type = order_type_element.find_parent("th").find_next_sibling("td").get_text(strip=True)
                    message = f"{code} {unit}株 {order_type}：注文キャンセルが完了しました"
                    print(message)
                    order_list.append(OrderResult(success=True, message=message))
                except Exception:
                    message = "キャンセル処理中にエラーが発生しました"
                    print(message)
                    order_list.append(OrderResult(success=False, message=message, error_code="CANCEL_ERROR"))
        except Exception as e:
            error_message = f"注文キャンセル中にエラーが発生しました: {str(e)}"
            traceback.print_exc()
            order_list.append(OrderResult(success=False, message=error_message, error_code="SYSTEM_ERROR"))
        finally:
            return order_list

    async def _input_trade_pass(self) -> None:
        named_tab = self.browser_manager.get_tab('SBI')
        await named_tab.tab.utils.wait_for('input[id="pwd3"]', is_css=True)
        await named_tab.tab.utils.send_keys_to_element('input[id="pwd3"]', is_css=True, keys=os.getenv('SBI_TRADEPASS'))
