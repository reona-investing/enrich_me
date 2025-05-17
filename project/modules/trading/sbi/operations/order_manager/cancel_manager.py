from trading.sbi.operations.order_manager.order_manager_base import OrderManagerBase
from trading.sbi.operations.position_manager import PositionManager
from trading.sbi.operations.trade_possibility_manager import TradePossibilityManager
from trading.sbi.browser.sbi_browser_manager import SBIBrowserManager
import traceback
import re
from bs4 import BeautifulSoup as soup


class CancelManager(OrderManagerBase):
    """
    発注操作を管理するクラス（新規注文、再発注、決済など）。
    """
    def __init__(self, browser_manager: SBIBrowserManager):
        super().__init__(browser_manager)
        self.browser_manager = browser_manager
        self.position_manager = PositionManager()
        self.trade_possibility_manager = TradePossibilityManager(self.browser_manager)
        self.has_successfully_ordered = False
        self.error_tickers = []

    async def cancel_all_orders(self) -> None:
        """
        すべての注文をキャンセルする。
        """
        try:
            await self.browser_manager.launch()
            await self.extract_order_list()

            if len(self.order_list_df) == 0:
                print("キャンセルする注文はありません。")
                return

            for i in range(len(self.order_list_df)):
                await self.page_navigator.order_cancel()
                await self._cancel_single_order()
        except Exception as e:
            print(f"注文キャンセル中にエラーが発生しました: {e}")
            traceback.print_exc()  # スタックトレースを出力

    async def _cancel_single_order(self) -> None:
        named_tab = self.browser_manager.get_tab('SBI')
        await named_tab.tab.utils.wait_for('取引パスワード')
        await self._input_trade_pass()
        await named_tab.tab.utils.click_element('input[value=注文取消]', is_css = True)
        await named_tab.tab.utils.wait_for('ご注文を受け付けました。')
        await named_tab.tab.utils.wait(1)
        await self._handle_cancel_response()

    async def _handle_cancel_response(self) -> None:
        named_tab = self.browser_manager.get_tab('SBI')
        html_content = await named_tab.tab.utils.get_html_content()
        html = soup(html_content, "html.parser")
        code_element = html.find("b", string=re.compile("銘柄コード"))
        code = code_element.find_parent("th").find_next_sibling("td").get_text(strip=True)
        unit_element = html.find("b", string=re.compile("株数"))
        unit = unit_element.find_parent("th").find_next_sibling("td").get_text(strip=True)
        unit = int(unit[:-1].replace(',', ''))
        order_type_element = html.find("b", string=re.compile("取引"))
        order_type = order_type_element.find_parent("th").find_next_sibling("td").get_text(strip=True)

        if "ご注文を受け付けました。" in html_content:
            print(f"{code} {unit}株 {order_type}：注文取消が完了しました。")
            await self._edit_position_manager_for_cancel()
        else:
            print(f"{code} {unit}株 {order_type}：注文取消に失敗しました。")

    async def _edit_position_manager_for_cancel(self) -> None:
        named_tab = self.browser_manager.get_tab('SBI')
        html_content = await named_tab.tab.utils.get_html_content()
        html = soup(html_content, "html.parser")
        order_id_element = html.find("b", string=re.compile("注文番号"))
        order_id = order_id_element.find_parent("th").find_next_sibling("td").get_text(strip=True)
        order_type_element = html.find("b", string=re.compile("取引"))
        order_type = order_type_element.find_parent("th").find_next_sibling("td").get_text(strip=True)

        # 注文タイプに応じてステータスタイプを判定
        if any(order in order_type for order in ["信用新規買", "信用新規売", "現物買"]):
            status_type_to_update = 'order_status'
        if any(order in order_type for order in ["信用返済買", "信用返済売", "現物売"]):
            status_type_to_update = 'settlement_status'
            
        # 文字列型の注文IDを数値に変換（必要な場合）
        if isinstance(order_id, str) and order_id.isdigit():
            order_id = int(order_id)
            
        # PositionManagerのupdate_statusメソッドを呼び出し
        success = self.position_manager.update_status(
            order_id, 
            status_type=status_type_to_update, 
            new_status=self.position_manager.STATUS_UNORDERED
        )
        
        # 注文の場合は、待機注文を削除
        if status_type_to_update == 'order_status' and success:
            self.position_manager.remove_waiting_order(order_id)