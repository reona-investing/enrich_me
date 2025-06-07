from __future__ import annotations

from typing import List
from .base import BaseExecutor
from trading.sbi.orders.interface import OrderResult
from bs4 import BeautifulSoup as soup
import re
import traceback

class OrderCancellerMixin(BaseExecutor):
    """Handles order cancellation"""

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
                    order_list.append(OrderResult(success=True, message=message))
                except Exception:
                    message = "キャンセル処理中にエラーが発生しました"
                    order_list.append(OrderResult(success=False, message=message, error_code="CANCEL_ERROR"))
            return order_list
        except Exception as e:
            error_message = f"注文キャンセル中にエラーが発生しました: {e}"
            traceback.print_exc()
            return [OrderResult(success=False, message=error_message, error_code="SYSTEM_ERROR")]

    async def _extract_order_list(self) -> None:
        named_tab = await self.page_navigator.order_inquiry()
        await named_tab.tab.utils.wait(3)
        html_content = await named_tab.tab.utils.get_html_content()
        html = soup(html_content, "html.parser")
        table = html.find("th", string=re.compile("注文状況"))
        if table is None:
            self.order_list_df = pd.DataFrame()
            return
        table = table.findParent("table")
        data = []
        for tr in table.find("tbody").findAll("tr"):
            if tr.find("td").find("a"):
                row_data = self._extract_order_row_data(tr)
                data.append(row_data)
        if not data:
            self.order_list_df = pd.DataFrame()
            return
        columns = [
            "注文番号", "注文状況", "注文種別", "銘柄", "コード", "取引", "預り", "手数料", "注文日",
            "注文期間", "注文株数", "（未約定）", "執行条件", "注文単価", "現在値", "条件"
        ]
        order_list_df = pd.DataFrame(data, columns=columns)
        order_list_df["注文番号"] = order_list_df["注文番号"].astype(int)
        order_list_df["コード"] = order_list_df["コード"].astype(str)
        self.order_list_df = order_list_df[order_list_df["注文状況"] != "取消中"].reset_index(drop=True)

    def _extract_order_row_data(self, tr) -> list:
        import unicodedata
        row = []
        row.append(tr.findAll("td")[0].getText().strip())
        row.append(tr.findAll("td")[1].getText().strip())
        row.append(tr.findAll("td")[2].getText().strip())
        text = unicodedata.normalize("NFKC", tr.findAll("td")[3].getText().strip())
        row.append(text.splitlines()[0].strip().split(" ")[0])
        row.append(text.splitlines()[0].strip().split(" ")[-1])
        tmp_data = []
        for t in tr.findNext("tr").findAll("td")[0].getText().strip().replace(" ", "").splitlines():
            if t != "" and t != "/":
                tmp_data.append(t)
        if len(tmp_data) == 4:
            row.extend([tmp_data[0] + tmp_data[1], tmp_data[2], tmp_data[3]])
        else:
            row.extend(tmp_data)
        row.extend(tr.findNext("tr").findAll("td")[1].getText().replace(" ", "").strip().splitlines())
        row.append(tr.findNext("tr").findAll("td")[2].getText().replace(" ", "").strip().splitlines()[0])
        row.append(tr.findNext("tr").findAll("td")[2].getText().replace(" ", "").strip().splitlines()[-1])
        row.append(tr.findNext("tr").findAll("td")[3].getText().strip())
        row.extend(tr.findNext("tr").findAll("td")[4].getText().strip().replace(" ", "").splitlines())
        if not tr.findNext("tr").findNext("tr").find("td").find("a"):
            row.append(tr.findNext("tr").findNext("tr").find("td").getText().strip())
        else:
            row.append("--")
        return row
