import os
import re
import traceback
from typing import Optional
import pandas as pd
from bs4 import BeautifulSoup as soup

from trading.sbi.browser import SBIBrowserManager, PageNavigator
from trading.sbi.orders.interface import OrderResult


class SBIPositionManager:
    """建玉の決済や一覧取得を担当するクラス"""

    def __init__(self, browser_manager: SBIBrowserManager):
        self.browser_manager = browser_manager
        self.page_navigator = PageNavigator(browser_manager)

    async def settle_position(self, symbol_code: str, unit: Optional[int] = None) -> OrderResult:
        try:
            await self.browser_manager.launch()
            named_tab = await self.page_navigator.credit_position_close()
            table_body_css = '#MAINAREA02_780 > form > table:nth-child(18) > tbody > tr > td > table > tbody > tr > td > table > tbody'
            positions = await self._get_position_elements(table_body_css)
            if not positions or symbol_code not in positions:
                return OrderResult(success=False, message=f"{symbol_code}: 該当する建玉が見つかりません", error_code="POSITION_NOT_FOUND")
            element_num = positions[symbol_code]
            await self._navigate_to_position_settlement(element_num, table_body_css)
            if unit is None:
                await named_tab.tab.utils.click_element('input[value="全株指定"]', is_css=True)
            else:
                await named_tab.tab.utils.send_keys_to_element('input[name="input_settlement_quantity"]', is_css=True, keys=str(unit))
            await named_tab.tab.utils.click_element('input[value="注文入力へ"]', is_css=True)
            order_type_elements = await named_tab.tab.utils.select_all('input[name="in_sasinari_kbn"]')
            await order_type_elements[1].click()
            selector = 'select[name="nariyuki_condition"] option[value="H"]'
            await named_tab.tab.utils.select_pulldown(selector)
            await self._input_trade_pass()
            await named_tab.tab.utils.click_element('input[id="shouryaku"]', is_css=True)
            await named_tab.tab.utils.wait_for('img[title="注文発注"]', is_css=True)
            await named_tab.tab.utils.click_element('img[title="注文発注"]', is_css=True)
            try:
                await named_tab.tab.utils.wait_for('ご注文を受け付けました。')
                order_id = await self._get_element('注文番号')
                message = f"{symbol_code}：正常に決済注文が完了しました"
                print(message)
                return OrderResult(success=True, order_id=order_id, message=message)
            except Exception:
                message = f"{symbol_code}：決済注文に失敗しました"
                print(message)
                return OrderResult(success=False, message=message, error_code="SETTLEMENT_ERROR")
        except Exception as e:
            error_message = f"{symbol_code}: ポジション決済中にエラーが発生しました: {str(e)}"
            print(error_message)
            traceback.print_exc()
            return OrderResult(success=False, message=error_message, error_code="SYSTEM_ERROR")

    async def get_positions(self) -> pd.DataFrame:
        try:
            await self.browser_manager.launch()
            named_tab = await self.page_navigator.credit_position()
            await named_tab.tab.utils.wait(1)
            html_content = await named_tab.tab.utils.get_html_content()
            positions_df = self._parse_positions_table(html_content)
            return positions_df
        except Exception as e:
            print(f"ポジション一覧の取得中にエラーが発生しました: {e}")
            traceback.print_exc()
            return pd.DataFrame()

    async def _input_trade_pass(self) -> None:
        named_tab = self.browser_manager.get_tab('SBI')
        await named_tab.tab.utils.wait_for('input[id="pwd3"]', is_css=True)
        await named_tab.tab.utils.send_keys_to_element('input[id="pwd3"]', is_css=True, keys=os.getenv('SBI_TRADEPASS'))

    async def _get_element(self, text: str):
        named_tab = self.browser_manager.get_tab('SBI')
        element = await named_tab.tab.utils.wait_for(text)
        element = element.parent.parent.children[1]
        return re.sub(r'\s+', '', element.text)

    async def _get_position_elements(self, table_body_css: str) -> dict:
        named_tab = self.browser_manager.get_tab('SBI')
        positions = {}
        try:
            rows = await named_tab.tab.utils.query_selector(f'{table_body_css} > tr', is_all=True)
            if not rows:
                alternative_selectors = [
                    'table tbody tr',
                    '#MAINAREA02_780 tbody tr',
                    'form table tbody tr'
                ]
                for selector in alternative_selectors:
                    try:
                        rows = await named_tab.tab.utils.query_selector(selector, is_all=True)
                        if rows:
                            break
                    except Exception:
                        continue
            if rows:
                for idx, row in enumerate(rows):
                    try:
                        row_text = await row.get_inner_text()
                        code_patterns = [r'コード\s*(\d{4})', r'([0-9]{4})', r'(\d{4})\s*[^\d]']
                        code = None
                        for pattern in code_patterns:
                            code_match = re.search(pattern, row_text)
                            if code_match:
                                code = code_match.group(1)
                                break
                        if code:
                            positions[code] = idx + 1
                    except Exception:
                        continue
        except Exception as e:
            print(f"建玉要素の取得中にエラーが発生しました: {e}")
            traceback.print_exc()
        return positions

    async def _navigate_to_position_settlement(self, element_num: int, table_body_css: str) -> None:
        named_tab = self.browser_manager.get_tab('SBI')
        await named_tab.tab.utils.click_element(f'{table_body_css} > tr:nth-child({element_num}) > td:nth-child(10) > a:nth-child(1) > u > font', is_css=True)

    def _parse_positions_table(self, html_content: str) -> pd.DataFrame:
        import unicodedata
        html = soup(html_content, "html.parser")
        table = html.find("td", string=re.compile("銘柄"))
        if table is None:
            print('保有建玉はありません。')
            return pd.DataFrame()
        table = table.findParent("table")
        data = []
        for tr in table.find("tbody").findAll("tr"):
            if tr.find("td").find("a"):
                row = []
                text = unicodedata.normalize("NFKC", tr.findAll("td")[0].getText().strip())
                row.append(text[-4:])
                row.append(text[:-4])
                row.append(tr.findAll("td")[1].getText().strip())
                text = unicodedata.normalize("NFKC", tr.findAll("td")[5].getText().strip())
                row.append(text.splitlines()[0].strip().split(" ")[0])
                text = unicodedata.normalize("NFKC", tr.findAll("td")[6].getText().strip())
                numbers = text.split("\n")
                row.append(numbers[0])
                row.append(numbers[1])
                data.append(row)
        if not data:
            return pd.DataFrame()
        columns = ["証券コード", "銘柄", "売・買建", "建株数", "建単価", "現在値"]
        df = pd.DataFrame(data, columns=columns)
        df["証券コード"] = df["証券コード"].astype(str)
        df["建株数"] = df["建株数"].str.replace(',', '').astype(int)
        df["建単価"] = df["建単価"].str.replace(',', '').astype(float)
        df["現在値"] = df["現在値"].str.replace(',', '').astype(float)
        df["建価格"] = df["建株数"] * df["建単価"]
        df["評価額"] = df["建株数"] * df["現在値"]
        df['評価損益'] = df["評価額"] - df["建価格"]
        df.loc[df['売・買建'] == '売建', '評価損益'] = df["建価格"] - df["評価額"]
        return df
