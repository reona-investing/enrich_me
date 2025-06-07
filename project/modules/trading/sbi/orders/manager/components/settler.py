from __future__ import annotations

from typing import Optional
from .base import BaseExecutor
from trading.sbi.orders.interface import OrderResult
import pandas as pd
import traceback

class PositionSettlerMixin(BaseExecutor):
    """Handles position settlement"""

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
            selector = f'select[name="nariyuki_condition"] option[value="H"]'
            await named_tab.tab.utils.select_pulldown(selector)
            await self._input_trade_pass()
            await named_tab.tab.utils.click_element('input[id="shouryaku"]', is_css=True)
            await named_tab.tab.utils.wait_for('img[title="注文発注"]', is_css=True)
            await named_tab.tab.utils.click_element('img[title="注文発注"]', is_css=True)
            try:
                await named_tab.tab.utils.wait_for('ご注文を受け付けました。')
                order_id = await self._get_element('注文番号')
                return OrderResult(success=True, order_id=order_id, message=f"{symbol_code}：正常に決済注文が完了しました")
            except Exception:
                return OrderResult(success=False, message=f"{symbol_code}：決済注文に失敗しました", error_code="SETTLEMENT_ERROR")
        except Exception as e:
            error_message = f"{symbol_code}: ポジション決済中にエラーが発生しました: {e}"
            traceback.print_exc()
            return OrderResult(success=False, message=error_message, error_code="SYSTEM_ERROR")

    async def _get_position_elements(self, table_body_css: str) -> dict:
        named_tab = self.browser_manager.get_tab('SBI')
        positions = {}
        try:
            rows = await named_tab.tab.utils.query_selector(f'{table_body_css} > tr', is_all=True)
            if not rows:
                alternative_selectors = ['table tbody tr', '#MAINAREA02_780 tbody tr', 'form table tbody tr']
                for alt_selector in alternative_selectors:
                    rows = await named_tab.tab.utils.query_selector(alt_selector, is_all=True)
                    if rows:
                        break
            if rows:
                for idx, row in enumerate(rows):
                    try:
                        row_text = row.text_all if hasattr(row, 'text_all') else await row.get_text()
                        if '返買' in row_text or '返売' in row_text:
                            code_match = re.search(r'(\d{4})', row_text)
                            if code_match:
                                positions[code_match.group(1)] = idx + 1
                    except Exception:
                        continue
        except Exception:
            traceback.print_exc()
        return positions

    async def _navigate_to_position_settlement(self, element_num: int, table_body_css: str) -> None:
        named_tab = self.browser_manager.get_tab('SBI')
        await named_tab.tab.utils.click_element(f'{table_body_css} > tr:nth-child({element_num}) > td:nth-child(10) > a:nth-child(1) > u > font', is_css=True)
