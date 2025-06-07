import os
import re
import traceback
from typing import Optional
import pandas as pd
from bs4 import BeautifulSoup as soup

from trading.sbi.browser import SBIBrowserManager, PageNavigator
from trading.sbi.orders.interface import OrderRequest, OrderResult


class SBIOrderPlacer:
    """SBI証券での発注処理を担当するクラス"""

    ORDER_PARAM_DICT = {
        '取引': {
            "現物買": "genK",
            "現物売": "genU",
            "信用新規買": "shinK",
            "信用新規売": "shinU",
        },
        '注文タイプ': {
            "指値": 0,
            "成行": 1,
            "逆指値": 2,
        },
        '指値タイプ': {
            "寄指": 'Z',
            "引指": 'I',
            "不成": 'F',
            "IOC指": 'P'
        },
        '成行タイプ': {
            "寄成": 'Y',
            "引成": 'H',
            "IOC成": 'I'
        },
        '逆指値タイプ': {
            "指値": 1,
            "成行": 2,
        },
        '期間': {
            "当日中": 0,
            "今週中": 1,
            "期間指定": 2
        },
        '預り区分': {
            "一般預り": 0,
            "特定預り": 1,
            "NISA預り": 2,
            "旧NISA預り": 3
        },
        '信用取引区分': {
            "制度": 0,
            "一般": 1,
            "日計り": 2
        }
    }

    def __init__(self, browser_manager: SBIBrowserManager):
        self.browser_manager = browser_manager
        self.page_navigator = PageNavigator(browser_manager)

    async def place_order(self, order_request: OrderRequest) -> OrderResult:
        try:
            await self.browser_manager.launch()
            await self.page_navigator.trade()

            await self._select_trade_type(order_request)
            await self._input_stock_and_quantity(order_request)
            await self._input_order_conditions(order_request)
            await self._input_period_conditions(order_request)
            await self._select_deposit_and_credit_type(order_request)

            success, order_id, failed_message = await self._confirm_order(order_request)
            if success:
                message = f"{order_request.symbol_code} {order_request.direction} {order_request.unit}株: 注文が成功しました"
                print(message)
                return OrderResult(success=True, order_id=order_id, message=message)
            else:
                message = f"{order_request.symbol_code} {order_request.direction} {order_request.unit}株: {failed_message}"
                print(message)
                return OrderResult(success=False, message=message)

        except Exception as e:
            error_message = f"注文中にエラーが発生しました: {str(e)}"
            traceback.print_exc()
            return OrderResult(success=False, message=error_message, error_code="SYSTEM_ERROR")

    def _get_selector(self, category: str, key: str) -> str:
        return self.ORDER_PARAM_DICT.get(category, {}).get(key, "")

    async def _select_trade_type(self, order_request: OrderRequest) -> None:
        named_tab = self.browser_manager.get_tab('SBI')
        trade_type = order_request.trade_type
        if not trade_type:
            if order_request.direction == "Long":
                trade_type = "信用新規買"
            else:
                trade_type = "信用新規売"
        selector_id = self._get_selector("取引", trade_type)
        await named_tab.tab.utils.click_element(f'#{selector_id}', is_css=True)

    async def _input_stock_and_quantity(self, order_request: OrderRequest) -> None:
        named_tab = self.browser_manager.get_tab('SBI')
        await named_tab.tab.utils.wait_for('input[name="stock_sec_code"]', is_css=True)
        await named_tab.tab.utils.send_keys_to_element('input[name="stock_sec_code"]', is_css=True, keys=order_request.symbol_code)
        await named_tab.tab.utils.send_keys_to_element('input[name="input_quantity"]', is_css=True, keys=str(order_request.unit))

    async def _input_order_conditions(self, order_request: OrderRequest) -> None:
        named_tab = self.browser_manager.get_tab('SBI')
        order_type = order_request.order_type
        order_type_value = order_request.order_type_value
        await named_tab.tab.utils.click_element(f'input[name="in_sasinari_kbn"][value="{self.ORDER_PARAM_DICT["注文タイプ"][order_type]}"]', is_css=True)
        if order_type == "指値" and order_request.limit_price:
            await named_tab.tab.utils.send_keys_to_element('input[name="input_sasine"]', is_css=True, keys=str(order_request.limit_price))
            if order_type_value:
                selector = f'select[name="sasine_condition"] option[value="{self.ORDER_PARAM_DICT["指値タイプ"][order_type_value]}"]'
                await named_tab.tab.utils.select_pulldown(selector)
        if order_type == "逆指値":
            await self._input_gyakusashine_params(order_request)

    async def _input_gyakusashine_params(self, order_request: OrderRequest) -> None:
        named_tab = self.browser_manager.get_tab('SBI')
        if order_request.trigger_price:
            await named_tab.tab.utils.click_element('#gsn2 > table > tbody > tr > td:nth-child(2) > label:nth-child(5) > input[type=radio]', is_css=True)
            if order_request.stop_order_type:
                stop_order_type_value = self.ORDER_PARAM_DICT["逆指値タイプ"][order_request.stop_order_type]
                selector = f'#gsn2 > table > tbody > tr > td:nth-child(2) > select:nth-child(6) > option:nth-child({stop_order_type_value})'
                await named_tab.tab.utils.click_element(selector, is_css=True)
            await named_tab.tab.utils.send_keys_to_element('#gsn2 > table > tbody > tr > td:nth-child(2) > input[type=text]', is_css=True, keys=str(order_request.trigger_price))
            if order_request.stop_order_type == "指値" and order_request.stop_order_price:
                await named_tab.tab.utils.send_keys_to_element('#gsn2 > table > tbody > tr > td:nth-child(2) > input[type=text]:nth-child(7)', is_css=True, keys=str(order_request.stop_order_price))

    async def _input_period_conditions(self, order_request: OrderRequest) -> None:
        named_tab = self.browser_manager.get_tab('SBI')
        await named_tab.tab.utils.click_element(order_request.period_type)
        if order_request.period_type == "期間指定":
            if order_request.period_value:
                selector = f'select[name="limit_in"] option[value="{order_request.period_value}"]'
                await named_tab.tab.utils.select_pulldown(selector)
            elif order_request.period_index is not None:
                period_option_div = await named_tab.tab.utils.select_element('select[name="limit_in"]', is_css=True)
                period_options = await period_option_div.select('option')
                if order_request.period_index < len(period_options):
                    await period_options[order_request.period_index].click()

    async def _select_deposit_and_credit_type(self, order_request: OrderRequest) -> None:
        named_tab = self.browser_manager.get_tab('SBI')
        trade_section = order_request.trade_section or "特定預り"
        await named_tab.tab.utils.click_element(trade_section)
        margin_trade_section = order_request.margin_trade_section or "日計り"
        await named_tab.tab.utils.click_element(margin_trade_section)

    async def _confirm_order(self, order_request: OrderRequest) -> tuple[bool, Optional[str], str]:
        named_tab = self.browser_manager.get_tab('SBI')
        await self._input_trade_pass()
        await named_tab.tab.utils.click_element('input[id="shouryaku"]', is_css=True)
        await named_tab.tab.utils.wait_for('img[title="注文発注"]', is_css=True)
        await named_tab.tab.utils.click_element('img[title="注文発注"]', is_css=True)
        max_retry = 10
        for _ in range(max_retry):
            try:
                await named_tab.tab.utils.wait_for("ご注文を受け付けました。", timeout=2)
                order_id = await self._get_element('注文番号')
                await self.browser_manager.close_popup()
                return True, order_id, "注文が成功しました"
            except Exception:
                pass
            try:
                failure_selector = '#MAINAREA02_780 > form > table:nth-child(22) > tbody > tr > td > b > p'
                error_element = await named_tab.tab.utils.wait_for(failure_selector, is_css=True, timeout=2)
                error_message = error_element.text if error_element else "不明なエラー"
                await self.browser_manager.close_popup()
                return False, None, error_message
            except Exception:
                pass
        return False, None, "注文処理がタイムアウトしました"

    async def _input_trade_pass(self) -> None:
        named_tab = self.browser_manager.get_tab('SBI')
        await named_tab.tab.utils.wait_for('input[id="pwd3"]', is_css=True)
        await named_tab.tab.utils.send_keys_to_element('input[id="pwd3"]', is_css=True, keys=os.getenv('SBI_TRADEPASS'))

    async def _get_element(self, text: str):
        named_tab = self.browser_manager.get_tab('SBI')
        element = await named_tab.tab.utils.wait_for(text)
        element = element.parent.parent.children[1]
        return re.sub(r'\s+', '', element.text)
