from __future__ import annotations

import os
import pandas as pd
import re
from trading.sbi.browser import SBIBrowserManager, PageNavigator

class BaseExecutor:
    """Order executor base providing common utilities"""

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
        self.order_list_df = pd.DataFrame()

    def _get_selector(self, category: str, key: str) -> str:
        return self.ORDER_PARAM_DICT.get(category, {}).get(key, "")

    async def _input_trade_pass(self) -> None:
        named_tab = self.browser_manager.get_tab('SBI')
        await named_tab.tab.utils.wait_for('input[id="pwd3"]', is_css=True)
        await named_tab.tab.utils.send_keys_to_element(
            'input[id="pwd3"]', is_css=True, keys=os.getenv('SBI_TRADEPASS'))

    async def _get_element(self, text: str):
        named_tab = self.browser_manager.get_tab('SBI')
        element = await named_tab.tab.utils.wait_for(text)
        element = element.parent.parent.children[1]
        return re.sub(r'\s+', '', element.text)
