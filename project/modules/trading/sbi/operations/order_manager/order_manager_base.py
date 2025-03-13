from trading.sbi.browser import PageNavigator, SBIBrowserManager
import re
import os
import unicodedata
import pandas as pd
from bs4 import BeautifulSoup as soup

class OrderManagerBase:
    order_param_dicts: dict = {
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
        '期間': {
            "当日中": 0,
            "今週中": 1,
            "期間指定": 2
        },
        '預り区分': {
            "一般預り": 0,
            "特定預り": 1,
            "NISA預り": 2
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

    def _get_selector(self, category: str, key: str) -> str:
        """
        指定されたカテゴリとキーに対応するセレクタを返す。
        """
        return OrderManagerBase.order_param_dicts.get(category, {}).get(key, "")

    async def _get_element(self, text: str):
        named_tab = self.browser_manager.get_tab('SBI')
        element = await named_tab.tab.utils.wait_for(text)
        element = element.parent.parent.children[1]
        return re.sub(r'\s+', '', element.text)

    async def _send_order(self):
        named_tab = self.browser_manager.get_tab('SBI')
        await self._input_trade_pass()
        await named_tab.tab.utils.click_element('input[id="shouryaku"]', is_css = True)
        await named_tab.tab.utils.click_element('img[title="注文発注"]', is_css = True)

    async def _input_trade_pass(self):
        '''取引パスワードを入力する。'''
        named_tab = self.browser_manager.get_tab('SBI')
        await named_tab.tab.utils.send_keys_to_element('input[id="pwd3"]',
                                                      is_css = True,
                                                      keys = os.getenv('SBI_TRADEPASS'))

    async def extract_order_list(self) -> None:
        """
        現在の注文一覧を取得する。
        """
        try:
            table = await self._fetch_order_list_table()
            if table is None:
                print('発注中の注文はありません。')
                self.order_list_df = pd.DataFrame()
            self.order_list_df = self._convert_table_to_df(table)

        except Exception as e:
            print(f"注文リストの取得中にエラーが発生しました: {e}")
    
    async def _fetch_order_list_table(self):
        named_tab = await self.page_navigator.order_inquiry()
        await named_tab.tab.utils.wait(3)
        html_content = await named_tab.tab.utils.get_html_content()
        html = soup(html_content, "html.parser")
        table = html.find("th", string=re.compile("注文状況"))
        return table
    
    def _convert_table_to_df(self, table):
        table = table.findParent("table")
        data = []
        for tr in table.find("tbody").findAll("tr"):
            if tr.find("td").find("a"):
                data = self._append_order_to_list(tr, data)

        columns = [
            "注文番号", "注文状況", "注文種別", "銘柄", "コード", "取引", "預り", "手数料", "注文日",
            "注文期間", "注文株数", "（未約定）", "執行条件", "注文単価", "現在値", "条件"
        ]
        order_list_df = pd.DataFrame(data, columns=columns)
        order_list_df["注文番号"] = order_list_df["注文番号"].astype(int)
        order_list_df["コード"] = order_list_df["コード"].astype(str)
        return order_list_df[order_list_df["注文状況"] != "取消中"].reset_index(drop=True)

    def _append_order_to_list(self, tr: object, data: list) -> list:
        """
        注文情報をリストに追加する。
        """
        row = []
        row.append(tr.findAll("td")[0].getText().strip())  # 注文番号
        row.append(tr.findAll("td")[1].getText().strip())  # 注文状況
        row.append(tr.findAll("td")[2].getText().strip())  # 注文種別
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
        data.append(row)
        return data