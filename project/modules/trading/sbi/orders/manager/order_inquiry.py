import re
import pandas as pd
from bs4 import BeautifulSoup as soup

from trading.sbi.browser import SBIBrowserManager, PageNavigator


class SBIOrderInquiry:
    """注文一覧を取得する責務を持つクラス"""

    def __init__(self, browser_manager: SBIBrowserManager):
        self.browser_manager = browser_manager
        self.page_navigator = PageNavigator(browser_manager)
        self.order_list_df = pd.DataFrame()

    async def get_active_orders(self) -> pd.DataFrame:
        try:
            await self.browser_manager.launch()
            await self._extract_order_list()
            return self.order_list_df
        except Exception as e:
            print(f"注文一覧の取得中にエラーが発生しました: {e}")
            return pd.DataFrame()

    async def _extract_order_list(self) -> None:
        named_tab = await self.page_navigator.order_inquiry()
        await named_tab.tab.utils.wait(3)

        html_content = await named_tab.tab.utils.get_html_content()
        html = soup(html_content, "html.parser")

        table = html.find("th", string=re.compile("注文状況"))
        if table is None:
            print('発注中の注文はありません。')
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
        for t in tr.findNext("tr").findAll("td")[0].getText().strip().replace("\u3000", "").splitlines():
            if t != "" and t != "/":
                tmp_data.append(t)

        if len(tmp_data) == 4:
            row.extend([tmp_data[0] + tmp_data[1], tmp_data[2], tmp_data[3]])
        else:
            row.extend(tmp_data)

        row.extend(tr.findNext("tr").findAll("td")[1].getText().replace("\u3000", "").strip().splitlines())
        row.append(tr.findNext("tr").findAll("td")[2].getText().replace("\u3000", "").strip().splitlines()[0])
        row.append(tr.findNext("tr").findAll("td")[2].getText().replace("\u3000", "").strip().splitlines()[-1])
        row.append(tr.findNext("tr").findAll("td")[3].getText().strip())
        row.extend(tr.findNext("tr").findAll("td")[4].getText().strip().replace("\u3000", "").splitlines())

        if not tr.findNext("tr").findNext("tr").find("td").find("a"):
            row.append(tr.findNext("tr").findNext("tr").find("td").getText().strip())
        else:
            row.append("--")

        return row
