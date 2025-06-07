from __future__ import annotations

from .base import BaseExecutor
from trading.sbi.orders.interface import OrderResult
import pandas as pd
import traceback
from bs4 import BeautifulSoup as soup
import re

class OrderInfoFetcherMixin(BaseExecutor):
    """Fetches active orders and positions"""

    async def get_active_orders(self) -> pd.DataFrame:
        try:
            await self.browser_manager.launch()
            await self._extract_order_list()
            return self.order_list_df
        except Exception as e:
            traceback.print_exc()
            return pd.DataFrame()

    async def get_positions(self) -> pd.DataFrame:
        try:
            await self.browser_manager.launch()
            named_tab = await self.page_navigator.credit_position()
            await named_tab.tab.utils.wait(1)
            html_content = await named_tab.tab.utils.get_html_content()
            return self._parse_positions_table(html_content)
        except Exception as e:
            traceback.print_exc()
            return pd.DataFrame()

    def _parse_positions_table(self, html_content: str) -> pd.DataFrame:
        import unicodedata
        html = soup(html_content, "html.parser")
        table = html.find("td", string=re.compile("銘柄"))
        if table is None:
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
