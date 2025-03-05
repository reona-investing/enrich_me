from trading.sbi.operations.order_manager.order_manager_base import OrderManagerBase
from trading.sbi.operations.position_manager import PositionManager
from trading.sbi.operations.trade_possibility_manager import TradePossibilityManager
from trading.sbi.session.login_handler import LoginHandler
import os
import re
import pandas as pd
from bs4 import BeautifulSoup as soup
import unicodedata

class SettlementManager(OrderManagerBase):
    """
    発注操作を管理するクラス（新規注文、再発注、決済など）。
    """
    def __init__(self, login_handler: LoginHandler):
        super().__init__(login_handler)
        self.login_handler = login_handler
        self.position_manager = PositionManager()
        self.trade_possibility_manager = TradePossibilityManager(self.login_handler)
        self.error_tickers = []

    async def settle_all_margins(self):
        await self._extract_margin_list()
        await self.extract_order_list()
        if len(self.margin_list_df) == 0:
            print('信用建玉がありません。決済処理を中断します。')
            return
        
        margin_tickers, ordered_tickers = self._prepare_settlement_data() 
        if sorted(margin_tickers) == sorted(ordered_tickers):
            print('すべての信用建玉の決済注文を発注済みです。')
            return
        
        print(f'保有建玉：{len(margin_tickers)}件')
        print(f'発注済み：{len(ordered_tickers)}件')
        
        await self.page_navigator.credit_position_close()

        table_body_css = '#MAINAREA02_780 > form > table:nth-child(18) > tbody > tr > td > \
            table > tbody > tr > td > table > tbody'
        css_element_nums = await self._get_css_element_nums(table_body_css)

        for ticker, element_num in zip(margin_tickers, css_element_nums):
            if ticker in ordered_tickers:
                print(f'{ticker}はすでに決済発注済です。')
                continue
            await self._process_single_settlement_order(ticker, element_num, table_body_css)

        print(f'全銘柄の決済処理が完了しました。')

    async def _extract_margin_list(self):
        await self.page_navigator.credit_position()
        await self.browser_utils.wait(1)
        html_content = await self.browser_utils.get_html_content()
        self.margin_list_df = self._parse_margin_table(html_content)

    def _prepare_settlement_data(self) -> tuple[list[str], list[str]]:
        margin_tickers = self.margin_list_df.sort_values(by="証券コード")["証券コード"].unique().tolist()
        ordered_tickers = self.order_list_df.sort_values(by="コード")["コード"].unique().tolist() if len(self.order_list_df) > 0 else []
        return margin_tickers, ordered_tickers

    def _parse_margin_table(self, html_content: str) -> pd.DataFrame:
        html = soup(html_content, "html.parser")
        table = html.find("td", string=re.compile("銘柄"))
        if table is None:
            print('保有建玉はありません。')
            return pd.DataFrame()
        table = table.findParent("table")

        data = []
        for tr in table.find("tbody").findAll("tr"):
            if tr.find("td").find("a"):
                data = self._append_margin_to_list(tr, data)

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

    def _append_margin_to_list(self, tr:object, data:list) -> list:
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
        return data

    async def _get_css_element_nums(self, table_body_css: str) -> list[int]:
        await self.browser_utils.wait(5)
        mylist = await self.browser_utils.query_selector(f'{table_body_css} > tr', is_all=True)
        css_element_nums = []
        for num, element in enumerate(mylist):
            element_text = element.text_all
            if '返買' in element_text or '返売' in element_text:
                css_element_nums.append(num + 1)
        return css_element_nums


    async def _process_single_settlement_order(self, ticker: str, element_num: int, table_body_css: str) -> None:
        retry_count = 0
        await self.page_navigator.credit_position_close()

        await self._navigate_to_individual_tickers_page(element_num, table_body_css)
        await self._input_order_conditions()
        await self._send_order()

        try:
            await self.browser_utils.wait_for('ご注文を受け付けました。')                
            print(f"{ticker}：正常に決済注文完了しました。")
        except:
            if retry_count < 3:
                print(f"{ticker}：発注失敗。再度発注を試みます。")
                retry_count += 1
            else:
                print(f"{ticker}：発注失敗。リトライ回数の上限に達しました。")
                self.error_tickers.append(ticker)
                retry_count = 0

        await self.browser_utils.wait(1)
        await self._update_settlement_status(ticker)
        

        retry_count = 0

    async def _navigate_to_individual_tickers_page(self, element_num: int, table_body_css: str):
        await self.browser_utils.wait_and_click(
            f'{table_body_css} > tr:nth-child({element_num}) > td:nth-child(10) > a:nth-child(1) > u > font', 
            is_css = True
            )

    async def _input_order_conditions(self) -> None:
        await self.browser_utils.wait_and_click('input[value="全株指定"]', is_css = True)
        await self.browser_utils.wait_and_click('input[value="注文入力へ"]', is_css = True)
        await self.browser_utils.wait(2)
        order_type_elements = await self.browser_utils.select_all('input[name="in_sasinari_kbn"]')
        await order_type_elements[1].click()  # 成行
        selector = f'select[name="nariyuki_condition"] option[value="H"]'
        await self.browser_utils.select_pulldown(selector)
    

    async def _update_settlement_status(self, ticker: str) -> None:
        extracted_unit = await self.browser_utils.wait_for('株数')
        extracted_unit = await self._get_element('株数')
        extracted_unit = int(extracted_unit[:-1].replace(',', ''))
        trade_type = await self._get_element('取引')
        if '信用返済買' in trade_type:
            trade_type = '信用新規売'
        if '信用返済売' in trade_type:
            trade_type = '信用新規買'
        order_id = await self._get_element('注文番号')
        self.position_manager.update_by_symbol(ticker, 'settlement_id', order_id)
        self.position_manager.update_by_symbol(ticker, 'settlement_status', self.position_manager.STATUS_ORDERED)
        self.position_manager.update_status(order_id, status_type = 'settlement_order', new_status = self.position_manager.STATUS_ORDERED)