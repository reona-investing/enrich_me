from trading.sbi.operations.position_manager import PositionManager
from trading.sbi.operations.trade_possibility_manager import TradePossibilityManager
from trading.sbi.operations.trade_parameters import TradeParameters
from trading.sbi.browser import PageNavigator, SBIBrowserUtils
from trading.sbi.session.login_handler import LoginHandler
import os
import re
import unicodedata
import pandas as pd
from bs4 import BeautifulSoup as soup
import traceback
import asyncio

# ヘルパー関数の定義
def _get_selector(order_param_dicts: dict, category: str, key: str) -> str:
    """
    指定されたカテゴリとキーに対応するセレクタを返す。
    """
    return order_param_dicts.get(category, {}).get(key, "")

class OrderManager:
    """
    発注操作を管理するクラス（新規注文、再発注、決済など）。
    """

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

    def __init__(self, login_handler: LoginHandler):
        self.login_handler = login_handler
        self.position_manager = PositionManager()
        self.trade_possibility_manager = TradePossibilityManager(self.login_handler)
        self.page_navigator = PageNavigator(self.login_handler)
        self.browser_utils = SBIBrowserUtils(self.login_handler)
        self.has_successfully_ordered = False
        self.error_tickers = []

    async def place_new_order(self, trade_params: TradeParameters) -> bool:
        """
        指定された取引パラメータを使用して新規注文を行う。
        """
        try:
            await self.page_navigator.trade()
            await self._select_trade_type(trade_params)
            await self._input_stock_and_quantity(trade_params)
            await self._input_sashinari_params(trade_params)
            await self._get_duration_params(trade_params)
            await self._select_deposit_and_credit_type(trade_params)
            has_successfully_ordered = await self._confirm_order(trade_params)
            
        except Exception as e:
            print(f"注文中にエラーが発生しました: {e}")
            traceback.print_exc()  # スタックトレースを出力
            self.error_tickers.append(trade_params.symbol_code)
            has_successfully_ordered = False
        finally:
            return has_successfully_ordered

    async def _select_trade_type(self, trade_params: TradeParameters) -> None:
        """
        取引タイプを選択する。
        """
        await self.browser_utils.wait_and_click(f'#{_get_selector(self.order_param_dicts, "取引", trade_params.trade_type)}')

    async def _input_stock_and_quantity(self, trade_params: TradeParameters) -> None:
        """
        銘柄コードと数量を入力する。
        """
        await self.browser_utils.send_keys_to_element('input[name="stock_sec_code"]',
                                                      is_css = True, 
                                                      keys = trade_params.symbol_code)

        await self.browser_utils.send_keys_to_element('input[name="input_quantity"]', 
                                                      is_css = True,
                                                      keys = str(trade_params.unit))

    async def _input_sashinari_params(self, trade_params: TradeParameters) -> None:
        await self.browser_utils.click_element(trade_params.order_type)
        
        if trade_params.order_type == '成行':
            if trade_params.order_type_value is not None:
                selector = \
                    f'select[name="nariyuki_condition"] option[value="{self.order_param_dicts["成行タイプ"][trade_params.order_type_value]}"]'
                await self.browser_utils.select_pulldown(selector)

        if trade_params.order_type == "指値":
            await self.browser_utils.send_keys_to_element('#gsn0 > input[type=text]',
                                                          is_css = True,
                                                          keys = str(trade_params.limit_order_price))

            if trade_params.order_type_value is not None:
                selector = \
                    f'select[name="sasine_condition"] option[value="{self.order_param_dicts["指値タイプ"][trade_params.order_type_value]}"]'
                await self.browser_utils.select_pulldown(selector)

        if trade_params.order_type == "逆指値":
            await self.browser_utils.click_element(
                '#gsn2 > table > tbody > tr > td:nth-child(2) > label:nth-child(5) > input[type=radio]',
                is_css = True)
            await self.browser_utils.select_element(
                '#gsn2 > table > tbody > tr > td:nth-child(2) > select:nth-child(6)',
                is_css = True)
            await self.browser_utils.send_keys_to_element(
                f'#gsn2 > table > tbody > tr > td:nth-child(2) > select:nth-child(6) > option:nth-child({self.order_param_dicts["逆指値タイプ"][trade_params.stop_order_type]})',
                is_css = True,
                keys = trade_params.stop_order_trigger_price)

            if trade_params.stop_order_type == "指値":
                await self.browser_utils.send_keys_to_element(
                    '#gsn2 > table > tbody > tr > td:nth-child(2) > input[type=text]:nth-child(7)',
                    is_css = True,
                    keys = trade_params.stop_order_price
                )

    async def _get_duration_params(self, trade_params: TradeParameters) -> None:
        """
        期間のパラメータを入力する。
        """
        await self.browser_utils.click_element(trade_params.period_type)
        if trade_params.period_type == "期間指定":
            if trade_params.period_value is None and trade_params.period_index is None:
                raise ValueError("期間を指定してください。period_value or period_index")
            period_option_div = \
                await self.browser_utils.select_element('select[name="limit_in"]', is_css = True)

            if trade_params.period_value is not None:
                options = await period_option_div.select('option')
                period_value_list = [await option.get_attribute('value') for option in options]
                if trade_params.period_value not in period_value_list:
                    raise ValueError("period_valueが存在しません")
                else:
                    selector = f'option[value="{trade_params.period_value}"]'
                    await self.browser_utils.select_pulldown(selector)
            if trade_params.period_index is not None:
                period_options = await period_option_div.select('option')
                if trade_params.period_index < len(period_options):
                    await period_options[trade_params.period_index].click()
                else:
                    raise ValueError("指定したインデックスが範囲外")

    async def _select_deposit_and_credit_type(self, trade_params: TradeParameters) -> None:
        """
        預り区分と信用取引区分を選択する。
        """
        await self.browser_utils.click_element(trade_params.trade_section)
        await self.browser_utils.click_element(trade_params.margin_trade_section)

    async def _input_trade_pass(self):
        '''取引パスワードを入力する。'''
        await self.browser_utils.send_keys_to_element('input[id="pwd3"]',
                                                      is_css = True,
                                                      keys = os.getenv('SBI_TRADEPASS'))

    async def _confirm_order(self, trade_params: TradeParameters) -> bool:
        '''注文を確定する。'''
        await self._input_trade_pass()
        await self.browser_utils.click_element('input[id="shouryaku"]', is_css = True)
        await self.browser_utils.click_element('img[title="注文発注"]', is_css = True)

        text_to_show = f'{trade_params.symbol_code} {trade_params.trade_type} {trade_params.unit}株:'
        order_index = self._append_trade_params_to_orders(trade_params)
        success_text = "ご注文を受け付けました。"
        error_selector = '#MAINAREA02_780 > form > table:nth-child(22) > tbody > tr > td > b > p'

        max_retry = 10
        for _ in range(max_retry):
            try: 
                await self.browser_utils.wait_for(success_text, timeout=1)
                print(f"{text_to_show} 注文が成功しました")
                await self._edit_position_manager_for_order(order_index)
                await self.browser_utils.close_popup()
                return True
            except TimeoutError:
                pass
            except Exception as e:
                print(f"エラーが発生しました: {e}")

            try:
                await self.browser_utils.wait_for(error_selector, is_css=True, timeout=1)
                print(f"{text_to_show} 注文が失敗しました")       
                self.error_tickers.append(trade_params.symbol_code)
                return False      
            except TimeoutError:
                pass
            except Exception as e:
                print(f"エラーが発生しました: {e}")
        print(f"{text_to_show} 注文が失敗しました")       
        self.error_tickers.append(trade_params.symbol_code)
        return False                

    async def _edit_position_manager_for_order(self, order_index: int) -> None:
            order_id = await self._get_element('注文番号') 
            self.position_manager.update_order_id(index = order_index, order_id = order_id)
            self.position_manager.update_status(order_id, status_type = 'order_status', new_status = self.position_manager.STATUS_ORDERED)

    def _append_trade_params_to_orders(self, trade_params: TradeParameters):
        '''発注情報を登録'''
        order_index = self.position_manager.find_unordered_position_by_params(trade_params)
        if order_index is None:
            self.position_manager.add_position(trade_params)
            order_index = len(self.position_manager.positions) - 1
        return order_index

    async def cancel_all_orders(self) -> None:
        """
        すべての注文をキャンセルする。
        """
        try:
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

    async def extract_order_list(self) -> None:
        """
        現在の注文一覧を取得する。
        """
        try:
            await self.page_navigator.order_inquiry()
            await self.browser_utils.wait_for('未約定注文一覧')
            await self.browser_utils.wait(2)
            html_content = await self.browser_utils.get_html_content()
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
                    data = self._append_order_to_list(tr, data)

            columns = [
                "注文番号", "注文状況", "注文種別", "銘柄", "コード", "取引", "預り", "手数料", "注文日",
                "注文期間", "注文株数", "（未約定）", "執行条件", "注文単価", "現在値", "条件"
            ]
            self.order_list_df = pd.DataFrame(data, columns=columns)
            self.order_list_df["注文番号"] = self.order_list_df["注文番号"].astype(int)
            self.order_list_df["コード"] = self.order_list_df["コード"].astype(str)
            self.order_list_df = self.order_list_df[self.order_list_df["注文状況"] != "取消中"].reset_index(drop=True)
            print('キャンセル対象注文')
            print(self.order_list_df)
        except Exception as e:
            print(f"注文リストの取得中にエラーが発生しました: {e}")
            

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


    async def _cancel_single_order(self) -> None:
        await self.browser_utils.wait_for('取引パスワード')
        await self._input_trade_pass()
        await self.browser_utils.click_element('input[value=注文取消]', is_css = True)
        await self.browser_utils.wait_for('ご注文を受け付けました。')
        await self.browser_utils.wait(1)
        await self._handle_cancel_response()

    async def _handle_cancel_response(self) -> None:
        html_content = await self.browser_utils.get_html_content()
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

    async def _get_element(self, text: str):
        element = await self.browser_utils.select_element(text)
        element = element.parent.parent.children[1]
        return re.sub(r'\s+', '', element.text)

    async def _edit_position_manager_for_cancel(self) -> None:
        html_content = await self.browser_utils.get_html_content()
        html = soup(html_content, "html.parser")
        order_id_element = html.find("b", string=re.compile("注文番号"))
        order_id = order_id_element.find_parent("th").find_next_sibling("td").get_text(strip=True)
        order_type_element = html.find("b", string=re.compile("取引"))
        order_type = order_type_element.find_parent("th").find_next_sibling("td").get_text(strip=True)

        if any(order in order_type for order in ["信用新規買", "信用新規売", "現物買"]):
            status_type_to_update = 'order_status'
        if any(order in order_type for order in ["信用返済買", "信用返済売", "現物売"]):
            status_type_to_update = 'settlement_status'
        for order in self.position_manager.positions:
            if (order['order_id'] == order_id):
                self.position_manager.update_status(order_id, status_type = status_type_to_update, new_status = self.position_manager.STATUS_UNORDERED)
                if status_type_to_update == 'order_status':
                    self.position_manager.remove_waiting_order(order_id)

    async def reorder_pending(self) -> None:
        """
        現在のポジションと市場条件に基づいて発注待ちの注文を再発注する。
        """
        await self.page_navigator.trade()
        pending_positions = self.position_manager.get_pending_positions()
        for position in pending_positions:
            trade_params = TradeParameters(**position['order_params'])
            await self.place_new_order(trade_params)


    async def settle_all_margins(self):
        await self._extract_margin_list()
        await self.extract_order_list()

        if len(self.margin_list_df) == 0:
            print('信用建玉がありません。決済処理を中断します。')
            return
        margin_tickers = self.margin_list_df.sort_values(by="証券コード")["証券コード"].unique().tolist()
        ordered_tickers = self.order_list_df.sort_values(by="コード")["コード"].unique().tolist() if len(self.order_list_df) > 0 else []
        if sorted(margin_tickers) == sorted(ordered_tickers):
            print('すべての信用建玉の決済注文を発注済みです。')
            return
        print(f'保有建玉：{len(margin_tickers)}件')
        print(f'発注済み：{len(ordered_tickers)}件')
        retry_count = 0
        await self.browser_utils.wait_and_click('img[title=取引]', is_css = True)
        await self.browser_utils.wait_and_click('信用返済')
        # '返買', '返売' が存在するCSSセレクタの位置から、要素の番号を特定
        table_body_css = '#MAINAREA02_780 > form > table:nth-child(18) > tbody > tr > td > \
            table > tbody > tr > td > table > tbody'
        
        css_element_nums = await self._get_css_element_nums(table_body_css)

        for ticker, element_num in zip(margin_tickers, css_element_nums):
            if ticker in ordered_tickers:
                print(f'{ticker}はすでに決済発注済です。')
                continue
            else:
                await self.browser_utils.wait_and_click('img[title=取引]', is_css = True)
                await self.browser_utils.wait_and_click('信用返済')
                await self.browser_utils.wait_and_click(
                    f'{table_body_css} > tr:nth-child({element_num}) > td:nth-child(10) > a:nth-child(1) > u > font', 
                    is_css = True
                        )
                await self.browser_utils.wait_and_click('input[value="全株指定"]', is_css = True)
                await self.browser_utils.wait_and_click('input[value="注文入力へ"]', is_css = True)
                await self.browser_utils.wait(2)
                order_type_elements = await self.browser_utils.select_all('input[name="in_sasinari_kbn"]')
                await order_type_elements[1].click()  # 成行
                selector = f'select[name="nariyuki_condition"] option[value="H"]'
                await self.browser_utils.select_pulldown(selector)

                await self.browser_utils.send_keys_to_element('input[id="pwd3"]', is_css = True, keys = os.getenv('SBI_TRADEPASS'))
                await self.browser_utils.wait_and_click('input[id="shouryaku"]', is_css = True)
                await self.browser_utils.wait_and_click('img[title="注文発注"]', is_css =  True)
                await self.browser_utils.wait(2)
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

                retry_count = 0
        print(f'全銘柄の決済処理が完了しました。')


    async def _extract_margin_list(self):
        await self.page_navigator.credit_position()
        await self.browser_utils.wait(1)
        html_content = await self.browser_utils.get_html_content()
        self.margin_list_df = self._parse_margin_table(html_content)
    
    async def _get_css_element_nums(self, table_body_css: str) -> list[int]:
        await self.browser_utils.wait(5)
        mylist = await self.browser_utils.query_selector(f'{table_body_css} > tr', is_all=True)
        css_element_nums = []
        for num, element in enumerate(mylist):
            element_text = element.text_all
            if '返買' in element_text or '返売' in element_text:
                css_element_nums.append(num + 1)
        return css_element_nums

    async def _navigate_to_margin_page(self):
        await self.page_navigator.credit_position()

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


    async def handle_order_errors(self) -> None:
        """
        注文エラーを処理し、必要に応じて再試行する。
        """
        await self.page_navigator.trade()

        error_positions = self.position_manager.get_error_positions()
        for position in error_positions:
            trade_params = TradeParameters(**position['order_params'])
            await self.place_new_order(trade_params)