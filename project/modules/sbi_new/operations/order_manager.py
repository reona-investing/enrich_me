from .position_manager import PositionManager
from .trade_possibility_manager import TradePossibilityManager
from .trade_parameters import TradeParameters
from ..browser.browser_utils import BrowserUtils
from ..session.login_handler import LoginHandler
import os
import re
import unicodedata
import pandas as pd
from bs4 import BeautifulSoup as soup
import traceback

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
        self.login_handler: LoginHandler = login_handler
        self.position_manager: PositionManager = PositionManager()
        self.trade_possibility_manager: TradePossibilityManager = TradePossibilityManager(login_handler)
        self.browser_utils: BrowserUtils = BrowserUtils()
        self.has_successfully_ordered: bool = False
        self.error_tickers: list = []

    async def place_new_order(self, trade_params: TradeParameters) -> None:
        """
        指定された取引パラメータを使用して新規注文を行う。
        """
        await self.login_handler.sign_in()
        self.tab = self.login_handler.session.tab  # ��縮参照を設定
        try:
            await self._navigate_to_trade_page()
            await self._select_trade_type(trade_params)
            await self._input_stock_and_quantity(trade_params)
            await self._input_sashinari_params(trade_params)
            await self._get_duration_params(trade_params)
            await self._select_deposit_and_credit_type(trade_params)
            await self._confirm_order(trade_params)
            print("注文が正常に完了しました！")
            self.has_successfully_ordered = True
        except Exception as e:
            print(f"注文中にエラーが発生しました: {e}")
            traceback.print_exc()  # スタックトレースを出力
            self.error_tickers.append(trade_params.symbol_code)
        finally:
            self.login_handler.session.tab = self.tab

    async def _navigate_to_trade_page(self) -> None:
        """
        取引ページに遷移する。
        """
        trade_button = await self.tab.wait_for('img[title="取引"]')
        await trade_button.click()
        await self.tab.wait(2)

    async def _select_trade_type(self, trade_params: TradeParameters) -> None:
        """
        取引タイプを選択する。
        """
        trade_type_button = await self.tab.wait_for(f'#{_get_selector(self.order_param_dicts, "取引", trade_params.trade_type)}')
        await trade_type_button.click()

    async def _input_stock_and_quantity(self, trade_params: TradeParameters) -> None:
        """
        銘柄コードと数量を入力する。
        """
        stock_code_input = await self.tab.select('input[name="stock_sec_code"]')
        await stock_code_input.send_keys(trade_params.symbol_code)

        quantity_input = await self.tab.select('input[name="input_quantity"]')
        await quantity_input.send_keys(str(trade_params.unit))

    async def _input_sashinari_params(self, trade_params: TradeParameters) -> None:
        button = await self.tab.find(trade_params.order_type)
        await button.click()

        if trade_params.order_type == '成行':
            if trade_params.order_type_value is not None:
                selector = f'select[name="nariyuki_condition"] option[value="{self.order_param_dicts["成行タイプ"][trade_params.order_type_value]}"]'
                await self.browser_utils.select_pulldown(self.tab, selector)

        if trade_params.order_type == "指値":
            form = await self.tab.select('#gsn0 > input[type=text]')
            await form.send_keys(trade_params.limit_order_price)

            if trade_params.order_type_value is not None:
                selector = f'select[name="sasine_condition"] option[value="{self.order_param_dicts["指値タイプ"][trade_params.order_type_value]}"]'
                await self.browser_utils.select_pulldown(self.tab, selector)

        if trade_params.order_type == "逆指値":
            choice = await self.tab.select('#gsn2 > table > tbody > tr > td:nth-child(2) > label:nth-child(5) > input[type=radio]')
            await choice.click()

            await self.tab.select('#gsn2 > table > tbody > tr > td:nth-child(2) > select:nth-child(6)')
            choice = await self.tab.select(
                f'#gsn2 > table > tbody > tr > td:nth-child(2) > select:nth-child(6) > option:nth-child({self.order_param_dicts["逆指値タイプ"][trade_params.stop_order_type]})'
            )
            await choice.send_keys(trade_params.stop_order_trigger_price)

            if trade_params.stop_order_type == "指値":
                form = await self.tab.select('#gsn2 > table > tbody > tr > td:nth-child(2) > input[type=text]:nth-child(7)')
                await form.send_keys(trade_params.stop_order_price)

    async def _get_duration_params(self, trade_params: TradeParameters) -> None:
        """
        期間のパラメータを入力する。
        """
        button = await self.tab.find(trade_params.period_type)
        await button.click()
        if trade_params.period_type == "期間指定":
            if trade_params.period_value is None and trade_params.period_index is None:
                raise ValueError("期間を指定してください。period_value or period_index")
            period_option_div = await self.tab.select('select[name="limit_in"]')

            if trade_params.period_value is not None:
                options = await period_option_div.select('option')
                period_value_list = [await option.get_attribute('value') for option in options]
                if trade_params.period_value not in period_value_list:
                    raise ValueError("period_valueが存在しません")
                else:
                    selector = f'option[value="{trade_params.period_value}"]'
                    await self.browser_utils.select_pulldown(self.tab, selector)
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
        deposit_type_button = await self.tab.find(trade_params.trade_section)
        await deposit_type_button.click()

        credit_trade_type_button = await self.tab.find(trade_params.margin_trade_section)
        await credit_trade_type_button.click()

    async def _input_trade_pass(self):
        '''取引パスワードを入力する。'''
        trade_password_input = await self.tab.select('input[id="pwd3"]')
        await trade_password_input.send_keys(os.getenv('SBI_TRADEPASS'))

    async def _confirm_order(self, trade_params: TradeParameters) -> None:
        '''注文を確定する。'''
        await self._input_trade_pass()
        skip_button = await self.tab.select('input[id="shouryaku"]')
        await skip_button.click()
        order_button = await self.tab.select('img[title="注文発注"]')
        await order_button.click()
        await self.tab.wait(1)

        order_index = self._append_trade_params_to_orders(trade_params)
        html_content = await self.tab.get_content()
        if "ご注文を受け付けました。" in html_content:
            print(f"注文が成功しました: {trade_params.symbol_code}")
            await self._edit_position_manager_for_order(order_index)
        else:
            print(f"注文が失敗しました: {trade_params.symbol_code}")
            self.error_tickers.append(trade_params.symbol_code)

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
        await self.login_handler.sign_in()
        self.tab = self.login_handler.session.tab
        try:
            await self._extract_order_list()

            if len(self.order_list_df) == 0:
                print("キャンセルする注文はありません。")
                return

            for i in range(len(self.order_list_df)):
                await self._navigate_to_cancel_page()
                await self._cancel_single_order(i)
        except Exception as e:
            print(f"注文キャンセル中にエラーが発生しました: {e}")
            traceback.print_exc()  # スタックトレースを出力
        finally:
            self.login_handler.session.tab = self.tab

    async def _extract_order_list(self) -> None:
        """
        現在の注文一覧を取得する。
        """
        try:
            await self._navigate_to_trade_page()
            button = await self.tab.wait_for(text='注文照会')
            await button.click()
            await self.tab.wait(3)

            html_content = await self.tab.get_content()
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

    async def _navigate_to_cancel_page(self) -> None:
        await self._navigate_to_trade_page()
        button = await self.tab.find('注文照会')
        await button.click()
        await self.tab.wait(1)
        button = await self.tab.find('取消')
        await button.click()
        await self.tab.wait(1)

    async def _cancel_single_order(self, index: int) -> None:
        await self._input_trade_pass()
        button = await self.tab.select('input[value=注文取消]')
        await button.click()
        await self.tab.wait(1)
        await self._handle_cancel_response(index)

    async def _handle_cancel_response(self, index: int) -> None:
        html_content = await self.tab.get_content()

        code = await self._get_element("銘柄コード")
        code = str(code)
        unit = await self._get_element("株数")
        unit = int(str(unit)[:-1])
        order_type = await self._get_element("取引")

        if "ご注文を受け付けました。" in html_content:
            print(f"{code} {unit}株 {order_type}：注文取消が完了しました。")
            await self._edit_position_manager_for_cancel()
        else:
            print(f"{code} {unit}株 {order_type}：注文取消に失敗しました。")

    async def _get_element(self, text: str):
        element = await self.tab.find(text)
        element = element.parent.parent.children[1]
        return re.sub(r'\s+', '', element.text)

    async def _edit_position_manager_for_cancel(self) -> None:
        order_id = await self._get_element("注文番号")
        order_type = await self._get_element("取引")
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
        await self.login_handler.sign_in()
        self.tab = self.login_handler.session.tab
        await self._navigate_to_trade_page()

        pending_positions = self.position_manager.get_pending_positions()
        for position in pending_positions:
            trade_params = TradeParameters(**position['order_params'])
            await self.place_new_order(trade_params)


    async def settle_all_margins(self):
        await self.login_handler.sign_in()
        self.tab = self.login_handler.session.tab
        await self._extract_margin_list()
        await self._extract_order_list()

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
        i = n = 0

        while i < len(margin_tickers):
            if margin_tickers[i] in ordered_tickers:
                print(f'{margin_tickers[i]}はすでに決済発注済です。')
                i += 1
                continue
            else:
                button = await self.tab.wait_for('img[title=取引]')
                await button.click()
                button = await self.tab.wait_for(text='信用返済')
                await button.click()
                for _ in range(10):
                    try:
                        position_link = await self.tab.wait_for(
                            f'#MAINAREA02_780 > form > table:nth-child(18) > tbody > tr > td > \
                            table > tbody > tr > td > table > tbody > tr:nth-child({i +  n + 1}) > \
                            td:nth-child(10) > a:nth-child(1) > u > font', 
                            timeout = 5
                        )
                        await position_link.click()
                        break
                    except:
                        n += 1
                all_shares_button = await self.tab.wait_for('input[value="全株指定"]')
                await all_shares_button.click()
                order_input_button = await self.tab.wait_for('input[value="注文入力へ"]')
                await order_input_button.click()
                await self.tab.wait(1)
                order_type_elements = await self.tab.select_all('input[name="in_sasinari_kbn"]')
                await order_type_elements[1].click()  # 成行
                selector = f'select[name="nariyuki_condition"] option[value="H"]'
                await self.browser_utils.select_pulldown(self.tab, selector)
                trade_password_input = await self.tab.wait_for('input[id="pwd3"]')
                await trade_password_input.send_keys(os.getenv('SBI_TRADEPASS'))
                skip_button = await self.tab.wait_for('input[id="shouryaku"]')
                await skip_button.click()
                order_button = await self.tab.wait_for('img[title="注文発注"]')
                await order_button.click()
                await self.tab.wait(1.2)
                try:
                    await self.tab.wait_for(text='ご注文を受け付けました。')
                    print(f"{margin_tickers[i]}：正常に決済注文完了しました。")


                    html_content = await self.tab.get_content()
                    symbol_code = str(await self._get_element('銘柄コード'))
                    extracted_unit = await self._get_element('株数')
                    extracted_unit = int(extracted_unit[:-2])
                    trade_type = await self._get_element('取引')
                    if '信用返済買' in trade_type:
                        trade_type = '信用新規売'
                    if '信用返済売' in trade_type:
                        trade_type = '信用新規買'
                    order_id = await self._get_element('注文番号')
                    print([symbol_code, extracted_unit, trade_type])
                    params_to_compare = TradeParameters(symbol_code=symbol_code, unit=extracted_unit, trade_type=trade_type)
                    order_id = self.position_manager.find_unordered_position_by_params(params_to_compare)
                    self.position_manager.update_order_id(i, order_id)
                    self.position_manager.update_status(order_id, status_type = 'settlement_order', new_status = self.position_manager.STATUS_ORDERED)

                    retry_count = 0
                    i += 1
                    
                except:
                    if retry_count < 3:
                        print(f"{margin_tickers[i]}：発注失敗。再度発注を試みます。")
                        retry_count += 1
                    else:
                        print(f"{margin_tickers[i]}：発注失敗。リトライ回数の上限に達しました。")
                        self.error_tickers.append(margin_tickers[i])
                        retry_count = 0
                        i += 1
            self.login_handler.session.tab = self.tab
            print(f'全銘柄の決済処理が完了しました。')


    async def _extract_margin_list(self):
        await self._navigate_to_margin_page()
        html_content = await self.tab.get_content()
        self.margin_list_df = self._parse_margin_table(html_content)

    async def _navigate_to_margin_page(self):
        button = await self.tab.wait_for('img[title=口座管理]')
        await button.click()
        button = await self.tab.wait_for('area[title=信用建玉]')
        await button.click()
        await self.tab.wait(3)

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
        await self.login_handler.sign_in()
        self.tab = self.login_handler.session.tab
        await self._navigate_to_trade_page()

        error_positions = self.position_manager.get_error_positions()
        for position in error_positions:
            trade_params = TradeParameters(**position['order_params'])
            await self.place_new_order(trade_params)