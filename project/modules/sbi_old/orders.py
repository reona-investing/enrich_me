# orders.py
import os
import unicodedata
import re
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup as soup
from .session import SBISession
from .utils.decorators import retry
from .utils.web import select_pulldown
from .positions import TradeParameters, SBIOrderManager
import paths
import shutil

class SBIOrderMaker:
    order_param_dicts = {
                        '取引':{
                        "現物買": "genK",
                        "現物売": "genU",
                        "信用新規買": "shinK",
                        "信用新規売": "shinU",
                        },
                        '注文タイプ':{
                        "指値": 0,
                        "成行": 1,
                        "逆指値": 2,
                        },
                        '指値タイプ':{
                        "寄指":'Z',
                        "引指":'I',
                        "不成":'F',
                        "IOC指":'P'
                        },
                        '成行タイプ':{
                        "寄成":'Y',
                        "引成":'H',
                        "IOC成":'I'
                        },
                        '逆指値タイプ':{
                        "指値": 1,
                        "成行": 2,
                        },
                        '期間':{
                        "当日中": 0,
                        "今週中": 1,
                        "期間指定": 2
                        },
                        '預り区分':{
                        "一般預り": 0,
                        "特定預り": 1,
                        "NISA預り": 2
                        },
                        '信用取引区分':{
                        "制度": 0,
                        "一般": 1,
                        "日計り": 2
                        }
                        }
    def __init__(self, session: SBISession):
        self.session = session
        self.daily_order_manager = SBIOrderManager()
        self.has_successfully_ordered = False
        self.error_tickers = []
        self.margin_list_df = pd.DataFrame()
        self.order_list_df = pd.DataFrame()
        

    @retry()
    async def make_order(self, trade_params: TradeParameters):
        await self.session.sign_in()
        trade_button = await self.session.tab.wait_for('img[title="取引"]')
        await trade_button.click()
        await self.session.tab.wait(2)
        trade_type_button = await self.session.tab.wait_for(f'#{SBIOrderMaker.order_param_dicts["取引"][trade_params.trade_type]}')
        await trade_type_button.click()
        stock_code_input = await self.session.tab.select('input[name="stock_sec_code"]')
        await stock_code_input.send_keys(trade_params.symbol_code)
        quantity_input = await self.session.tab.select('input[name="input_quantity"]')
        await quantity_input.send_keys(str(trade_params.unit))
        await self._input_sashinari_params(trade_params)
        await self._get_duration_params(trade_params)

        deposit_type_button = await self.session.tab.find(trade_params.trade_section)
        await deposit_type_button.click()
        credit_trade_type_button = await self.session.tab.find(trade_params.margin_trade_section)
        await credit_trade_type_button.click()
        trade_password_input = await self.session.tab.select('input[id="pwd3"]')
        await trade_password_input.send_keys(os.getenv('SBI_TRADEPASS'))

        skip_button = await self.session.tab.select('input[id="shouryaku"]')
        await skip_button.click()
        order_button = await self.session.tab.select('img[title="注文発注"]')
        await order_button.click()
        await self.session.tab.wait(1)

        html_content = await self.session.tab.get_content()

        # 発注情報を登録
        order_num = self.daily_order_manager.search_orders(trade_params)
        if order_num is None:
            order_num = self.daily_order_manager.add_new_order(trade_params)
        # TODO jsonに格納した時点でNoneがnullになり、search_ordersがうまく働かない。

        if "ご注文を受け付けました。" in html_content:
            print(f"{trade_params.symbol_code} {trade_params.unit}株 {trade_params.trade_type} {trade_params.order_type}：正常に注文完了しました。")
            self.has_successfully_ordered = True       
            order_id = self._get_element_from_order_info(html_content, search_header = "注文番号")
            self.daily_order_manager.update_order_id(order_num, order_id)
            self.daily_order_manager.update_status(order_id, "発注済")
        else:
            print(f"{trade_params.symbol_code} {trade_params.unit}株 {trade_params.trade_type} {trade_params.order_type}：発注できませんでした。")
            await self.session.tab.save_screenshot(f'{paths.DEBUG_FILES_FOLDER}/{trade_params.symbol_code}_error.png')
            shutil.copy(f'{paths.DEBUG_FILES_FOLDER}/{trade_params.symbol_code}_error.png', f'{paths.ONLINE_BACKUP_FOLDER}/{trade_params.symbol_code}_error.png')
            self.has_successfully_ordered = False

    async def _input_sashinari_params(self, trade_params: TradeParameters):
        button = await self.session.tab.find(trade_params.order_type)
        await button.click()
        if trade_params.order_type == '成行':
            if trade_params.order_type_value is not None:
                selector = f'select[name="nariyuki_condition"] option[value="{SBIOrderMaker.order_param_dicts["成行タイプ"][trade_params.order_type_value]}"]'
                await select_pulldown(self.session.tab, selector)

        if trade_params.order_type == "指値":
            if trade_params.limit_order_price is None:
                raise ValueError("指値価格を設定してください。limit_order_price")
            form = await self.session.tab.select('#gsn0 > input[type=text]')
            await form.send_keys(trade_params.limit_order_price)
            
            if trade_params.order_type_value is not None:
                selector = f'select[name="sasine_condition"] option[value="{SBIOrderMaker.order_param_dicts["指値タイプ"][trade_params.order_type_value]}"]'
                await select_pulldown(self.session.tab, selector)

        if trade_params.order_type == "逆指値":
            choice = await self.session.tab.select('#gsn2 > table > tbody > tr > td:nth-child(2) > label:nth-child(5) > input[type=radio]')
            await choice.click()
            if trade_params.stop_order_trigger_price is None:
                raise ValueError("逆指値のトリガー価格を設定してください。stop_order_trigger_price")
            await self.session.tab.select('#gsn2 > table > tbody > tr > td:nth-child(2) > select:nth-child(6)')
            choice = await self.session.tab.select(
                f'#gsn2 > table > tbody > tr > td:nth-child(2) > select:nth-child(6) > option:nth-child({SBIOrderMaker.order_param_dicts["逆指値タイプ"][trade_params.stop_order_type]})'
            )
            await choice.send_keys(trade_params.stop_order_trigger_price)
            if trade_params.stop_order_type == "指値":
                if trade_params.stop_order_price is None:
                    raise ValueError("逆指値価格を設定してください。stop_order_type")
                form = await self.session.tab.select('#gsn2 > table > tbody > tr > td:nth-child(2) > input[type=text]:nth-child(7)')
                await form.send_keys(trade_params.stop_order_price)

    async def _get_duration_params(self, trade_params: TradeParameters):
        button = await self.session.tab.find(trade_params.period_type)
        await button.click()
        if trade_params.period_type=="期間指定":
            if trade_params.period_value is None and trade_params.period_index is None:
                raise ValueError("期間を指定してください。period_value or period_index")
            period_option_div = await self.session.tab.select('select[name="limit_in"]')
            
            if trade_params.period_value is not None:
                options = await period_option_div.select('option')
                period_value_list = [await option.get_attribute('value') for option in options]
                if trade_params.period_value not in period_value_list:
                    raise ValueError("period_valueが存在しません")
                else:
                    selector = f'option[value="{trade_params.period_value}"]'
                    await select_pulldown(self.session.tab, selector)
            if trade_params.period_index is not None:
                period_options = await period_option_div.select('option')
                if trade_params.period_index < len(period_options):
                    await options[trade_params.period_index].click()
                else:
                    raise ValueError("指定したインデックスが範囲外")

    def _get_element_from_order_info(self, html_content, search_header: str) -> int:
        # BeautifulSoupでHTMLを解析
        bs = soup(html_content, 'html.parser')
        # "注文番号"というテキストを持つ要素を探す
        target_element = bs.find('th', class_='vaM', text=lambda t: search_header in t)
        # 対応する兄弟要素 <td> を取得
        if target_element:
            value_element = target_element.find_next('td', class_='vaM')
            if value_element:
                value = value_element.text.strip()
                return str(value)
            else:
                raise ValueError("対応する<td>要素が見つかりません。")
        else:
            raise ValueError("注文番号が見つかりません。")


    @retry()
    async def settle_all_margins(self):
        await self.session.sign_in()
        await self._extract_margin_list()
        await self.extract_order_list()

        if len(self.margin_list_df) == 0:
            print('信用建玉がありません。決済処理を中断します。')
            return
        else:
            margin_tickers = self.margin_list_df.sort_values(by="証券コード")['証券コード'].unique().tolist()
        if len(self.order_list_df) > 0:
            ordered_tickers = self.order_list_df.sort_values(by="コード")['コード'].unique().tolist()
            if sorted(margin_tickers) == sorted(ordered_tickers):
                print('すべての信用建玉の決済注文を発注済みです。')
        else:
            ordered_tickers = []

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
                button = await self.session.tab.wait_for('img[title=取引]')
                await button.click()
                button = await self.session.tab.wait_for(text='信用返済')
                await button.click()
                for _ in range(10):
                    try:
                        position_link = await self.session.tab.wait_for(
                            f'#MAINAREA02_780 > form > table:nth-child(18) > tbody > tr > td > \
                            table > tbody > tr > td > table > tbody > tr:nth-child({i +  n + 1}) > \
                            td:nth-child(10) > a:nth-child(1) > u > font', 
                            timeout = 5
                        )
                        await position_link.click()
                        break
                    except:
                        n += 1
                all_shares_button = await self.session.tab.wait_for('input[value="全株指定"]')
                await all_shares_button.click()
                order_input_button = await self.session.tab.wait_for('input[value="注文入力へ"]')
                await order_input_button.click()
                await self.session.tab.wait(1)
                order_type_elements = await self.session.tab.select_all('input[name="in_sasinari_kbn"]')
                await order_type_elements[1].click()  # 成行
                selector = f'select[name="nariyuki_condition"] option[value="H"]'
                await select_pulldown(self.session.tab, selector)
                trade_password_input = await self.session.tab.wait_for('input[id="pwd3"]')
                await trade_password_input.send_keys(os.getenv('SBI_TRADEPASS'))
                skip_button = await self.session.tab.wait_for('input[id="shouryaku"]')
                await skip_button.click()
                order_button = await self.session.tab.wait_for('img[title="注文発注"]')
                await order_button.click()
                await self.session.tab.wait(1.2)
                try:
                    await self.session.tab.wait_for(text='ご注文を受け付けました。')
                    print(f"{margin_tickers[i]}：正常に決済注文完了しました。")

                    retry_count = 0
                    i += 1

                    html_content = await self.session.tab.get_content()
                    symbol_code = str(self._get_element_from_order_info(html_content, '銘柄コード'))
                    extracted_unit = self._get_element_from_order_info(html_content, ' 株数')
                    extracted_unit = int(extracted_unit[:-2])
                    trade_type = self._get_element_from_order_info(html_content, '取引')
                    if '信用返済買' in trade_type:
                        trade_type = '信用新規売'
                    if '信用返済売' in trade_type:
                        trade_type = '信用新規買'
                    order_id = self._get_element_from_order_info(html_content, '注文番号')
                    print([symbol_code, extracted_unit, trade_type])
                    params_to_compare = TradeParameters(symbol_code=symbol_code, unit=extracted_unit, trade_type=trade_type)
                    order_id = self.daily_order_manager.search_orders(params_to_compare)
                    self.daily_order_manager.update_status(order_id, "決済発注済")
                    # TODO order_idのアップデートを実装する！
                    
                except:
                    if retry_count < 3:
                        print(f"{margin_tickers[i]}：発注失敗。再度発注を試みます。")
                        retry_count += 1
                    else:
                        print(f"{margin_tickers[i]}：発注失敗。リトライ回数の上限に達しました。")
                        self.error_tickers.append(margin_tickers[i])
                        retry_count = 0
                        i += 1
        print(f'全銘柄の決済処理が完了しました。')

    async def _extract_margin_list(self):
        button = await self.session.tab.wait_for('img[title=口座管理]')
        await button.click()
        button = await self.session.tab.wait_for('area[title=信用建玉]')
        await button.click()
        await self.session.tab.wait(3)
        html_content = await self.session.tab.get_content()
        html = soup(html_content, "html.parser")
        table = html.find("td", string=re.compile("銘柄"))
        if table is None:
            print('保有建玉はありません。')
            return
        table = table.findParent("table")

        data = []
        for tr in table.find("tbody").findAll("tr"):
            if tr.find("td").find("a"):
                data = self._append_margin_to_list(tr, data)
        columns = ["証券コード", "銘柄", "売・買建", "建株数", "建単価", "現在値"]
        self.margin_list_df = pd.DataFrame(data, columns=columns)
        self.margin_list_df["証券コード"] = self.margin_list_df["証券コード"].astype(str)
        self.margin_list_df["建株数"] = self.margin_list_df["建株数"].str.replace(',', '').astype(int)
        self.margin_list_df["建単価"] = self.margin_list_df["建単価"].str.replace(',', '').astype(float)
        self.margin_list_df["現在値"] = self.margin_list_df["現在値"].str.replace(',', '').astype(float)
        self.margin_list_df["建価格"] = self.margin_list_df["建株数"] * self.margin_list_df["建単価"]
        self.margin_list_df["評価額"] = self.margin_list_df["建株数"] * self.margin_list_df["現在値"]
        self.margin_list_df['評価損益'] = self.margin_list_df["評価額"] - self.margin_list_df["建価格"]
        self.margin_list_df.loc[self.margin_list_df['売・買建'] == '売建', '評価損益'] = \
            self.margin_list_df["建価格"] - self.margin_list_df["評価額"]

    async def extract_order_list(self):
        trade_button = await self.session.tab.wait_for('img[title="取引"]')
        await trade_button.click()
        button = await self.session.tab.wait_for(text='注文照会')
        await button.click()
        await self.session.tab.wait(3)
        html_content = await self.session.tab.get_content()
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

        columns = ["注文番号", "注文状況", "注文種別", "銘柄", "コード", "取引", "預り", "手数料", "注文日",
                "注文期間", "注文株数", "（未約定）", "執行条件", "注文単価", "現在値", "条件"]
        self.order_list_df = pd.DataFrame(data, columns=columns)
        self.order_list_df["注文番号"] = self.order_list_df["注文番号"].astype(int)
        self.order_list_df["コード"] = self.order_list_df["コード"].astype(str)
        self.order_list_df = self.order_list_df[self.order_list_df["注文状況"]!="取消中"].reset_index(drop=True)

    @retry()
    async def cancel_all_orders(self):
        await self.session.sign_in()
        await self.extract_order_list()
        if len(self.order_list_df) == 0:
            return
        for i in range(len(self.order_list_df)):
            button = await self.session.tab.wait_for('img[title=取引]')
            await button.click()
            await self.session.tab.wait(1)
            button = await self.session.tab.find('注文照会')
            await button.click()
            await self.session.tab.wait(1)
            button = await self.session.tab.find('取消')
            await button.click()
            await self.session.tab.wait(1)
            input = await self.session.tab.select('input[id="pwd3"]')
            await input.send_keys(os.getenv('SBI_TRADEPASS'))
            button = await self.session.tab.select('input[value=注文取消]')
            await button.click()
            await self.session.tab.wait(1)
            html_content = await self.session.tab.get_content()

            code = self.order_list_df['コード'].iloc[i]
            unit = self.order_list_df['注文株数'].iloc[i]
            order_type = self.order_list_df['注文種別'].iloc[i]
            if "ご注文を受け付けました。" in html_content:
                print(f"{code} {unit}株 {order_type}：注文取消が完了しました。")
                order_id = self._get_element_from_order_info(html_content, "注文番号")
                if any(keyword in html_content for keyword in ["信用新規買", "信用新規売", "現物買"]):
                    for order in self.daily_order_manager.orders:
                        if (order['order_id'] == order_id) & (order['status'] == '発注済'):                            
                            self.daily_order_manager.update_status(order_id, "発注待ち")
                            self.daily_order_manager.remove_waiting_order(order_id)
                if any(keyword in html_content for keyword in ["信用返済買", "信用返済売", "現物売"]):
                    for order in self.daily_order_manager.orders:
                        if (order['order_id'] == order_id) & (order['status'] == '決済発注済'):  
                            self.daily_order_manager.update_status(order_id, "新規約定済")
            else:
                print(f"{code} {unit}株 {order_type}：注文取消に失敗しました。")

    def _append_order_to_list(self, tr:object, data:list) -> list:
        row = []
        row.append(tr.findAll("td")[0].getText().strip()) # 注文番号
        row.append(tr.findAll("td")[1].getText().strip()) # 注文状況
        row.append(tr.findAll("td")[2].getText().strip()) # 注文種別
        text = unicodedata.normalize("NFKC", tr.findAll("td")[3].getText().strip())
        row.append(text.splitlines()[0].strip().split(" ")[0])
        row.append(text.splitlines()[0].strip().split(" ")[-1])

        tmp_data = []
        for t in tr.findNext("tr").findAll("td")[0].getText().strip().replace(" ", "").splitlines():
            if t!="" and t!="/":
                tmp_data.append(t)
        if len(tmp_data)==4:
            row.extend([tmp_data[0]+tmp_data[1], tmp_data[2], tmp_data[3]])
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

# TODO OrderManagerとdfの二重管理を解消する！！