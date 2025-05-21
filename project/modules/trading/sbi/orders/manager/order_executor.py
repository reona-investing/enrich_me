import os
import pandas as pd
import re
import traceback
from typing import List, Optional
from bs4 import BeautifulSoup as soup

from trading.sbi.interface.orders import IOrderExecutor, OrderRequest, OrderResult
from trading.sbi.browser import SBIBrowserManager, PageNavigator
from utils.browser.named_tab import NamedTab


class SBIOrderExecutor(IOrderExecutor):
    """SBI証券での注文実行を管理するクラス"""
    
    # 注文パラメータ用の定数
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
        """コンストラクタ
        
        Args:
            browser_manager (SBIBrowserManager): SBI証券のブラウザセッションを管理するオブジェクト
        """
        self.browser_manager = browser_manager
        self.page_navigator = PageNavigator(browser_manager)
        self.order_list_df = pd.DataFrame()
        
    async def place_order(self, order_request: OrderRequest) -> OrderResult:
        """注文を発注する
        
        Args:
            order_request (OrderRequest): 注文リクエスト情報

        Returns:
            OrderResult: 注文結果
        """
        try:
            await self.browser_manager.launch()
            await self.page_navigator.trade()
            
            # 取引タイプを選択
            await self._select_trade_type(order_request)
            
            # 銘柄コードと数量を入力
            await self._input_stock_and_quantity(order_request)
            
            # 注文条件を設定（指値/成行/逆指値）
            await self._input_order_conditions(order_request)
            
            # 預り区分と信用取引区分を選択
            await self._select_deposit_and_credit_type(order_request)
            
            # 注文を確定
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
            return OrderResult(
                success=False,
                message=error_message,
                error_code="SYSTEM_ERROR"
            )
            
    async def cancel_all_orders(self) -> List[OrderResult]:
        """全注文をキャンセルする

        Returns:
            List[OrderResult]: キャンセル結果をまとめたリスト
        """
        try:
            order_list = []
            
            await self.browser_manager.launch()
            
            await self.page_navigator.order_inquiry()
            await self._extract_order_list()

            for i in range(len(self.order_list_df)):           
                await self.page_navigator.order_cancel()
                
                # 取引パスワードを入力してキャンセル処理を実行
                named_tab = self.browser_manager.get_tab('SBI')
                await named_tab.tab.utils.wait_for('取引パスワード')
                await self._input_trade_pass()
                await named_tab.tab.utils.click_element('input[value=注文取消]', is_css=True)
                
                # キャンセル結果を確認
                try:
                    await named_tab.tab.utils.wait_for('ご注文を受け付けました。')
                    html_content = await named_tab.tab.utils.get_html_content()
                    html = soup(html_content, "html.parser")
                    
                    # 銘柄情報を取得
                    code_element = html.find("b", string=re.compile("銘柄コード"))
                    code = code_element.find_parent("th").find_next_sibling("td").get_text(strip=True)
                    
                    # 株数を取得
                    unit_element = html.find("b", string=re.compile("株数"))
                    unit = unit_element.find_parent("th").find_next_sibling("td").get_text(strip=True)
                    unit = int(unit[:-1].replace(',', ''))
                    
                    # 取引タイプを取得
                    order_type_element = html.find("b", string=re.compile("取引"))
                    order_type = order_type_element.find_parent("th").find_next_sibling("td").get_text(strip=True)
                    
                    message = f"{code} {unit}株 {order_type}：注文キャンセルが完了しました"
                    print(message)
                    order_list.append(OrderResult(success=True, message=message))
                
                except Exception:
                    message="キャンセル処理中にエラーが発生しました"
                    print(message)
                    order_list.append(OrderResult(success=False, message=message, error_code="CANCEL_ERROR"))
                
        except Exception as e:
            error_message = f"注文キャンセル中にエラーが発生しました: {str(e)}"
            traceback.print_exc()
            order_list.append(
                OrderResult(
                    success=False,
                    message=error_message,
                    error_code="SYSTEM_ERROR"
                    )
                    )
        finally:
            return order_list
            
    async def settle_position(self, symbol_code: str, unit: Optional[int] = None) -> OrderResult:
        """ポジションを決済する
        
        Args:
            symbol_code (str): 決済対象の銘柄コード
            unit (Optional[int], optional): 決済する単位数（指定しない場合は全数量）

        Returns:
            OrderResult: 決済結果
        """
        try:
            await self.browser_manager.launch()
            
            # 信用建玉一覧ページに遷移
            named_tab = await self.page_navigator.credit_position_close()
            
            # 信用建玉一覧からシンボルコードに一致する行を見つける
            table_body_css = '#MAINAREA02_780 > form > table:nth-child(18) > tbody > tr > td > ' \
                           'table > tbody > tr > td > table > tbody'
            
            # 全建玉の要素番号リストを取得
            positions = await self._get_position_elements(table_body_css)
            
            # 指定された銘柄コードの建玉がない場合
            if not positions or symbol_code not in positions:
                return OrderResult(
                    success=False,
                    message=f"{symbol_code}: 該当する建玉が見つかりません",
                    error_code="POSITION_NOT_FOUND"
                )
            
            # 指定された銘柄の建玉の要素番号を取得
            element_num = positions[symbol_code]
            
            # 決済画面に遷移
            await self._navigate_to_position_settlement(element_num, table_body_css)
            
            # 数量選択（指定がなければ全株選択）
            if unit is None:
                await named_tab.tab.utils.click_element('input[value="全株指定"]', is_css=True)
            else:
                # 特定の株数を指定する場合の処理（実装が必要）
                await named_tab.tab.utils.send_keys_to_element('input[name="input_settlement_quantity"]',
                                                              is_css=True, 
                                                              keys=str(unit))
            
            # 注文入力画面に進む
            await named_tab.tab.utils.click_element('input[value="注文入力へ"]', is_css=True)
            
            # 成行注文を選択
            order_type_elements = await named_tab.tab.utils.select_all('input[name="in_sasinari_kbn"]')
            await order_type_elements[1].click()  # 成行
            
            # 引成（引け成行）を選択
            selector = f'select[name="nariyuki_condition"] option[value="H"]'
            await named_tab.tab.utils.select_pulldown(selector)
            
            # 注文を発注
            await self._input_trade_pass()
            await named_tab.tab.utils.click_element('input[id="shouryaku"]', is_css=True)
            await named_tab.tab.utils.wait_for('img[title="注文発注"]', is_css=True)
            await named_tab.tab.utils.click_element('img[title="注文発注"]', is_css=True)
            
            # 注文結果を確認
            try:
                await named_tab.tab.utils.wait_for('ご注文を受け付けました。')
                order_id = await self._get_element('注文番号')
                
                return OrderResult(
                    success=True,
                    order_id=order_id,
                    message=f"{symbol_code}：正常に決済注文が完了しました"
                )
            except Exception:
                return OrderResult(
                    success=False,
                    message=f"{symbol_code}：決済注文に失敗しました",
                    error_code="SETTLEMENT_ERROR"
                )
                
        except Exception as e:
            error_message = f"ポジション決済中にエラーが発生しました: {str(e)}"
            traceback.print_exc()
            return OrderResult(
                success=False,
                message=error_message,
                error_code="SYSTEM_ERROR"
            )
            
    async def get_active_orders(self) -> pd.DataFrame:
        """有効な注文一覧を取得する
        
        Returns:
            pd.DataFrame: 有効な注文一覧
        """
        try:
            await self.browser_manager.launch()
            await self._extract_order_list()
            return self.order_list_df
        except Exception as e:
            print(f"注文一覧の取得中にエラーが発生しました: {e}")
            traceback.print_exc()
            return pd.DataFrame()
            
    async def get_positions(self) -> pd.DataFrame:
        """現在のポジション一覧を取得する
        
        Returns:
            pd.DataFrame: ポジション一覧
        """
        try:
            await self.browser_manager.launch()
            
            # 信用建玉一覧ページに遷移
            named_tab = await self.page_navigator.credit_position()
            await named_tab.tab.utils.wait(1)
            
            # HTMLを取得して解析
            html_content = await named_tab.tab.utils.get_html_content()
            positions_df = self._parse_positions_table(html_content)
            
            return positions_df
        except Exception as e:
            print(f"ポジション一覧の取得中にエラーが発生しました: {e}")
            traceback.print_exc()
            return pd.DataFrame()
            
    # 以下、プライベートヘルパーメソッド
    
    def _get_selector(self, category: str, key: str) -> str:
        """指定されたカテゴリとキーに対応するセレクタを返す"""
        return self.ORDER_PARAM_DICT.get(category, {}).get(key, "")
    
    async def _select_trade_type(self, order_request: OrderRequest) -> None:
        """取引タイプを選択する"""
        named_tab = self.browser_manager.get_tab('SBI')
        
        # 取引タイプを決定
        if order_request.direction == "Long":
            trade_type = "信用新規買" if order_request.is_borrowing_stock else "現物買"
        else:  # Short
            trade_type = "信用新規売"
            
        # セレクタを使用して取引タイプを選択
        selector_id = self._get_selector("取引", trade_type)
        await named_tab.tab.utils.click_element(f'#{selector_id}', is_css=True)
    
    async def _input_stock_and_quantity(self, order_request: OrderRequest) -> None:
        """銘柄コードと数量を入力する"""
        named_tab = self.browser_manager.get_tab('SBI')
        await named_tab.tab.utils.send_keys_to_element('input[name="stock_sec_code"]', 
                                                      is_css=True, 
                                                      keys=order_request.symbol_code)
        await named_tab.tab.utils.send_keys_to_element('input[name="input_quantity"]', 
                                                      is_css=True, 
                                                      keys=str(order_request.unit))
    
    async def _input_order_conditions(self, order_request: OrderRequest) -> None:
        """注文条件を設定（指値/成行/逆指値）"""
        named_tab = self.browser_manager.get_tab('SBI')
        
        # 注文タイプを選択
        await named_tab.tab.utils.click_element(order_request.order_type)
        
        if order_request.order_type == "成行":
            # 成行注文の詳細設定
            if order_request.order_type_value:
                selector = f'select[name="nariyuki_condition"] option[value="{self.ORDER_PARAM_DICT["成行タイプ"][order_request.order_type_value]}"]'
                await named_tab.tab.utils.select_pulldown(selector)
                
        elif order_request.order_type == "指値":
            # 指値注文の詳細設定
            if order_request.limit_price:
                await named_tab.tab.utils.send_keys_to_element('#gsn0 > input[type=text]',
                                                              is_css=True,
                                                              keys=str(order_request.limit_price))
            
            if order_request.order_type_value:
                selector = f'select[name="sasine_condition"] option[value="{self.ORDER_PARAM_DICT["指値タイプ"][order_request.order_type_value]}"]'
                await named_tab.tab.utils.select_pulldown(selector)
                
        elif order_request.order_type == "逆指値":
            # 逆指値注文の詳細設定（実装は省略）
            pass
    
    async def _select_deposit_and_credit_type(self, order_request: OrderRequest) -> None:
        """預り区分と信用取引区分を選択する"""
        named_tab = self.browser_manager.get_tab('SBI')
        
        # 預り区分の選択（固定で「特定預り」を使用）
        await named_tab.tab.utils.click_element("特定預り")
        
        # 信用取引区分の選択
        credit_type = "制度" if order_request.is_borrowing_stock else "日計り"
        await named_tab.tab.utils.click_element(credit_type)
    
    async def _confirm_order(self, order_request: OrderRequest) -> tuple[bool, Optional[str], str]:
        """注文を確定する"""
        named_tab = self.browser_manager.get_tab('SBI')
        
        # 取引パスワードを入力し、確認画面へ進む
        await self._input_trade_pass()
        await named_tab.tab.utils.click_element('input[id="shouryaku"]', is_css=True)
        await named_tab.tab.utils.wait_for('img[title="注文発注"]', is_css=True)
        await named_tab.tab.utils.click_element('img[title="注文発注"]', is_css=True)
        
        # 注文結果を確認
        max_retry = 10
        for _ in range(max_retry):
            try:
                # 成功確認
                await named_tab.tab.utils.wait_for("ご注文を受け付けました。", timeout=2)
                order_id = await self._get_element('注文番号')
                await self.browser_manager.close_popup()
                return True, order_id, "注文が成功しました"
            except:
                pass
                
            try:
                # 失敗確認
                failure_selector = '#MAINAREA02_780 > form > table:nth-child(22) > tbody > tr > td > b > p'
                error_element = await named_tab.tab.utils.wait_for(failure_selector, is_css=True, timeout=2)
                error_message = error_element.text if error_element else "不明なエラー"
                await self.browser_manager.close_popup()
                return False, None, error_message
            except:
                pass
                
        return False, None, "注文処理がタイムアウトしました"
    
    async def _input_trade_pass(self) -> None:
        """取引パスワードを入力する"""
        named_tab = self.browser_manager.get_tab('SBI')
        await named_tab.tab.utils.wait_for('input[id="pwd3"]', is_css=True)
        await named_tab.tab.utils.send_keys_to_element('input[id="pwd3"]',
                                                     is_css=True,
                                                     keys=os.getenv('SBI_TRADEPASS'))
    
    async def _extract_order_list(self) -> None:
        """現在の注文一覧を取得する"""
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
        
        # 取消中の注文を除外
        self.order_list_df = order_list_df[order_list_df["注文状況"] != "取消中"].reset_index(drop=True)
    
    def _extract_order_row_data(self, tr) -> list:
        """注文一覧の行データを抽出する"""
        import unicodedata
        
        row = []
        row.append(tr.findAll("td")[0].getText().strip())  # 注文番号
        row.append(tr.findAll("td")[1].getText().strip())  # 注文状況
        row.append(tr.findAll("td")[2].getText().strip())  # 注文種別
        
        text = unicodedata.normalize("NFKC", tr.findAll("td")[3].getText().strip())
        row.append(text.splitlines()[0].strip().split(" ")[0])  # 銘柄
        row.append(text.splitlines()[0].strip().split(" ")[-1])  # コード

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
            
        return row
    
    def _order_exists(self, order_id: str) -> bool:
        """指定された注文IDの注文が存在するか確認"""
        if self.order_list_df.empty:
            return False
        return str(order_id) in self.order_list_df["注文番号"].astype(str).values
    
    async def _get_element(self, text: str):
        """指定されたテキストを含む要素の値を取得"""
        named_tab = self.browser_manager.get_tab('SBI')
        element = await named_tab.tab.utils.wait_for(text)
        element = element.parent.parent.children[1]
        return re.sub(r'\s+', '', element.text)
    
    async def _get_position_elements(self, table_body_css: str) -> dict:
        """建玉一覧の要素番号を取得"""
        named_tab = self.browser_manager.get_tab('SBI')
        positions = {}
        
        try:
            rows = await named_tab.tab.utils.query_selector(f'{table_body_css} > tr', is_all=True)
            
            for idx, row in enumerate(rows):
                # 行のテキスト内容を取得
                row_text = await row.get_text()
                
                if '返買' in row_text or '返売' in row_text:
                    # 銘柄コードを抽出（実際のHTMLに合わせて調整が必要かもしれません）
                    code_match = re.search(r'(\d{4})', row_text)
                    if code_match:
                        code = code_match.group(1)
                        positions[code] = idx + 1
        except Exception as e:
            print(f"建玉要素の取得中にエラーが発生しました: {e}")
            traceback.print_exc()
            
        return positions
    
    async def _navigate_to_position_settlement(self, element_num: int, table_body_css: str) -> None:
        """指定された建玉の決済画面に遷移"""
        named_tab = self.browser_manager.get_tab('SBI')
        
        # 建玉一覧から該当する行の「返済」リンクをクリック
        await named_tab.tab.utils.click_element(
            f'{table_body_css} > tr:nth-child({element_num}) > td:nth-child(10) > a:nth-child(1) > u > font', 
            is_css=True
        )
    
    def _parse_positions_table(self, html_content: str) -> pd.DataFrame:
            """建玉一覧のHTMLをパースしてデータフレームに変換する
            
            Args:
                html_content (str): 建玉一覧のHTML
                
            Returns:
                pd.DataFrame: 建玉情報のデータフレーム
            """
            import unicodedata
            
            html = soup(html_content, "html.parser")
            table = html.find("td", string=re.compile("銘柄"))
            
            if table is None:
                print('保有建玉はありません。')
                return pd.DataFrame()
                
            table = table.findParent("table")
            data = []
            
            for tr in table.find("tbody").findAll("tr"):
                if tr.find("td").find("a"):
                    row = []
                    
                    # 証券コードと銘柄名
                    text = unicodedata.normalize("NFKC", tr.findAll("td")[0].getText().strip())
                    row.append(text[-4:])  # 証券コード
                    row.append(text[:-4])  # 銘柄名
                    
                    # 売・買建
                    row.append(tr.findAll("td")[1].getText().strip())
                    
                    # 建株数
                    text = unicodedata.normalize("NFKC", tr.findAll("td")[5].getText().strip())
                    row.append(text.splitlines()[0].strip().split(" ")[0])
                    
                    # 建単価と現在値
                    text = unicodedata.normalize("NFKC", tr.findAll("td")[6].getText().strip())
                    numbers = text.split("\n")
                    row.append(numbers[0])  # 建単価
                    row.append(numbers[1])  # 現在値
                    
                    data.append(row)
            
            if not data:
                return pd.DataFrame()
                
            columns = ["証券コード", "銘柄", "売・買建", "建株数", "建単価", "現在値"]
            df = pd.DataFrame(data, columns=columns)
            
            # データ型の変換
            df["証券コード"] = df["証券コード"].astype(str)
            df["建株数"] = df["建株数"].str.replace(',', '').astype(int)
            df["建単価"] = df["建単価"].str.replace(',', '').astype(float)
            df["現在値"] = df["現在値"].str.replace(',', '').astype(float)
            
            # 追加の計算項目
            df["建価格"] = df["建株数"] * df["建単価"]
            df["評価額"] = df["建株数"] * df["現在値"]
            df['評価損益'] = df["評価額"] - df["建価格"]
            df.loc[df['売・買建'] == '売建', '評価損益'] = df["建価格"] - df["評価額"]
            
            return df


    # テスト用コード
    if __name__ == '__main__':
        import asyncio
        
        async def main():
            browser_manager = SBIBrowserManager()
            order_executor = SBIOrderExecutor(browser_manager)