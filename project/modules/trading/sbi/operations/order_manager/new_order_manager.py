from trading.sbi.operations.order_manager.order_manager_base import OrderManagerBase
from trading.sbi.operations.position_manager import PositionManager
from trading.sbi.operations.trade_possibility_manager import TradePossibilityManager
from trading.sbi.operations.trade_parameters import TradeParameters
from trading.sbi.browser.sbi_browser_manager import SBIBrowserManager
import traceback

class NewOrderManager(OrderManagerBase):
    """
    発注操作を管理するクラス（新規注文、再発注、決済など）。
    """
    def __init__(self, browser_manager: SBIBrowserManager):
        super().__init__(browser_manager)
        self.browser_manager = browser_manager
        self.position_manager = PositionManager()
        self.trade_possibility_manager = TradePossibilityManager(self.browser_manager)
        self.has_successfully_ordered = False
        self.error_tickers = []

    async def place_new_order(self, trade_params: TradeParameters) -> bool:
        """
        指定された取引パラメータを使用して新規注文を行う。
        """
        try:
            await self.browser_manager.launch()
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


    async def reorder_pending(self) -> None:
        """
        現在のポジションと市場条件に基づいて発注待ちの注文を再発注する。
        """
        await self.browser_manager.launch()
        await self.page_navigator.trade()
        pending_positions = self.position_manager.get_pending_positions()
        for position in pending_positions:
            trade_params = TradeParameters(**position['order_params'])
            await self.place_new_order(trade_params)


    async def _select_trade_type(self, trade_params: TradeParameters) -> None:
        """
        取引タイプを選択する。
        """
        named_tab = self.browser_manager.get_tab('SBI')
        await named_tab.tab.utils.click_element(f'#{self._get_selector("取引", trade_params.trade_type)}', is_css=True)

    async def _input_stock_and_quantity(self, trade_params: TradeParameters) -> None:
        """
        銘柄コードと数量を入力する。
        """
        named_tab = self.browser_manager.get_tab('SBI')
        await named_tab.tab.utils.send_keys_to_element('input[name="stock_sec_code"]', is_css = True, keys = trade_params.symbol_code)
        await named_tab.tab.utils.send_keys_to_element('input[name="input_quantity"]', is_css = True, keys = str(trade_params.unit))

    async def _input_sashinari_params(self, trade_params: TradeParameters) -> None:
        '''
        指値・成行のパラメータを設定します。
        '''
        named_tab = self.browser_manager.get_tab('SBI')
        await named_tab.tab.utils.click_element(trade_params.order_type)
        
        if trade_params.order_type == '成行':
            await self._input_nariyuki_params(trade_params)
        if trade_params.order_type == "指値":
            await self._input_sashine_params(trade_params)
        if trade_params.order_type == "逆指値":
            await self._input_gyakusashine_params(trade_params)

            if trade_params.stop_order_type == "指値":
                await self._input_sashine_stop(trade_params)

    async def _input_nariyuki_params(self, trade_params: TradeParameters) -> None:
        '''成行注文の各パラメータを入力'''
        named_tab = self.browser_manager.get_tab('SBI')
        if trade_params.order_type_value is not None:
            selector = \
                f'select[name="nariyuki_condition"] option[value="{self.order_param_dicts["成行タイプ"][trade_params.order_type_value]}"]'
            await named_tab.tab.utils.select_pulldown(selector)

    async def _input_sashine_params(self, trade_params: TradeParameters) -> None:
        '''指値注文の各パラメータを入力'''
        named_tab = self.browser_manager.get_tab('SBI')
        await named_tab.tab.utils.send_keys_to_element('#gsn0 > input[type=text]',
                                                        is_css = True,
                                                        keys = str(trade_params.limit_order_price))

        if trade_params.order_type_value is not None:
            selector = \
                f'select[name="sasine_condition"] option[value="{self.order_param_dicts["指値タイプ"][trade_params.order_type_value]}"]'
            await named_tab.tab.utils.select_pulldown(selector)

    async def _input_gyakusashine_params(self, trade_params: TradeParameters) -> None:
        '''逆指値注文の各パラメータを入力'''
        named_tab = self.browser_manager.get_tab('SBI')
        await named_tab.tab.utils.click_element(
            '#gsn2 > table > tbody > tr > td:nth-child(2) > label:nth-child(5) > input[type=radio]',
            is_css = True)
        await named_tab.tab.utils.select_element(
            '#gsn2 > table > tbody > tr > td:nth-child(2) > select:nth-child(6)',
            is_css = True)
        await named_tab.tab.utils.send_keys_to_element(
            f'#gsn2 > table > tbody > tr > td:nth-child(2) > select:nth-child(6) > option:nth-child({self.order_param_dicts["逆指値タイプ"][trade_params.stop_order_type]})',
            is_css = True,
            keys = trade_params.stop_order_trigger_price)

    async def _input_sashine_stop(self, trade_params: TradeParameters) -> None:
        '''逆指値注文時の執行金額を指値指定'''
        named_tab = self.browser_manager.get_tab('SBI')
        await named_tab.tab.utils.send_keys_to_element(
            '#gsn2 > table > tbody > tr > td:nth-child(2) > input[type=text]:nth-child(7)',
            is_css = True,
            keys = trade_params.stop_order_price)

    async def _get_duration_params(self, trade_params: TradeParameters) -> None:
        """
        期間のパラメータを入力する。
        """
        named_tab = self.browser_manager.get_tab('SBI')
        await named_tab.tab.utils.click_element(trade_params.period_type)
        if trade_params.period_type == "期間指定":
            if trade_params.period_value is None and trade_params.period_index is None:
                raise ValueError("期間を指定してください。period_value or period_index")
            period_option_div = \
                await named_tab.tab.utils.select_element('select[name="limit_in"]', is_css = True)

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
        '''預り区分と信用取引区分を選択する。'''
        named_tab = self.browser_manager.get_tab('SBI')
        await named_tab.tab.utils.click_element(trade_params.trade_section)
        await named_tab.tab.utils.click_element(trade_params.margin_trade_section)

    async def _confirm_order(self, trade_params: TradeParameters) -> bool:
        '''注文を確定する。'''
        await self._send_order()
        text_to_show = f'{trade_params.symbol_code} {trade_params.trade_type} {trade_params.unit}株:'
        order_index = self._append_trade_params_to_orders(trade_params)

        max_retry = 10
        for _ in range(max_retry):
            try: 
                true = await self._order_success(text_to_show, order_index)
                return true
            except TimeoutError:
                pass
            except Exception as e:
                print(f"エラーが発生しました: {e}")

            try:
                false = await self._order_failure(text_to_show, trade_params)
                return false
            except TimeoutError:
                pass
            except Exception as e:
                print(f"エラーが発生しました: {e}")
        print(f"{text_to_show} 注文が失敗しました")       
        self.error_tickers.append(trade_params.symbol_code)
        return False                

    async def _order_success(self, text_to_show: str, order_index: int) -> bool:
        '''
        注文成功時の処理
        
        Args:
            text_to_show (str): 銘柄コードや株数を記載した表示用テキスト
            order_index (int): 対象注文のPositionManager上での順番

        Return:
            bool: 発注成功（True）
        '''
        named_tab = self.browser_manager.get_tab('SBI')
        success_text = "ご注文を受け付けました。"
        await named_tab.tab.utils.wait_for(success_text, timeout=2)
        print(f"{text_to_show} 注文が成功しました")
        await self._edit_position_manager_for_order(order_index)
        await self.browser_manager.close_popup()
        return True

    async def _order_failure(self, text_to_show: str, trade_params: TradeParameters) -> bool:
        '''
        注文成功時の処理
        
        Args:
            text_to_show (str): 銘柄コードや株数を記載した表示用テキスト
            trade_params (TradeParameters): 再発注のためにエラー発生を記録するオブジェクト

        Return:
            bool: 発注失敗（False）
        '''
        named_tab = self.browser_manager.get_tab('SBI')
        failure_selector = '#MAINAREA02_780 > form > table:nth-child(22) > tbody > tr > td > b > p'
        await named_tab.tab.utils.wait_for(failure_selector, is_css=True, timeout=2)
        print(f"{text_to_show} 注文が失敗しました")       
        self.error_tickers.append(trade_params.symbol_code)
        await self.browser_manager.close_popup()
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