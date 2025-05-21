import math
import pandas as pd
from typing import Literal
from trading.sbi import TradeParameters, MarginManager
from trading.sbi.operations.order_manager import NewOrderManager, SettlementManager, CancelManager
from utils.paths import Paths
from trading.sbi_trading_logic.stock_selector import StockSelector

class OrderMakerBase:
    '''発注用の基底クラス'''
    def __init__(self, 
                 new_order_manager: NewOrderManager = None, 
                 settlement_manager: SettlementManager = None, 
                 cancel_manager: CancelManager = None,
                 margin_manager: MarginManager = None):
        '''
        Args:
            order_manager (object): NewOrderManager, SettlementManager, CancelManagerのいずれかを選択
        '''
        self.new_order_manager = new_order_manager
        self.settlement_manager = settlement_manager
        self.cancel_manager = cancel_manager
        self.margin_manager = margin_manager
        self.remaining_margin = None
        self.failed_orders = []
        self.failed_symbol_codes = []

    async def _make_orders(self, 
                          orders_df: pd.DataFrame, 
                          order_type: Literal["指値", "成行", "逆指値"] = '成行', 
                          order_type_value: Literal["寄指", "引指", "不成", "IOC指", "寄成", "引成", "IOC成", None] = None) -> None:
        '''
        orders_dfに存在する銘柄注文を一括発注します。
        Args:
            orders_df (pd.DataFrame): 注文内容を定義している
            order_type (Literal): 注文タイプ
            order_type_value(Literal): 注文タイプの詳細
        '''
        await self.margin_manager.fetch()
        self.remaining_margin = self.margin_manager.margin_power
        
        for symbol_code, unit, L_or_S, price, is_borrowing_stock, upper_limit_total \
                in zip(orders_df['Code'], 
                       orders_df['Unit'], 
                       orders_df['Direction'], 
                       orders_df['EstimatedCost'], 
                       orders_df['isBorrowingStock'], 
                       orders_df['UpperLimitTotal']):
            margin_trade_section = self._get_margin_trade_section(is_borrowing_stock)
            if upper_limit_total <= self.remaining_margin:
                has_successfully_ordered = await self.make_order(order_type, order_type_value, symbol_code, unit, L_or_S, price, margin_trade_section)
                if has_successfully_ordered:
                    self.remaining_margin -= upper_limit_total
            else:
                trade_params = self.generate_trade_params(order_type, order_type_value, symbol_code, unit, L_or_S, price, margin_trade_section)
                self.new_order_manager.add_position(trade_params)
                print(f'{symbol_code} {int(unit * 100)}株 {L_or_S}: 信用建余力不足のため発注しませんでした。')
                self.failed_symbol_codes.append(symbol_code)

        failed_orders_df = orders_df.loc[orders_df['Code'].isin(self.failed_symbol_codes), :]
        # 発注失敗した銘柄をdfとして保存
        failed_orders_df.to_csv(Paths.FAILED_ORDERS_CSV)

    def _get_margin_trade_section(self, is_borrowing_stock: bool) -> str:
        if is_borrowing_stock == False:
            return "日計り"
        return "制度"

    def generate_trade_params(self, 
                         order_type: Literal["指値", "成行", "逆指値"],
                         order_type_value: Literal["寄指", "引指", "不成", "IOC指", "寄成", "引成", "IOC成", None], 
                         symbol_code: str, unit: int, L_or_S: Literal['Long', 'Short'], price: float,
                         margin_trade_section: Literal["制度", "一般", "日計り"]) -> TradeParameters:
        '''
        TradeParametersインスタンスを作成します。
        Args:
            order_type (Literal): 注文タイプ
            order_type_value(Literal): 注文タイプの詳細
            symbol_code (str): 銘柄コード
            unit (int): 発注単位数
            L_or_S (Literal): "Long"か"Short"かの選択
            price (float): 前日終値
        
        Returns:
            TradeParameters: 取引パラメータを格納したクラス
        '''
        unit = int(unit * 100)
        price /= 100

        order_type = '成行'
        if order_type_value is not None:
            order_type_value = order_type_value.replace('指', '成')
        limit_order_price = None
        trade_type = '信用新規買'
        if (L_or_S == 'Short') and (symbol_code != '1356'):
            trade_type = '信用新規売'
            order_type, order_type_value, limit_order_price = self._avoid_short_selling_restrictions(order_type_value, price)
        order_params = TradeParameters(trade_type=trade_type, symbol_code=symbol_code, unit=unit, order_type=order_type, order_type_value=order_type_value,
                                    limit_order_price=limit_order_price, stop_order_trigger_price=None, stop_order_type="成行", stop_order_price=None,
                                    period_type="当日中", period_value=None, period_index=None, trade_section="特定預り",
                                    margin_trade_section=margin_trade_section)
        return order_params  

    async def make_order(self, 
                         order_type: Literal["指値", "成行", "逆指値"],
                         order_type_value: Literal["寄指", "引指", "不成", "IOC指", "寄成", "引成", "IOC成", None], 
                         symbol_code: str, unit: int, L_or_S: Literal['Long', 'Short'], price: float,
                         margin_trade_section: Literal["制度", "一般", "日計り"]) -> bool:
        '''
        単体注文を発注します。
        Args:
            order_type (Literal): 注文タイプ
            order_type_value(Literal): 注文タイプの詳細
            symbol_code (str): 銘柄コード
            unit (int): 発注単位数
            L_or_S (Literal): "Long"か"Short"かの選択
            price (float): 前日終値
        
        Returns:
            bool: 注文の成否
        '''
        order_params = self.generate_trade_params(order_type, order_type_value, symbol_code, unit, L_or_S, price, margin_trade_section)
        has_successfully_ordered = await self.new_order_manager.place_new_order(order_params)
        if not has_successfully_ordered:
            self.failed_orders.append(f'{order_params.trade_type}: {order_params.symbol_code} {order_params.unit}株')
            self.failed_symbol_codes.append(symbol_code)
        
        return has_successfully_ordered

    def _avoid_short_selling_restrictions(self, 
                                          order_type_value: Literal["寄指", "引指", "不成", "IOC指", "寄成", "引成", "IOC成", None],
                                          price: float):
        '''
        空売り規制に対応します。
        Args:
            order_type_value (Literal): 注文タイプの詳細
            price (float): 前日終値
        Returns:
            str: 注文タイプ（指値に切り替え）
            str: 注文タイプの詳細（成→指に置換）
            float: 指値
        '''
        order_type = '指値'
        
        if order_type_value is not None:
            order_type_value = order_type_value.replace('成', '指')
        limit_order_price = self._set_limit_order_price(price)
        return order_type, order_type_value, limit_order_price

    def _set_limit_order_price(self, cost):
        '''
        呼び値を考慮した指値価格を設定
        Args:
            cost (float): 株価
        Returns:
            str: 呼び値を考慮した指値価格
        '''
        if cost <= 3000:
            return str(math.ceil(cost * 0.905))
        elif cost <= 5000:
            return str(math.ceil(cost * 0.905 / 5) * 5)
        elif cost <= 30000:
            return str(math.ceil(cost * 0.905 / 10) * 10)
        elif cost <= 50000:
            return str(math.ceil(cost * 0.905 / 50) * 50)
        return str(math.ceil(cost * 0.905 / 100) * 100)


class NewOrderMaker(OrderMakerBase):
    def __init__(self, orders_df: pd.DataFrame, new_order_manager: NewOrderManager, margin_manager: MarginManager):
        '''新規発注用のクラス'''
        super().__init__(new_order_manager = new_order_manager, margin_manager = margin_manager)
        self.orders_df = orders_df
        self.new_order_manager = new_order_manager

    async def run_new_orders(self) -> list[dict]:
        '''
        新規注文を発注する。
        returns:
            list[dict]: 発注失敗銘柄のリスト
        '''
        #現時点での注文リストをsbi証券から取得
        await self.new_order_manager.extract_order_list()
        #発注処理の条件に当てはまるときのみ処理実行
        if len(self.new_order_manager.order_list_df) > 0:
            position_list = [x[:2] for x in self.new_order_manager.order_list_df['取引'].unique()]
            #信用新規がある場合のみ注文キャンセル
            if '信新' in position_list:
                return None
        await self._make_orders(orders_df = self.orders_df, order_type = '成行', order_type_value = '寄成')

        return self.failed_orders


class AdditionalOrderMaker(OrderMakerBase):
    def __init__(self, new_order_manager: NewOrderManager, margin_manager: MarginManager):
        '''追加発注用のクラス'''
        super().__init__(new_order_manager = new_order_manager, margin_manager = margin_manager)

    async def run_additional_orders(self) -> list[dict]:
        '''
        新規注文時にエラーだった注文を再発注する。
        returns:
            list[dict]: 発注失敗銘柄のリスト
        '''
        #現時点での注文リストをSBI証券から取得
        orders_df = pd.read_csv(Paths.FAILED_ORDERS_CSV)
        orders_df['Code'] = orders_df['Code'].astype(str)
        #ポジションの発注
        await self._make_orders(orders_df = orders_df, order_type = '成行', order_type_value=None)
        return self.failed_orders

class PositionSettler(OrderMakerBase):
    def __init__(self, settlement_manager: SettlementManager):
        '''決済注文時に起動'''
        super().__init__(settlement_manager = settlement_manager)

    async def settle_all_margins(self):
        '''決済注文を発注する'''
        await self.settlement_manager.settle_all_margins()