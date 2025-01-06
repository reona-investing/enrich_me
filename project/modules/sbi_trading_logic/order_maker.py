import math
import pandas as pd
from typing import Literal, Optional
from sbi import TradeParameters, OrderManager
import paths
from sbi_trading_logic.stock_selector import StockSelector

class OrderMakerBase:
    '''発注用の基底クラス'''
    def __init__(self, order_manager: OrderManager):
        self.order_manager = order_manager
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
        for symbol_code, unit, L_or_S, price, is_borrowing_stock \
                in zip(orders_df['Code'], orders_df['Unit'], orders_df['LorS'], orders_df['EstimatedCost'], orders_df['isBorrowingStock']):
            margin_trade_section = self._get_margin_trade_section(is_borrowing_stock)
            await self.make_order(order_type, order_type_value, symbol_code, unit, L_or_S, price, margin_trade_section)
        failed_orders_df = orders_df.loc[orders_df['Code'].isin(self.failed_symbol_codes), :]
        # 発注失敗した銘柄をdfとして保存
        failed_orders_df.to_csv(paths.FAILED_ORDERS_CSV)
        failed_orders_df.to_csv(paths.FAILED_ORDERS_BACKUP)

    def _get_margin_trade_section(self, is_borrowing_stock: bool) -> str:
        if is_borrowing_stock == False:
            return "日計り"
        return "制度"

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
            #if unit > 5000: # 51単元以上のときは、空売り規制を回避。
            #    print('51単元以上の信用売りは、指値注文で発注されます。')
            #    order_type, order_type_value, limit_order_price = self._avoid_short_selling_restrictions(order_type_value, price)
            order_type, order_type_value, limit_order_price = self._avoid_short_selling_restrictions(order_type_value, price)
        order_params = TradeParameters(trade_type=trade_type, symbol_code=symbol_code, unit=unit, order_type=order_type, order_type_value=order_type_value,
                                    limit_order_price=limit_order_price, stop_order_trigger_price=None, stop_order_type="成行", stop_order_price=None,
                                    period_type="当日中", period_value=None, period_index=None, trade_section="特定預り",
                                    margin_trade_section=margin_trade_section)
        has_successfully_ordered =  await self.order_manager.place_new_order(order_params)
        if not has_successfully_ordered:
            self.failed_orders.append(f'{order_params.trade_type}: {order_params.symbol_code} {order_params.unit}株')
            self.failed_symbol_codes.append(symbol_code)

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
    def __init__(self, stock_selector: StockSelector, order_manager: OrderManager):
        '''新規発注用のクラス'''
        super().__init__(order_manager)
        self.stock_selector = stock_selector
        self.order_manager = order_manager

    async def run_new_orders(self) -> list[dict]:
        '''
        新規注文を発注する。
        returns:
            list[dict]: 発注失敗銘柄のリスト
        '''
        long_orders, short_orders, _ = await self.stock_selector.select()
        #現時点での注文リストをsbi証券から取得
        await self.order_manager.extract_order_list()
        #発注処理の条件に当てはまるときのみ処理実行
        if len(self.order_manager.order_list_df) > 0:
            position_list = [x[:2] for x in self.order_manager.order_list_df['取引'].unique()]
            #信用新規がある場合のみ注文キャンセル
            if '信新' in position_list:
                return None
        orders_df = pd.concat([long_orders, short_orders], axis=0).sort_values('CumCost_byLS', ascending=True)
        await self._make_orders(orders_df = orders_df, order_type = '成行', order_type_value = '寄成')

        return self.failed_orders


class AdditionalOrderMaker(OrderMakerBase):
    def __init__(self, order_manager: OrderManager):
        '''追加発注用のクラス'''
        super().__init__(order_manager)

    async def run_additional_orders(self) -> list[dict]:
        '''
        新規注文時にエラーだった注文を再発注する。
        returns:
            list[dict]: 発注失敗銘柄のリスト
        '''
        #現時点での注文リストをsbi_operations証券から取得
        orders_df = pd.read_csv(paths.FAILED_ORDERS_CSV)
        orders_df['Code'] = orders_df['Code'].astype(str)
        #ポジションの発注
        await self._make_orders(orders_df = orders_df, order_type = '成行', order_type_value=None)
        return self.failed_orders

class PositionSettler(OrderMakerBase):
    def __init__(self, order_manager: OrderManager):
        '''決済注文時に起動'''
        super().__init__(order_manager)

    async def settle_all_margins(self):
        '''決済注文を発注する'''
        await self.order_manager.settle_all_margins()

if __name__ == '__main__':
    async def main():
        from models import MLDataset
        from sbi.operations import TradePossibilityManager, MarginManager
        from sbi.session import LoginHandler
        ml = MLDataset(f'{paths.ML_DATASETS_FOLDER}/New48sectors')
        lh = LoginHandler()
        tpm = TradePossibilityManager(lh)
        mm = MarginManager(lh)
        sd = f'{paths.SECTOR_REDEFINITIONS_FOLDER}/48sectors_2024-2025.csv'
        ss = StockSelector(ml, tpm, mm, sd)
        om = OrderManager(lh)
        nom = NewOrderMaker(ss, om)
        failed_list = await nom.run_new_orders()
    import asyncio
    asyncio.run(main())