from typing import Literal
from trading.sbi.common.interface import IMarginProvider
from trading.sbi.orders.interface.order_executor import IOrderExecutor, OrderRequest

class OrderMaker:
    """注文組立の基底クラス"""
    
    def __init__(self, order_executor: IOrderExecutor, margin_provider: IMarginProvider):
        self.order_executor = order_executor
        self.margin_provider = margin_provider
        self.pending_orders = []
        self.failed_orders = []
    
    def create_order_request(self, 
                            symbol_code: str, 
                            unit: float,  # 単位数（100株単位）
                            direction: Literal['Long', 'Short'],
                            estimated_price: float,
                            is_borrowing_stock: bool = False,
                            **kwargs) -> OrderRequest:
        """注文リクエストを作成する"""
        
        # 取引タイプを決定（常に信用取引）
        if direction == 'Long':
            trade_type = "信用新規買"
        else:  # Short
            trade_type = "信用新規売"
        
        # 信用取引区分を決定
        margin_trade_section = self._get_margin_trade_section(is_borrowing_stock)
        
        # 注文条件を決定
        order_type = kwargs.get('order_type', "成行")
        order_type_value = kwargs.get('order_type_value', "寄成")  # デフォルト値を設定
        limit_price = kwargs.get('limit_price')
        
        # 空売り規制対応（Short かつ ETFでない場合）
        if direction == 'Short' and symbol_code != '1356':
            order_type = "指値"
            if order_type_value:
                order_type_value = order_type_value.replace('成', '指')
            else:
                order_type_value = "寄指"  # デフォルトを寄指に
            limit_price = self._calculate_short_selling_limit_price(estimated_price)
        
        return OrderRequest(
            symbol_code=symbol_code,
            unit=int(unit),
            direction=direction,
            estimated_price=estimated_price,
            is_borrowing_stock=is_borrowing_stock,
            order_type=order_type,
            order_type_value=order_type_value,
            limit_price=limit_price,
            trigger_price=kwargs.get('trigger_price'),
            trade_type=trade_type,
            margin_trade_section=margin_trade_section,
            # リファクタリング前から落ちているパラメータを追加
            stop_order_type=kwargs.get('stop_order_type', "成行"),
            stop_order_price=kwargs.get('stop_order_price'),
            period_type=kwargs.get('period_type', "当日中"),
            period_value=kwargs.get('period_value'),
            period_index=kwargs.get('period_index'),
            trade_section=kwargs.get('trade_section', "特定預り")
        )
    
    def _get_margin_trade_section(self, is_borrowing_stock: bool) -> str:
        """信用取引区分を取得する"""
        return "制度" if is_borrowing_stock else "日計り"
    
    def _calculate_short_selling_limit_price(self, price: float) -> float:
        """空売り制限価格を計算する"""
        import math
        
        if price <= 3000:
            return math.ceil(price * 0.905)
        elif price <= 5000:
            return math.ceil(price * 0.905 / 5) * 5
        elif price <= 30000:
            return math.ceil(price * 0.905 / 10) * 10
        elif price <= 50000:
            return math.ceil(price * 0.905 / 50) * 50
        return math.ceil(price * 0.905 / 100) * 100