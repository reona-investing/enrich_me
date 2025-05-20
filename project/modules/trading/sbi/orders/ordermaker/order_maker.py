from typing import Literal
from trading.sbi.orders.interface import IOrderExecutor, IMarginProvider, OrderRequest

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
        # 注文条件を決定するロジック
        # 現在のgenerate_trade_paramsの機能を移植するが、
        # ブラウザ操作に直接依存しないよう修正
        
        # 空売り規制対応（必要に応じて）
        if direction == 'Short' and symbol_code != '1356':
            limit_price = self._calculate_short_selling_limit_price(estimated_price)
            return OrderRequest(
                symbol_code=symbol_code,
                unit=int(unit),
                direction=direction,
                estimated_price=estimated_price,
                is_borrowing_stock=is_borrowing_stock,
                order_type="指値",
                order_type_value="寄指" if kwargs.get('order_type_value') else None,
                limit_price=limit_price
            )
        
        return OrderRequest(
            symbol_code=symbol_code,
            unit=int(unit),
            direction=direction,
            estimated_price=estimated_price,
            is_borrowing_stock=is_borrowing_stock,
            order_type=kwargs.get('order_type', "成行"),
            order_type_value=kwargs.get('order_type_value'),
            limit_price=kwargs.get('limit_price')
        )
    
    def _calculate_short_selling_limit_price(self, price: float) -> str:
        """空売り制限価格を計算する"""
        # 現在の_set_limit_order_priceをそのまま利用
        import math
        
        if price <= 3000:
            return str(math.ceil(price * 0.905))
        elif price <= 5000:
            return str(math.ceil(price * 0.905 / 5) * 5)
        elif price <= 30000:
            return str(math.ceil(price * 0.905 / 10) * 10)
        elif price <= 50000:
            return str(math.ceil(price * 0.905 / 50) * 50)
        return str(math.ceil(price * 0.905 / 100) * 100)
    
    def _get_margin_trade_section(self, is_borrowing_stock: bool) -> str:
        """信用取引区分を取得する"""
        return "制度" if is_borrowing_stock else "日計り"