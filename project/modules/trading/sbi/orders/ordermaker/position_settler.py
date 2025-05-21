from typing import List

from trading.sbi.interface.orders import OrderResult
from trading.sbi.orders.ordermaker.order_maker import OrderMaker

class PositionSettler(OrderMaker):
    """ポジション決済を行うクラス"""
    
    async def settle_all_positions(self) -> List[OrderResult]:
        """すべてのポジションを決済する"""
        
        # 現在のポジションを取得
        positions = await self.order_executor.get_positions()
        
        if positions.empty:
            return [OrderResult(success=False, message="決済対象のポジションがありません")]
        
        # 現在の注文一覧を取得
        current_orders = await self.order_executor.get_active_orders()
        
        # 既に決済注文を出している銘柄をスキップするロジック
        settled_symbols = set()
        if not current_orders.empty and '信用返済' in current_orders['取引'].values:
            settled_symbols = set(current_orders.loc[current_orders['取引'].str.contains('信用返済'), 'コード'].astype(str).unique())
        
        results = []
        
        # 各ポジションを決済
        for symbol in positions['証券コード'].unique():
            if symbol in settled_symbols:
                results.append(OrderResult(
                    success=False,
                    message=f"{symbol}: 既に決済注文済みです",
                ))
                continue
            
            result = await self.order_executor.settle_position(symbol)
            results.append(result)
        
        return results