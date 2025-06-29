import pandas as pd
from typing import List

from trading.sbi.orders.ordermaker.order_maker import OrderMaker
from trading.sbi.orders.interface.order_executor import OrderResult

class BatchOrderMaker(OrderMaker):
    """一括注文を行うクラス"""

    async def place_batch_orders(self, orders_df: pd.DataFrame) -> List[OrderResult]:
        """一括で注文を発注する"""

        # 前回の失敗注文をリセット
        self.failed_orders = []

        # 証拠金を更新
        await self.margin_provider.refresh()
        remaining_margin = await self.margin_provider.get_available_margin()
        
        results = []
        
        # 現在の注文一覧を取得
        current_orders = await self.order_executor.get_active_orders()
        
        # 既に注文している銘柄をスキップするロジック
        existing_symbols = set()
        if not current_orders.empty:
            existing_symbols = set(current_orders['コード'].astype(str).unique())
        
        # 各注文を処理
        for _, row in orders_df.iterrows():
            symbol_code = str(row['Code'])
            
            # 既に注文済みの銘柄はスキップ
            if symbol_code in existing_symbols:
                message = f"{symbol_code}: 既に注文済みです"
                results.append(OrderResult(
                    success=False,
                    message=message,
                ))
                print(message)
                continue
            
            # 証拠金が不足している場合はスキップ
            if row['UpperLimitTotal'] > remaining_margin:
                message = f"{symbol_code}: 証拠金不足のため発注しませんでした"
                results.append(OrderResult(
                    success=False,
                    message=message,
                ))
                print(message)
                self.failed_orders.append(symbol_code)
                continue
            
            order_request = self.create_order_request(
                symbol_code=symbol_code,
                unit=row['Unit'] * 100,
                direction=row['Direction'],
                estimated_price=row['EstimatedCost'] / 100,  # 単価を計算
                is_borrowing_stock=bool(row.get("isBorrowingStock", False)),
                order_type='成行',
                period_type="当日中",
                trade_section="特定預り"
            )
            
            # 注文を発注
            result = await self.order_executor.place_order(order_request)
            results.append(result)
            print(result.message)
            
            if result.success:
                remaining_margin -= row['UpperLimitTotal']
            else:
                self.failed_orders.append(symbol_code)
        
        # 失敗した注文を保存
        self._save_failed_orders(orders_df)
        
        return results
    
    def _save_failed_orders(self, orders_df: pd.DataFrame) -> None:
        """失敗した注文をCSVに保存する"""
        from utils.paths import Paths
        
        if not self.failed_orders:
            return
        self.failed_orders = list(map(str, self.failed_orders))
        failed_orders_df = orders_df.loc[orders_df['Code'].astype(str).isin(self.failed_orders), :]
        # インデックス列が不要なため、index=Falseで保存する
        failed_orders_df.to_csv(Paths.FAILED_ORDERS_CSV, index=False)
