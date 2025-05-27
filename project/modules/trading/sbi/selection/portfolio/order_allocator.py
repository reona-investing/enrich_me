import pandas as pd
from typing import List, Tuple, Literal
from trading.sbi.selection.interface import IOrderAllocator, StockWeight, OrderUnit
from trading.sbi.selection.portfolio.unit_calculator import UnitCalculator
from trading.sbi.selection.portfolio.etf_allocator import ETFAllocator

class OrderAllocator(IOrderAllocator):
    """注文配分クラス"""
    
    def __init__(self, unit_calculator: UnitCalculator, etf_allocator: ETFAllocator):
        """
        Args:
            unit_calculator: 注文単位計算
            etf_allocator: ETF配分
        """
        self.unit_calculator = unit_calculator
        self.etf_allocator = etf_allocator
    
    def allocate_orders(self, 
                       weights: List[StockWeight], 
                       target_sectors: List[str], 
                       tradable_symbols: pd.DataFrame, 
                       margin_power: float, 
                       direction: Literal['Long', 'Short']) -> List[OrderUnit]:
        """注文を配分"""
        # 対象セクターの銘柄を抽出
        weights_df = pd.DataFrame([w.__dict__ for w in weights])
        sector_weights = weights_df[weights_df['Sector'].isin(target_sectors)].copy()
        
        # 取引可能な銘柄に限定
        tradable_symbols['Code'] = tradable_symbols['Code'].astype(str)
        tradable_sector_weights = pd.merge(
            sector_weights, 
            tradable_symbols, 
            on='Code', 
            how='inner'
        )
        
        # セクターごとのウェイト再計算
        tradable_sector_weights['Weight'] = tradable_sector_weights.groupby('Sector')['Weight'].transform(
            lambda x: x / x.sum()
        )
        
        # セクターごとの最大金額計算
        max_cost_per_sector = margin_power / len(target_sectors)
        
        # 各セクターの取引単位を計算
        orders = []
        for sector in target_sectors:
            sector_df = tradable_sector_weights[tradable_sector_weights['Sector'] == sector]
            if not sector_df.empty:
                sector_orders = self.unit_calculator.calculate_units(
                    sector_df,
                    max_cost_per_sector,
                    direction
                )
                orders.extend(sector_orders)
        
        return orders

    def allocate_balanced_orders(self,
                               long_weights: List[StockWeight],
                               short_weights: List[StockWeight],
                               buy_sectors: List[str],
                               sell_sectors: List[str],
                               buyable_symbols: pd.DataFrame,
                               sellable_symbols: pd.DataFrame,
                               margin_power: float) -> Tuple[List[OrderUnit], List[OrderUnit]]:
        """買い・売りのバランスをとった注文を配分"""
        # セクター数の調整を計算
        sectors_to_trade_num = max(len(buy_sectors), len(sell_sectors))
        buy_adjuster_num = len(sell_sectors) - len(buy_sectors) if len(sell_sectors) > len(buy_sectors) else 0
        sell_adjuster_num = len(buy_sectors) - len(sell_sectors) if len(buy_sectors) > len(sell_sectors) else 0
        
        # ETF配分を追加
        long_weights_df = pd.DataFrame([w.__dict__ for w in long_weights])
        short_weights_df = pd.DataFrame([w.__dict__ for w in short_weights])
        
        if buy_adjuster_num > 0 or sell_adjuster_num > 0:
            long_weights_df, short_weights_df = self.etf_allocator.allocate_etfs(
                long_weights_df,
                short_weights_df,
                buy_adjuster_num,
                sell_adjuster_num
            )
        
        # 各側の注文を配分
        long_orders = self.allocate_orders(
            [StockWeight(**row) for _, row in long_weights_df.iterrows()],
            buy_sectors,
            buyable_symbols,
            margin_power / 2,  # 半分ずつ配分
            'Long'
        )
        
        short_orders = self.allocate_orders(
            [StockWeight(**row) for _, row in short_weights_df.iterrows()],
            sell_sectors,
            sellable_symbols,
            margin_power / 2,  # 半分ずつ配分
            'Short'
        )
        
        return long_orders, short_orders