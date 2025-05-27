import pandas as pd
from typing import List, Tuple, Optional
import os
from utils.paths import Paths
from trading.sbi.selection.interface import (
    IStockSelector, ISectorProvider, IPriceProvider, ITradeLimitProvider,
    ISectorAnalyzer, IWeightCalculator, IOrderAllocator, OrderUnit
)
from trading.sbi.selection.analyzer.tradability_analyzer import TradabilityAnalyzer

class StockSelector(IStockSelector):
    """銘柄選択クラス"""
    
    def __init__(self,
                pred_result_df: pd.DataFrame,
                sector_provider: ISectorProvider,
                price_provider: IPriceProvider,
                trade_limit_provider: ITradeLimitProvider,
                sector_analyzer: ISectorAnalyzer,
                weight_calculator: IWeightCalculator,
                order_allocator: IOrderAllocator,
                num_sectors_to_trade: int = 3,
                num_candidate_sectors: int = 5,
                top_slope: float = 1.0):
        """
        Args:
            pred_result_df: 予測結果データフレーム
            sector_provider: セクター情報プロバイダー
            price_provider: 価格情報プロバイダー
            trade_limit_provider: 取引制限情報プロバイダー
            sector_analyzer: セクター分析
            weight_calculator: ウェイト計算
            order_allocator: 注文配分
            num_sectors_to_trade: 取引するセクター数
            num_candidate_sectors: 候補セクター数
            top_slope: トップセクターの傾斜
        """
        self.pred_result_df = pred_result_df
        self.sector_provider = sector_provider
        self.price_provider = price_provider
        self.trade_limit_provider = trade_limit_provider
        self.sector_analyzer = sector_analyzer
        self.weight_calculator = weight_calculator
        self.order_allocator = order_allocator
        self.tradability_analyzer = TradabilityAnalyzer(sector_provider, trade_limit_provider)
        self.num_sectors_to_trade = num_sectors_to_trade
        self.num_candidate_sectors = num_candidate_sectors
        self.top_slope = top_slope
        self.buy_sectors = []
        self.sell_sectors = []
    
    async def select_stocks(self, margin_power: Optional[float] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """取引銘柄を選択"""
        print("[INFO] StockSelector.select_stocks を開始します")
        
        # 1. データの準備
        sector_definitions = self.sector_provider.get_sector_definitions()
        price_data = self.price_provider.get_price_data()
        todays_pred_df = self._get_todays_pred_df()
        
        # 2. 取引可能性分析
        todays_pred_with_tradability = await self.tradability_analyzer.analyze_sector_tradability(todays_pred_df)
        
        # 3. セクター分析と選択
        sector_predictions = self.sector_analyzer.analyze_sector_predictions(todays_pred_with_tradability)
        buy_sectors, sell_sectors = self.sector_analyzer.select_sectors_to_trade(
            sector_predictions, 
            self.num_sectors_to_trade, 
            self.num_candidate_sectors
        )
        self.buy_sectors = buy_sectors
        self.sell_sectors = sell_sectors
        
        print('本日のロング予想業種:', buy_sectors)
        print('本日のショート予想業種:', sell_sectors)
        
        # 4. 銘柄ウェイト計算
        stock_weights = self.weight_calculator.calculate_weights(sector_definitions, price_data)
        
        # 5. 買い・売り可能銘柄の取得 - 一度だけ取得して再利用
        print("[INFO] 買い・売り可能銘柄を取得します")
        buyable_symbols = await self.trade_limit_provider.get_buyable_symbols()
        sellable_symbols = await self.trade_limit_provider.get_sellable_symbols()
        
        # 6. 証拠金取得 - 一度だけ取得
        if margin_power is None:
            print("[INFO] 証拠金を取得します")
            margin_power = await self.trade_limit_provider.get_margin_power()
        print(f'利用可能証拠金: {margin_power:,}円')
        
        # 7. 注文配分
        long_weights = [w for w in stock_weights if w.Sector in buy_sectors]
        short_weights = [w for w in stock_weights if w.Sector in sell_sectors]
        
        long_orders, short_orders = self.order_allocator.allocate_balanced_orders(
            long_weights,
            short_weights,
            buy_sectors,
            sell_sectors,
            buyable_symbols,
            sellable_symbols,
            margin_power
        )
        
        # 8. 注文をデータフレームに変換
        orders_df = self._convert_orders_to_dataframe(long_orders, short_orders)
        
        # 9. 注文をCSVに保存
        self._save_orders_to_csv(orders_df)
        
        return orders_df, todays_pred_with_tradability
    
    def _get_todays_pred_df(self) -> pd.DataFrame:
        """最新日の予測結果を取得"""
        pred_df = self.pred_result_df.copy()
        pred_df = pred_df.reset_index().set_index('Date', drop=True)
        latest_date = pred_df.index.max()
        todays_pred = pred_df[pred_df.index == latest_date].sort_values('Pred')
        return todays_pred
    
    def _convert_orders_to_dataframe(self, long_orders: List[OrderUnit], short_orders: List[OrderUnit]) -> pd.DataFrame:
        """注文をデータフレームに変換"""
        long_df = pd.DataFrame([order.__dict__ for order in long_orders])
        short_df = pd.DataFrame([order.__dict__ for order in short_orders])
        
        # 必要なカラムを確保
        for df in [long_df, short_df]:
            if not df.empty:
                # 累積コスト列を追加
                df['CumCost_byLS'] = df['TotalCost'].cumsum()
        
        # データフレームの結合
        if long_df.empty and short_df.empty:
            return pd.DataFrame()
        elif long_df.empty:
            return short_df
        elif short_df.empty:
            return long_df
        else:
            orders_df = pd.concat([long_df, short_df], ignore_index=True)
            orders_df = orders_df.sort_values('CumCost_byLS', ascending=True)
            return orders_df
    
    def _save_orders_to_csv(self, orders_df: pd.DataFrame) -> None:
        """注文をCSVに保存"""
        if not orders_df.empty:
            orders_df.to_csv(Paths.ORDERS_CSV, index=False)
            print(f'注文をCSVに保存しました: {Paths.ORDERS_CSV}')