# trading/stock_selection/facade.py
from typing import Optional, Tuple
import pandas as pd

from trading.sbi import TradePossibilityManager
from utils.paths import Paths
from trading.sbi.common.interface import IMarginProvider
from trading.sbi.common.provider import SBIMarginProvider

from trading.sbi.selection.provider import SectorProvider, PriceProvider, TradeLimitProvider
from trading.sbi.selection.analyzer import WeightCalculator, SectorAnalyzer
from trading.sbi.selection.portfolio import UnitCalculator, ETFAllocator, OrderAllocator
from trading.sbi.selection.selector import StockSelector

class OneStopStockSelector:
    """
    銘柄選択プロセスのためのファサードクラス。
    複数のコンポーネントの初期化と連携を簡略化します。
    """
    
    def __init__(self, 
                order_price_df: pd.DataFrame,
                pred_result_df: pd.DataFrame,
                browser_manager,
                sector_definitions_path: Optional[str] = None,
                num_sectors_to_trade: int = 3,
                num_candidate_sectors: int = 5,
                top_slope: float = 1.0):
        """
        Args:
            order_price_df: 価格データ
            pred_result_df: 予測結果データ
            browser_manager: ブラウザマネージャー
            sector_definitions_path: セクター定義ファイルパス (省略時はデフォルト)
            num_sectors_to_trade: 取引するセクター数
            num_candidate_sectors: 候補セクター数
            top_slope: トップセクターの傾斜
        """
        # デフォルト値の設定
        if sector_definitions_path is None:
            raise ValueError(f'引数"sector_definitions_path"が設定されていません。')
        
        # SBI関連のマネージャー
        self.browser_manager = browser_manager
        self.trade_possibility_manager = TradePossibilityManager(browser_manager)

        # プロバイダー
        self.margin_provider: IMarginProvider = SBIMarginProvider(browser_manager)
        self.sector_provider = SectorProvider(sector_definitions_path)
        self.price_provider = PriceProvider(order_price_df)
        self.trade_limit_provider = TradeLimitProvider(
            self.trade_possibility_manager, 
            self.margin_provider
        )
        
        # アナライザー
        self.weight_calculator = WeightCalculator()
        self.sector_analyzer = SectorAnalyzer()
        
        # ポートフォリオ計算
        self.unit_calculator = UnitCalculator()
        self.etf_allocator = ETFAllocator(self.price_provider)
        self.order_allocator = OrderAllocator(
            self.unit_calculator, 
            self.etf_allocator
        )
        
        # セレクター
        self.stock_selector = StockSelector(
            pred_result_df=pred_result_df,
            sector_provider=self.sector_provider,
            price_provider=self.price_provider,
            trade_limit_provider=self.trade_limit_provider,
            sector_analyzer=self.sector_analyzer,
            weight_calculator=self.weight_calculator,
            order_allocator=self.order_allocator,
            num_sectors_to_trade=num_sectors_to_trade,
            num_candidate_sectors=num_candidate_sectors,
            top_slope=top_slope
        )
    
    async def select_stocks(self, margin_power: Optional[float] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        銘柄選択プロセスを実行し、注文データフレームと予測データフレームを返します。
        
        Args:
            margin_power: 使用する証拠金額（省略時は自動取得）
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (注文データフレーム, 予測データフレーム)
        """
        print("[INFO] OneStopStockSelector.select_stocks を開始します")
        
        try:
            # fetch 回数カウント用のモンキーパッチ (デバッグ用)
            original_fetch = self.trade_possibility_manager.fetch
            fetch_count = 0
            
            async def fetch_with_counter(*args, **kwargs):
                nonlocal fetch_count
                fetch_count += 1
                print(f"[DEBUG] TradePossibilityManager.fetch() が {fetch_count} 回目呼び出されました")
                return await original_fetch(*args, **kwargs)
            
            # モンキーパッチ適用
            self.trade_possibility_manager.fetch = fetch_with_counter

            result = await self.stock_selector.select_stocks(margin_power)
            
            # モンキーパッチを戻す
            self.trade_possibility_manager.fetch = original_fetch
            
            print(f"[INFO] fetch() 呼び出し回数: {fetch_count}")
            return result
            
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame(), pd.DataFrame()
    
    @property
    def buy_sectors(self) -> list:
        """選択された買いセクターのリストを取得"""
        return self.stock_selector.buy_sectors
    
    @property
    def sell_sectors(self) -> list:
        """選択された売りセクターのリストを取得"""
        return self.stock_selector.sell_sectors